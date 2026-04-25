from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import (
    KEY_VOCAB,
    SessionSample,
    apply_scalers,
    build_user_index,
    fit_feature_scalers,
    merge_all_sources,
    split_users,
)
from dataset import BehavioralDataset
from model import TriBranchLSTM

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DB = ROOT.parent.parent / "fastapiCollector" / "sentinel_lab.db"
DEFAULT_DATASETS_DIR = ROOT / "datasets"


def _subset(user_data: Dict[str, List[SessionSample]], users: List[str]) -> Dict[str, List[SessionSample]]:
    return {u: user_data[u] for u in users if u in user_data}


def _batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def evaluate(model, loader, device):
    model.eval()
    triplet_loss_fn = nn.TripletMarginLoss(margin=0.3)
    bce_loss_fn = nn.BCELoss()

    losses = []
    similarities = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)

            a_dna, a_risk = model(
                batch["anchor_keys"],
                batch["anchor_scrolls"],
                batch["anchor_imu"],
                key_mask=batch["anchor_key_mask"],
                scroll_mask=batch["anchor_scroll_mask"],
                imu_mask=batch["anchor_imu_mask"],
            )
            p_dna, _ = model(
                batch["positive_keys"],
                batch["positive_scrolls"],
                batch["positive_imu"],
                key_mask=batch["positive_key_mask"],
                scroll_mask=batch["positive_scroll_mask"],
                imu_mask=batch["positive_imu_mask"],
            )
            n_dna, _ = model(
                batch["negative_keys"],
                batch["negative_scrolls"],
                batch["negative_imu"],
                key_mask=batch["negative_key_mask"],
                scroll_mask=batch["negative_scroll_mask"],
                imu_mask=batch["negative_imu_mask"],
            )

            triplet = triplet_loss_fn(a_dna, p_dna, n_dna)

            owner_target = torch.zeros_like(a_risk)
            imp_target = torch.ones_like(a_risk)

            pos_risk = model(
                batch["positive_keys"],
                batch["positive_scrolls"],
                batch["positive_imu"],
                key_mask=batch["positive_key_mask"],
                scroll_mask=batch["positive_scroll_mask"],
                imu_mask=batch["positive_imu_mask"],
            )[1]
            neg_risk = model(
                batch["negative_keys"],
                batch["negative_scrolls"],
                batch["negative_imu"],
                key_mask=batch["negative_key_mask"],
                scroll_mask=batch["negative_scroll_mask"],
                imu_mask=batch["negative_imu_mask"],
            )[1]

            bce_owner = bce_loss_fn(a_risk, owner_target) + bce_loss_fn(pos_risk, owner_target)
            bce_imp = bce_loss_fn(neg_risk, imp_target)
            bce = 0.5 * (bce_owner + bce_imp)

            total = triplet + 0.5 * bce
            losses.append(float(total.item()))

            sim_pos = torch.sum(a_dna * p_dna, dim=1).cpu().numpy()
            sim_neg = torch.sum(a_dna * n_dna, dim=1).cpu().numpy()

            similarities.extend(sim_pos.tolist())
            labels.extend([1] * len(sim_pos))
            similarities.extend(sim_neg.tolist())
            labels.extend([0] * len(sim_neg))

    eer = compute_eer(np.array(labels), np.array(similarities))
    return float(np.mean(losses) if losses else 0.0), float(eer)


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(labels) == 0:
        return 1.0

    thresholds = np.linspace(-1.0, 1.0, 400)
    best_gap = 1.0
    best_eer = 1.0

    pos = labels == 1
    neg = labels == 0

    for th in thresholds:
        far = np.mean(scores[neg] >= th) if np.any(neg) else 1.0
        frr = np.mean(scores[pos] < th) if np.any(pos) else 1.0
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap = gap
            best_eer = (far + frr) / 2.0

    return float(best_eer)


def _phase_train(
    model: TriBranchLSTM,
    phase_name: str,
    all_data: Dict[str, List[SessionSample]],
    user_to_idx: Dict[str, int],
    mode: str,
    epochs: int,
    lr: float,
    patience: int,
    batch_size: int,
    device,
):
    eligible = [u for u, sessions in all_data.items() if sessions]
    phase_pool: Dict[str, List[SessionSample]] = {}
    for uid in eligible:
        s0 = all_data[uid][0]
        if mode == "pretrain_keys" and any(s.has_keystroke for s in all_data[uid]):
            phase_pool[uid] = all_data[uid]
        elif mode == "pretrain_scrolls" and any(s.has_scroll for s in all_data[uid]):
            phase_pool[uid] = all_data[uid]
        elif mode == "finetune_all" and s0.source == "booth":
            phase_pool[uid] = all_data[uid]

    if len(phase_pool) < 2:
        print(f"[WARN] Skipping {phase_name}; not enough users for mode={mode}.")
        return {"skipped": True}

    train_users, val_users, test_users = split_users(phase_pool)
    train_data = _subset(phase_pool, train_users)
    val_data = _subset(phase_pool, val_users) or train_data
    test_data = _subset(phase_pool, test_users) or train_data

    train_ds = BehavioralDataset(train_data, user_to_idx, mode=mode, triplets_per_user=50, is_training=True)
    val_ds = BehavioralDataset(val_data, user_to_idx, mode=mode, triplets_per_user=20, is_training=False)
    test_ds = BehavioralDataset(test_data, user_to_idx, mode=mode, triplets_per_user=20, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cpu")

    triplet_loss_fn = nn.TripletMarginLoss(margin=0.3)
    bce_loss_fn = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)

    best_val_eer = 1.0
    best_state = None
    no_improve = 0

    history = {
        "phase": phase_name,
        "mode": mode,
        "train_loss": [],
        "val_loss": [],
        "val_eer": [],
        "test_eer": None,
        "split": {
            "train_users": train_users,
            "val_users": val_users,
            "test_users": test_users,
        },
    }

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            batch = _batch_to_device(batch, device)

            optimizer.zero_grad()

            a_dna, a_risk = model(
                batch["anchor_keys"],
                batch["anchor_scrolls"],
                batch["anchor_imu"],
                key_mask=batch["anchor_key_mask"],
                scroll_mask=batch["anchor_scroll_mask"],
                imu_mask=batch["anchor_imu_mask"],
            )
            p_dna, p_risk = model(
                batch["positive_keys"],
                batch["positive_scrolls"],
                batch["positive_imu"],
                key_mask=batch["positive_key_mask"],
                scroll_mask=batch["positive_scroll_mask"],
                imu_mask=batch["positive_imu_mask"],
            )
            n_dna, n_risk = model(
                batch["negative_keys"],
                batch["negative_scrolls"],
                batch["negative_imu"],
                key_mask=batch["negative_key_mask"],
                scroll_mask=batch["negative_scroll_mask"],
                imu_mask=batch["negative_imu_mask"],
            )

            triplet = triplet_loss_fn(a_dna, p_dna, n_dna)

            owner_target = torch.zeros_like(a_risk)
            imp_target = torch.ones_like(n_risk)
            bce_owner = bce_loss_fn(a_risk, owner_target) + bce_loss_fn(p_risk, owner_target)
            bce_imp = bce_loss_fn(n_risk, imp_target)
            bce = 0.5 * (bce_owner + bce_imp)

            loss = triplet + 0.5 * bce
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.item()))

        train_loss = float(np.mean(epoch_losses) if epoch_losses else 0.0)
        val_loss, val_eer = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_eer"].append(val_eer)

        print(f"[{phase_name}] epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_eer={val_eer:.4f}")

        if val_eer < best_val_eer:
            best_val_eer = val_eer
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print("Early stopping triggered.")
            break

    if best_state is None:
        best_state = model.state_dict()

    model.load_state_dict(best_state)
    test_loss, test_eer = evaluate(model, test_loader, device)
    history["test_eer"] = float(test_eer)
    history["test_loss"] = float(test_loss)

    return history, best_state


def _set_phase_freeze(model: TriBranchLSTM, phase: str) -> None:
    for p in model.parameters():
        p.requires_grad = True

    if phase == "pretrain_keys":
        for p in model.scroll_lstm.parameters():
            p.requires_grad = False
        for p in model.imu_lstm.parameters():
            p.requires_grad = False
    elif phase == "pretrain_scrolls":
        for p in model.key_embed.parameters():
            p.requires_grad = False
        for p in model.key_lstm.parameters():
            p.requires_grad = False
        for p in model.imu_lstm.parameters():
            p.requires_grad = False
    elif phase == "finetune_all":
        pass
    else:
        raise ValueError(f"Unknown training phase: {phase}")


def run_training(
    db_path: Path = DEFAULT_DB,
    datasets_dir: Path = DEFAULT_DATASETS_DIR,
    phase: str = "all",
    batch_size: int = 16,
    lr: float = 1e-3,
):
    device = torch.device("cpu")

    all_data = merge_all_sources(db_path, datasets_dir)
    if len(all_data) < 2:
        raise RuntimeError("Not enough users after merging datasets.")

    scalers = fit_feature_scalers(all_data)
    scaled_all = apply_scalers(all_data, scalers)
    user_to_idx = build_user_index(scaled_all)

    model = TriBranchLSTM(key_vocab_size=len(KEY_VOCAB)).to(device)

    phase_plan = []
    if phase == "all":
        phase_plan = [
            ("pretrain_keys", "pretrain_keys", 20, 5),
            ("pretrain_scrolls", "pretrain_scrolls", 20, 5),
            ("finetune_all", "finetune_all", 50, 10),
        ]
    elif phase in {"pretrain_keys", "pretrain_scrolls", "finetune_all"}:
        epochs = 20 if phase != "finetune_all" else 50
        patience = 5 if phase != "finetune_all" else 10
        phase_plan = [(phase, phase, epochs, patience)]
    else:
        raise ValueError("--phase must be one of: all, pretrain_keys, pretrain_scrolls, finetune_all")

    logs = {"phases": []}
    last_state = None

    for phase_name, mode, epochs, patience in phase_plan:
        _set_phase_freeze(model, phase_name)
        result = _phase_train(
            model=model,
            phase_name=phase_name,
            all_data=scaled_all,
            user_to_idx=user_to_idx,
            mode=mode,
            epochs=epochs,
            lr=lr,
            patience=patience,
            batch_size=batch_size,
            device=device,
        )

        if isinstance(result, dict) and result.get("skipped"):
            logs["phases"].append({"phase": phase_name, "skipped": True})
            continue

        hist, state = result
        logs["phases"].append(hist)
        last_state = state
        model.load_state_dict(state)

    if last_state is None:
        raise RuntimeError("All phases skipped. Ensure at least two users per selected phase.")

    torch.save(last_state, CHECKPOINT_DIR / "best_model.pt")
    with open(CHECKPOINT_DIR / "scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)
    with open(CHECKPOINT_DIR / "key_vocab.json", "w", encoding="utf-8") as f:
        json.dump(KEY_VOCAB, f)
    with open(CHECKPOINT_DIR / "training_log.json", "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    print(f"Saved model and artifacts to {CHECKPOINT_DIR}")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all", choices=["all", "pretrain_keys", "pretrain_scrolls", "finetune_all"])
    parser.add_argument("--db-path", default=str(DEFAULT_DB))
    parser.add_argument("--datasets-dir", default=str(DEFAULT_DATASETS_DIR))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_training(
        db_path=Path(args.db_path),
        datasets_dir=Path(args.datasets_dir),
        phase=args.phase,
        batch_size=args.batch_size,
        lr=args.lr,
    )
