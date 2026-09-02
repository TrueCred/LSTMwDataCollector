"""Training script for BehavioralLSTM — verification with triplet loss."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import (
    load_raw_enrollment, load_keyrecs_fixed, window_user_data,
    fit_scalers_from_samples, save_scalers, save_key_vocab
)
from dataset import VerificationDataset, split_users
from model import BehavioralLSTM


def compute_eer(genuine_scores: list, impostor_scores: list) -> float:
    """
    Compute Equal Error Rate from genuine and impostor similarity scores.
    
    Args:
        genuine_scores: Cosine similarities for same-user pairs
        impostor_scores: Cosine similarities for different-user pairs
        
    Returns:
        EER value (lower is better, 0 = perfect)
    """
    genuine = np.array(genuine_scores)
    impostor = np.array(impostor_scores)
    
    thresholds = np.linspace(0, 1, 1000)
    
    min_diff = float("inf")
    eer = 0.5
    
    for t in thresholds:
        frr = np.mean(genuine < t)    # False Reject: genuine pairs rejected
        far = np.mean(impostor >= t)  # False Accept: impostors accepted
        
        diff = abs(frr - far)
        if diff < min_diff:
            min_diff = diff
            eer = (frr + far) / 2
    
    return float(eer)


def extract_all_embeddings(
    model: BehavioralLSTM,
    user_samples: dict,
    scalers_dict: dict,
    device: torch.device
) -> dict:
    """Extract DNA embeddings for all users' samples (no augmentation)."""
    from data_loader import preprocess_keystrokes, preprocess_scrolls, extract_imu_stats
    
    model.eval()
    user_embeddings = {}
    
    with torch.no_grad():
        for uid, samples in user_samples.items():
            embeddings = []
            for sample in samples[:20]:  # cap per user for speed
                ks = preprocess_keystrokes(sample["keystrokes"], scalers_dict.get("keystrokes"))
                sc = preprocess_scrolls(sample["scrolls"], scalers_dict.get("scrolls"))
                imu = extract_imu_stats(sample["imu"], scalers_dict.get("imu"))
                
                ks_t = torch.from_numpy(ks).float().to(device)
                sc_t = torch.from_numpy(sc).float().to(device)
                imu_t = torch.from_numpy(imu).float().to(device)
                
                dna = model(ks_t, sc_t, imu_t, return_dna=True)
                embeddings.append(dna.cpu().numpy()[0])
            
            user_embeddings[uid] = np.array(embeddings)
    
    return user_embeddings


def evaluate_verification(user_embeddings: dict) -> tuple[float, float, float]:
    """
    Evaluate verification performance.
    
    Returns: (eer, mean_genuine_sim, mean_impostor_sim)
    """
    user_ids = list(user_embeddings.keys())
    genuine_scores = []
    impostor_scores = []
    
    for i, uid in enumerate(user_ids):
        embs = user_embeddings[uid]
        if len(embs) < 2:
            continue
        
        # Genuine: pairs within same user
        sims = cosine_similarity(embs)
        for r in range(len(embs)):
            for c in range(r + 1, len(embs)):
                genuine_scores.append(sims[r, c])
        
        # Impostor: compare with other users
        for j in range(i + 1, len(user_ids)):
            other_embs = user_embeddings[user_ids[j]]
            cross_sims = cosine_similarity(embs, other_embs)
            impostor_scores.extend(cross_sims.flatten().tolist())
    
    if not genuine_scores or not impostor_scores:
        return 0.5, 0.0, 0.0
    
    eer = compute_eer(genuine_scores, impostor_scores)
    return eer, np.mean(genuine_scores), np.mean(impostor_scores)


def train_epoch(
    model: BehavioralLSTM,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.TripletMarginLoss,
    device: torch.device
) -> float:
    """Train one epoch with triplet loss. Returns avg loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in loader:
        # Extract anchor, positive, negative
        a_ks = batch["anchor_keystrokes"].to(device)
        a_sc = batch["anchor_scrolls"].to(device)
        a_imu = batch["anchor_imu_stats"].to(device)
        
        p_ks = batch["pos_keystrokes"].to(device)
        p_sc = batch["pos_scrolls"].to(device)
        p_imu = batch["pos_imu_stats"].to(device)
        
        n_ks = batch["neg_keystrokes"].to(device)
        n_sc = batch["neg_scrolls"].to(device)
        n_imu = batch["neg_imu_stats"].to(device)
        
        optimizer.zero_grad()
        
        # Get DNA embeddings
        anchor_dna = model(a_ks, a_sc, a_imu, return_dna=True)
        pos_dna = model(p_ks, p_sc, p_imu, return_dna=True)
        neg_dna = model(n_ks, n_sc, n_imu, return_dna=True)
        
        loss = criterion(anchor_dna, pos_dna, neg_dna)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def eval_epoch(
    model: BehavioralLSTM,
    loader: DataLoader,
    criterion: nn.TripletMarginLoss,
    device: torch.device
) -> float:
    """Evaluate one epoch. Returns avg loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            a_ks = batch["anchor_keystrokes"].to(device)
            a_sc = batch["anchor_scrolls"].to(device)
            a_imu = batch["anchor_imu_stats"].to(device)
            
            p_ks = batch["pos_keystrokes"].to(device)
            p_sc = batch["pos_scrolls"].to(device)
            p_imu = batch["pos_imu_stats"].to(device)
            
            n_ks = batch["neg_keystrokes"].to(device)
            n_sc = batch["neg_scrolls"].to(device)
            n_imu = batch["neg_imu_stats"].to(device)
            
            anchor_dna = model(a_ks, a_sc, a_imu, return_dna=True)
            pos_dna = model(p_ks, p_sc, p_imu, return_dna=True)
            neg_dna = model(n_ks, n_sc, n_imu, return_dna=True)
            
            loss = criterion(anchor_dna, pos_dna, neg_dna)
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / max(n_batches, 1)


def main():
    """Main training pipeline for verification."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    root = Path(__file__).resolve().parent
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # ── 1. Load all data ──────────────────────────────────────────
    db_path = root / "sentinel_lab.db"
    keyrecs_path = root / "datasets" / "keyrecs" / "fixed-text.csv"
    
    # Load & window raw enrollment data
    print("Loading raw enrollment data...")
    unified_data = load_raw_enrollment(db_path)
    print(f"  {len(unified_data)} users from DB")
    
    user_samples = {}
    for uid, data in unified_data.items():
        windows = window_user_data(data, ks_window=8, ks_stride=4)
        if len(windows) >= 2:
            user_samples[f"db_{uid[:8]}"] = windows
    print(f"  Windowed -> {sum(len(v) for v in user_samples.values())} samples from {len(user_samples)} users")
    
    # Load keyrecs data
    print("Loading keyrecs fixed-text data...")
    keyrecs = load_keyrecs_fixed(keyrecs_path)
    for uid, samples in keyrecs.items():
        if len(samples) >= 2:
            user_samples[f"kr_{uid}"] = samples
    print(f"  Added {sum(len(v) for v in keyrecs.values())} keyrecs samples from {len(keyrecs)} participants")
    
    total_samples = sum(len(v) for v in user_samples.values())
    print(f"\nTotal: {len(user_samples)} users, {total_samples} samples")
    
    # ── 2. Fit scalers ────────────────────────────────────────────
    print("\nFitting scalers...")
    all_flat = [s for samples in user_samples.values() for s in samples]
    ks_scaler, sc_scaler, imu_scaler = fit_scalers_from_samples(all_flat)
    save_scalers(ks_scaler, sc_scaler, imu_scaler, output_path=root / "scalers.pkl")
    save_key_vocab(output_path=root / "key_vocab.json")
    
    scalers_dict = {"keystrokes": ks_scaler, "scrolls": sc_scaler, "imu": imu_scaler}
    
    # ── 3. Train/val split (by user) ─────────────────────────────
    train_data, val_data = split_users(user_samples, train_ratio=0.8)
    print(f"Train: {len(train_data)} users, {sum(len(v) for v in train_data.values())} samples")
    print(f"Val:   {len(val_data)} users, {sum(len(v) for v in val_data.values())} samples")
    
    train_dataset = VerificationDataset(train_data, scalers_dict, augment=True)
    val_dataset = VerificationDataset(val_data, scalers_dict, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # ── 4. Model ──────────────────────────────────────────────────
    # num_users doesn't matter for verification, but keep for architecture compat
    model = BehavioralLSTM(num_users=len(user_samples)).to(device)
    
    criterion = nn.TripletMarginLoss(margin=0.5, p=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    
    # ── 5. Training loop ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Starting triplet training...")
    print("=" * 60)
    
    best_val_loss = float("inf")
    patience = 15
    patience_counter = 0
    
    for epoch in range(100):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1:3d}/100 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", end="")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
            print(f"  <- saved (best)")
        else:
            patience_counter += 1
            print()
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # -- 6. Evaluate verification performance ----------------------
    print("\n" + "=" * 60)
    print("Evaluating verification performance...")
    print("=" * 60)
    
    model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt", map_location=device))
    
    # Evaluate on validation users
    val_embeddings = extract_all_embeddings(model, val_data, scalers_dict, device)
    eer, mean_gen, mean_imp = evaluate_verification(val_embeddings)
    
    print(f"\nValidation Results:")
    print(f"  EER:                      {eer:.4f}  (lower = better, 0 = perfect)")
    print(f"  Mean genuine similarity:  {mean_gen:.4f}  (higher = better)")
    print(f"  Mean impostor similarity: {mean_imp:.4f}  (lower = better)")
    print(f"  Separation gap:           {mean_gen - mean_imp:.4f}")
    
    if eer < 0.15:
        print("\n[OK] Good verification performance!")
    elif eer < 0.30:
        print("\n[WARN] Moderate performance -- may need more data or tuning")
    else:
        print("\n[FAIL] Poor performance -- model needs improvement")
    
    print(f"\nModel saved to: {checkpoint_dir / 'best_model.pt'}")
    print(f"Scalers saved to: {root / 'scalers.pkl'}")


if __name__ == "__main__":
    main()
