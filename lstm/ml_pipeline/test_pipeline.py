"""Test behavioral authentication pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data_loader import load_raw_enrollment, load_scalers
from dataset import BehavioralDataset
from inference import BehavioralInference
from model import BehavioralLSTM


def test_pipeline():
    """Test the complete pipeline."""
    root = Path(__file__).resolve().parent
    checkpoint_dir = root / "checkpoints"
    device = torch.device("cpu")
    
    # Load data
    print("Loading data...")
    db_path = root / "sentinel_lab.db"
    unified_data = load_raw_enrollment(db_path)
    print(f"Loaded {len(unified_data)} users")
    
    # Load scalers
    scalers = load_scalers(output_path=root / "scalers.pkl")
    
    # Create dataset
    dataset = BehavioralDataset(unified_data, scalers_dict=scalers)
    
    # Load model
    print("Loading model...")
    num_users = len(dataset.users)
    model = BehavioralLSTM(num_users=num_users).to(device)
    model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt", map_location=device))
    model.eval()
    
    # Test on first 3 users
    num_test_users = min(3, len(dataset.users))
    test_users = dataset.users[:num_test_users]
    
    print(f"\nTesting on {num_test_users} users: {test_users}")
    
    # Extract DNA for each test user
    user_dnas = {}
    
    with torch.no_grad():
        for user_id in test_users:
            if user_id not in unified_data:
                continue
            
            data = unified_data[user_id]
            
            # Preprocess
            from data_loader import preprocess_keystrokes, preprocess_scrolls, extract_imu_stats
            
            ks_scaler = scalers.get("keystrokes")
            sc_scaler = scalers.get("scrolls")
            imu_scaler = scalers.get("imu")
            
            keystrokes = preprocess_keystrokes(data["keystrokes"], scaler=ks_scaler)
            scrolls = preprocess_scrolls(data["scrolls"], scaler=sc_scaler)
            imu_stats = extract_imu_stats(data["imu"], scaler=imu_scaler)
            
            # Convert to tensors
            ks_tensor = torch.from_numpy(keystrokes).float().to(device)
            sc_tensor = torch.from_numpy(scrolls).float().to(device)
            imu_tensor = torch.from_numpy(imu_stats).float().to(device)
            
            # Extract DNA
            dna = model(ks_tensor, sc_tensor, imu_tensor, return_dna=True)
            user_dnas[user_id] = dna.cpu().numpy()[0]  # [32]
            
            print(f"Extracted DNA for {user_id}: shape={user_dnas[user_id].shape}, norm={np.linalg.norm(user_dnas[user_id]):.4f}")
    
    # Test self-similarity and cross-similarity
    print("\nTesting similarities:")
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    self_similarities = []
    cross_similarities = []
    
    for user_id in test_users:
        if user_id not in user_dnas:
            continue
        
        dna = user_dnas[user_id]
        
        # Self-similarity (should be ~1.0)
        self_sim = cosine_similarity([dna], [dna])[0, 0]
        self_similarities.append(self_sim)
        print(f"{user_id} self-similarity: {self_sim:.4f}")
        
        # Cross-similarity with other users
        for other_user_id in test_users:
            if other_user_id == user_id or other_user_id not in user_dnas:
                continue
            
            other_dna = user_dnas[other_user_id]
            cross_sim = cosine_similarity([dna], [other_dna])[0, 0]
            cross_similarities.append(cross_sim)
            print(f"  Cross-similarity with {other_user_id}: {cross_sim:.4f}")
    
    # Assertions
    avg_self = np.mean(self_similarities)
    avg_cross = np.mean(cross_similarities)
    
    print(f"\nAverage self-similarity: {avg_self:.4f}")
    print(f"Average cross-similarity: {avg_cross:.4f}")
    
    assert avg_self > 0.7, f"Self-similarity {avg_self:.4f} should be > 0.7"
    assert avg_cross < 0.5, f"Cross-similarity {avg_cross:.4f} should be < 0.5"
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_pipeline()
