"""
train_world_model.py
====================
Trains the Data-Driven Neural Traffic World Model on real-world Indian traffic data
(Delhi Traffic Density Dataset + Pune Heterogeneous flow distributions).
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Ensure src is in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from neural_world_model import TrafficDynamicsWorldModel

DATA_PATH = os.path.join("data", "processed", "delhi_pune_transitions.npz")
MODEL_SAVE_PATH = os.path.join("mappo_models", "traffic_world_model.pth")

BATCH_SIZE = 256
EPOCHS = 15
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    print("=" * 65)
    print("  Training Neural Traffic World Model on Real Indian Datasets")
    print(f"  Device: {DEVICE}")
    print("=" * 65)

    if not os.path.exists(DATA_PATH):
        print(f"  [!] Missing dataset at {DATA_PATH}. Run tools/download_and_prep_datasets.py first.")
        return

    print(f"  Loading dataset: {DATA_PATH}...")
    data = np.load(DATA_PATH)
    states = torch.tensor(data["states"], dtype=torch.float32)
    actions = torch.tensor(data["actions"], dtype=torch.int64)
    rewards = torch.tensor(data["rewards"], dtype=torch.float32)
    next_states = torch.tensor(data["next_states"], dtype=torch.float32)

    n_samples = len(states)
    print(f"  Total Transitions: {n_samples:,}")

    # Train / Val Split (85% / 15%)
    val_size = int(0.15 * n_samples)
    train_size = n_samples - val_size

    indices = np.random.permutation(n_samples)
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    train_dataset = TensorDataset(states[train_idx], actions[train_idx], rewards[train_idx], next_states[train_idx])
    val_dataset = TensorDataset(states[val_idx], actions[val_idx], rewards[val_idx], next_states[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"  Train samples: {train_size:,} | Validation samples: {val_size:,}")

    model = TrafficDynamicsWorldModel().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    mse_loss = nn.MSELoss()

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_s_losses, train_r_losses = [], []

        for b_s, b_a, b_r, b_ns in train_loader:
            b_s, b_a, b_r, b_ns = b_s.to(DEVICE), b_a.to(DEVICE), b_r.to(DEVICE), b_ns.to(DEVICE)

            pred_ns, pred_r = model(b_s, b_a)

            l_state = mse_loss(pred_ns, b_ns)
            l_reward = mse_loss(pred_r, b_r)
            loss = l_state + 0.1 * l_reward

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_s_losses.append(l_state.item())
            train_r_losses.append(l_reward.item())

        scheduler.step()

        # Validation
        model.eval()
        val_s_losses, val_r_losses = [], []
        with torch.no_grad():
            for b_s, b_a, b_r, b_ns in val_loader:
                b_s, b_a, b_r, b_ns = b_s.to(DEVICE), b_a.to(DEVICE), b_r.to(DEVICE), b_ns.to(DEVICE)
                pred_ns, pred_r = model(b_s, b_a)
                val_s_losses.append(mse_loss(pred_ns, b_ns).item())
                val_r_losses.append(mse_loss(pred_r, b_r).item())

        mean_train_s = np.mean(train_s_losses)
        mean_val_s = np.mean(val_s_losses)
        mean_val_r = np.mean(val_r_losses)
        total_val = mean_val_s + 0.1 * mean_val_r

        print(f"  Epoch {epoch:02d}/{EPOCHS:02d} | Train State MSE: {mean_train_s:.6f} | Val State MSE: {mean_val_s:.6f} | Val Rew MSE: {mean_val_r:.6f}")

        if total_val < best_val_loss:
            best_val_loss = total_val
            model.save(MODEL_SAVE_PATH)
            print(f"   -> [BEST] Checkpoint saved to {MODEL_SAVE_PATH} (Val Loss: {best_val_loss:.6f})")

    elapsed = time.time() - start_time
    print("=" * 65)
    print(f"  World Model Training Complete in {elapsed:.1f}s!")
    print(f"  Final Model Saved: {MODEL_SAVE_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    train()
