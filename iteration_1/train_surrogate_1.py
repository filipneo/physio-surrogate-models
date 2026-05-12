import json
import os
import time
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from hemodynamics_dataset import HemodynamicsDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.utils import get_device

# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------

TARGET_VARS = ["LA.q_in.pressure", "LV.q_in.pressure", "AOV.q_out.pressure", "LV.excessVolume"]
SAMPLING_RATE = 30
SEQ_LEN = 75
PRED_LEN = 1
BATCH_SIZE = 64
EMBED_DIM = 128
NUM_LAYERS = 2
DEVICE = get_device()


# ---------------------------------------------
# MODEL DEFINITION
# ---------------------------------------------


class RCNBlock(nn.Module):
    """Single residual convolutional block combining a causal conv with a feed-forward network."""

    def __init__(self, embed_dim):
        super().__init__()
        self.proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, x):
        # x: (B, T, D)
        res = x
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.proj(x).transpose(1, 2)  # back to (B, T, D)
        x = self.norm(x + res)
        return x + self.ffn(x)


class RCNSim(nn.Module):
    """Residual Convolutional Forecaster for hemodynamic time-series prediction."""

    def __init__(self, input_dim, embed_dim, num_layers):
        super().__init__()
        self.embed = nn.Linear(input_dim, embed_dim)
        self.encoder = nn.Sequential(*[RCNBlock(embed_dim) for _ in range(num_layers)])
        self.out = nn.Linear(embed_dim, len(TARGET_VARS))

    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        return self.out(x[:, -1])  # predict based on last token


# ---------------------------------------------
# TRAINING
# ---------------------------------------------


def train_model(
    csv_dir,
    epochs=10,
    save_path=f"./trained_models/rcn_{SEQ_LEN}seq_{SAMPLING_RATE}hz_phase.pt",
):
    """Trains the RCN model on hemodynamics data and exports weights and ONNX artifact."""
    dataset = HemodynamicsDataset(
        csv_dir,
        target_vars=TARGET_VARS,
        sampling_rate=SAMPLING_RATE,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        include_phase=True,
    )
    print(dataset)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = RCNSim(
        input_dim=len(TARGET_VARS) + 2,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
    )
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        losses = []
        progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)
        for x, y in progress_bar:
            x, y = x.to(DEVICE), y.squeeze(1).to(DEVICE)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            progress_bar.set_postfix(loss=f"{loss.item():.5f}")
        print(f"Epoch {epoch + 1}: loss = {np.mean(losses):.5f}")

    # Save the model weights
    torch.save(model.state_dict(), save_path)

    # Compute global normalization parameters across all CSV files
    norm_params_path = save_path.replace(".pt", "_norm_params.npz")
    csv_files = glob(os.path.join(csv_dir, "*.csv"))
    all_data = []
    for file in csv_files:
        df = pd.read_csv(file)
        if not all(col in df.columns for col in TARGET_VARS):
            continue
        all_data.append(df[TARGET_VARS].values.astype(np.float32))

    combined_data = np.vstack(all_data)
    global_mean = combined_data.mean(0)
    global_std = combined_data.std(0) + 1e-6
    export_to_onnx(model, save_path, global_mean, global_std)

    np.savez(norm_params_path, mean=global_mean, std=global_std)
    print(f"Model and config saved to {save_path}")
    print(f"Normalization parameters saved to {norm_params_path}")
    return model


def export_to_onnx(model, save_path, mean, std):
    """Exports the trained model to ONNX format and saves normalization parameters as JSON."""
    dummy_input = torch.randn(1, SEQ_LEN, len(TARGET_VARS) + 2).to(DEVICE)

    onnx_path = save_path.replace(".pt", ".onnx")
    torch.onnx.export(
        model,
        dummy_input,  # type: ignore
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch", 1: "seq"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX model exported to {onnx_path}")

    # Save normalization parameters as JSON for downstream use
    norm_json_path = save_path.replace(".pt", "_norm_params.json")
    with open(norm_json_path, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)
    print(f"Normalization parameters saved to {norm_json_path}")


# ---------------------------------------------
# INFERENCE
# ---------------------------------------------


def simulate(model, heart_rate, steps=100):
    """Autoregressively rolls the model forward for the given number of steps."""
    model.eval()
    with torch.no_grad():
        hr_hz = torch.tensor([[heart_rate / 60.0]], dtype=torch.float32).to(DEVICE)
        dt = 1 / SAMPLING_RATE
        period = 60.0 / heart_rate
        x = torch.zeros(1, SEQ_LEN, len(TARGET_VARS)).to(DEVICE)
        phase = torch.zeros(1, SEQ_LEN, 1).to(DEVICE)
        out_series = []
        for t in range(steps):
            current_phase = torch.tensor([[(t * dt % period) / period]], dtype=torch.float32).to(
                DEVICE
            )
            hr_feat = hr_hz.expand(1, SEQ_LEN, 1)
            phase = torch.roll(phase, shifts=-1, dims=1)
            phase[:, -1] = current_phase
            x_in = torch.cat([x, hr_feat, phase], dim=-1)
            pred = model(x_in)
            out_series.append(pred.cpu().numpy())
            x = torch.roll(x, shifts=-1, dims=1)
            x[:, -1] = pred
        return np.vstack(out_series)


# ---------------------------------------------
# EVALUATION
# ---------------------------------------------


def evaluate_90_bpm(model, file_path):
    """Runs a 10-second simulation at 90 bpm and plots predicted vs. ground-truth traces."""
    df = pd.read_csv(file_path)
    true = df[TARGET_VARS].values[: (SAMPLING_RATE * 10)]
    true = (true - true.mean(0)) / (true.std(0) + 1e-6)
    start_time = time.time()
    pred = simulate(model, heart_rate=90, steps=(SAMPLING_RATE * 10))
    end_time = time.time()
    print(f"Simulation took {end_time - start_time:.2f} seconds")

    plt.figure(figsize=(12, 8))
    for i, var in enumerate(TARGET_VARS):
        plt.subplot(len(TARGET_VARS), 1, i + 1)
        plt.plot(true[:, i], label=f"True {var}", linestyle="--")
        plt.plot(pred[:, i], label=f"Pred {var}")
        plt.ylabel(var)
        plt.legend()
        plt.grid(True)
    plt.xlabel("Time step")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print(DEVICE)
    model = train_model(f"./simulated_data/{SAMPLING_RATE}hz", epochs=20)
    evaluate_90_bpm(model, (f"./test_data/default_90_bpm_{SAMPLING_RATE}hz.csv"))
