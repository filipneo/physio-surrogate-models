import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..utils.utils import get_device, get_variables

# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------

CONFIG = {
    # Paths
    "data_path": "./data/training_dataset.csv",
    "model_save_path": "surrogate_model_final.pth",
    "scalers_save_path": "scaler_final.pkl",
    # Data Params
    "window_size": 90,
    "stride": 1,
    "test_split": 0.2,
    # Model Params
    "hidden_dim": 256,
    "num_layers": 4,
    "num_heads": 8,
    "control_embed_dim": 32,
    "dropout": 0.1,
    "batch_size": 64,
    "epochs": 100,
    "learning_rate": 1e-3,
    "prediction_horizon": 1,
    "clinical_loss_weight": 2.0,
    "huber_beta": 1.0,
    "noise_std": 0.05,  # Standard deviation for exposure bias noise injection
    "diff_weight": 0.5,  # Multiplier for the smoothness/derivative penalty
    "device": get_device(),
    # Variable Definitions
    "plot_vars": get_variables(["plot_vars"]),
    "monitor_vars": get_variables(["monitor_vars"]),
    "control_vars": get_variables(["control_vars"]),
}

# ---------------------------------------------
# MODEL DEFINITION
# ---------------------------------------------


class RelativePositionalConv(nn.Module):
    """
    Replaces absolute positional encoding with a depthwise causal convolution.
    This allows the model to understand relative temporal distances inherently
    without tying physiological states to an absolute time index.
    """

    def __init__(self, hidden_dim, kernel_size=5):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            groups=hidden_dim,
        )

    def forward(self, x):
        # x expected shape: (Batch, Seq_Len, Dim)
        x_transposed = x.transpose(1, 2)
        out = self.conv(x_transposed)
        # Truncate the end to maintain causality
        out = out[:, :, : -self.padding]
        return out.transpose(1, 2) + x


class WaveformHead(nn.Module):
    """Prediction head for fast waveform signals."""

    def __init__(self, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.head(x)


class ClinicalHead(nn.Module):
    """Prediction head for slow clinical metrics with residual connection."""

    def __init__(self, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        self.residual = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.head(x) + 0.3 * self.residual(x)


class ContinuousSurrogateModel(nn.Module):
    """Transformer-based architecture with relative positioning."""

    def __init__(
        self,
        num_plot_vars,
        num_monitor_vars,
        num_controls,
        control_embed_dim,
        hidden_dim,
        num_layers,
        num_heads,
        dropout,
        prediction_horizon,
    ):
        super().__init__()
        self.num_plot_vars = num_plot_vars
        self.num_monitor_vars = num_monitor_vars
        self.num_controls = num_controls
        self.prediction_horizon = prediction_horizon
        self.hidden_dim = hidden_dim

        self.control_encoder = nn.Sequential(
            nn.Linear(num_controls, control_embed_dim),
            nn.GELU(),
            nn.Linear(control_embed_dim, control_embed_dim),
        )

        input_dim = num_plot_vars + num_monitor_vars + control_embed_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.relative_pos = RelativePositionalConv(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )

        self.waveform_head = WaveformHead(hidden_dim, num_plot_vars * prediction_horizon, dropout)
        self.clinical_head = ClinicalHead(
            hidden_dim, num_monitor_vars * prediction_horizon, dropout
        )

    def forward(self, x_phys, controls):
        encoded_controls = self.control_encoder(controls)
        x = torch.cat([x_phys, encoded_controls], dim=-1)

        x = self.input_proj(x)
        x = self.relative_pos(x)

        encoded = self.encoder(x)
        last_step = encoded[:, -1, :]

        waveform_pred = self.waveform_head(last_step)
        clinical_pred = self.clinical_head(last_step)

        return torch.cat([waveform_pred, clinical_pred], dim=1)


# ---------------------------------------------
# LOSS AND DATA PROCESSING
# ---------------------------------------------


class SmoothnessPenalizedLoss(nn.Module):
    """
    Computes Huber loss with an explicit penalty on the first-order derivative
    to eliminate high-frequency jitter in autoregressive waveform predictions.
    """

    def __init__(
        self,
        num_plot_vars,
        num_monitor_vars,
        clinical_weight,
        prediction_horizon,
        beta=1.0,
        diff_weight=0.5,
    ):
        super().__init__()
        self.num_plot_vars = num_plot_vars
        self.num_monitor_vars = num_monitor_vars
        self.clinical_weight = clinical_weight
        self.prediction_horizon = prediction_horizon
        self.beta = beta
        self.diff_weight = diff_weight

    def forward(self, predictions, targets, x_last):
        """
        Args:
            predictions: (Batch, horizon * num_vars)
            targets: (Batch, horizon * num_vars)
            x_last: (Batch, num_vars) - The final step of the physiological input window
        """
        batch_size = predictions.shape[0]

        # Reshape to (Batch, Horizon, Features)
        preds = predictions.reshape(batch_size, self.prediction_horizon, -1)
        targs = targets.reshape(batch_size, self.prediction_horizon, -1)

        # 1. Point-wise Huber Loss
        waveform_preds = preds[:, :, : self.num_plot_vars].contiguous()
        waveform_targs = targs[:, :, : self.num_plot_vars].contiguous()

        clinical_preds = preds[:, :, self.num_plot_vars :].contiguous()
        clinical_targs = targs[:, :, self.num_plot_vars :].contiguous()

        waveform_loss = F.smooth_l1_loss(waveform_preds, waveform_targs, beta=self.beta)
        clinical_loss = F.smooth_l1_loss(clinical_preds, clinical_targs, beta=self.beta)

        point_loss = waveform_loss + self.clinical_weight * clinical_loss

        # 2. Temporal Difference (Derivative) Loss
        x_last_expanded = x_last.unsqueeze(1)

        preds_with_hist = torch.cat([x_last_expanded, preds], dim=1)
        targs_with_hist = torch.cat([x_last_expanded, targs], dim=1)

        # Calculate discrete derivatives
        preds_diff = preds_with_hist[:, 1:, :] - preds_with_hist[:, :-1, :]
        targs_diff = targs_with_hist[:, 1:, :] - targs_with_hist[:, :-1, :]

        # Apply derivative penalty strictly to the high-frequency waveforms
        waveform_preds_diff = preds_diff[:, :, : self.num_plot_vars].contiguous()
        waveform_targs_diff = targs_diff[:, :, : self.num_plot_vars].contiguous()

        diff_loss = F.smooth_l1_loss(waveform_preds_diff, waveform_targs_diff, beta=self.beta)

        total_loss = point_loss + (self.diff_weight * diff_loss)

        return total_loss, waveform_loss, clinical_loss, diff_loss


def load_and_process_data(config):
    """Loads dataset and performs highly optimized vectorized windowing."""
    print("Loading continuous dataset...")
    if not os.path.exists(config["data_path"]):
        raise FileNotFoundError(f"Dataset not found at {config['data_path']}")

    df = pd.read_csv(config["data_path"])

    phys_vars = config["plot_vars"] + config["monitor_vars"]
    control_vars = config["control_vars"]

    num_phys_vars = len(phys_vars)
    num_plot_vars = len(config["plot_vars"])
    num_controls = len(control_vars)

    df[phys_vars] = df[phys_vars].ffill().bfill().fillna(0)

    phys_data = df[phys_vars].values
    control_data = df[control_vars].values

    print("Fitting scalers...")
    phys_scaler = StandardScaler()
    scaled_phys_data = phys_scaler.fit_transform(phys_data)

    control_scaler = MinMaxScaler()
    scaled_control_data = control_scaler.fit_transform(control_data)

    joblib.dump(
        {"phys_scaler": phys_scaler, "control_scaler": control_scaler}, config["scalers_save_path"]
    )

    print("Vectorizing sequential windows...")
    window_size = config["window_size"]
    horizon = config["prediction_horizon"]
    stride = config["stride"]

    phys_windows = np.lib.stride_tricks.sliding_window_view(
        scaled_phys_data[:-horizon], window_shape=(window_size, num_phys_vars)
    ).squeeze(1)

    control_windows = np.lib.stride_tricks.sliding_window_view(
        scaled_control_data[:-horizon], window_shape=(window_size, num_controls)
    ).squeeze(1)

    target_windows = np.lib.stride_tricks.sliding_window_view(
        scaled_phys_data[window_size:], window_shape=(horizon, num_phys_vars)
    ).squeeze(1)

    targets_flat = target_windows.reshape(target_windows.shape[0], -1)

    X_phys = phys_windows[::stride]
    X_ctrl = control_windows[::stride]
    y = targets_flat[::stride]

    print(f"Total sequences generated: {len(X_phys)}")
    return X_phys, X_ctrl, y, num_phys_vars, num_plot_vars, num_controls


# ---------------------------------------------
# TRAINING LOOP
# ---------------------------------------------


def add_exposure_noise(batch_X, std, device):
    """Injects Gaussian noise into the physiological history to combat exposure bias."""
    if std > 0.0:
        noise = torch.randn_like(batch_X) * std
        return batch_X + noise.to(device)
    return batch_X


def main():
    print("=" * 60)
    print("SURROGATE MODEL TRAINING")
    print("=" * 60)

    print(f"Using device: {CONFIG['device']}")

    X_phys, X_ctrl, y, num_phys_vars, num_plot_vars, num_controls = load_and_process_data(CONFIG)
    num_monitor_vars = num_phys_vars - num_plot_vars

    print("\nExecuting train and validation split...")
    X_phys_train, X_phys_val, X_ctrl_train, X_ctrl_val, y_train, y_val = train_test_split(
        X_phys, X_ctrl, y, test_size=CONFIG["test_split"], random_state=42
    )

    print(f"Train samples: {len(X_phys_train)}")
    print(f"Val samples:   {len(X_phys_val)}")

    train_ds = TensorDataset(
        torch.FloatTensor(X_phys_train),
        torch.FloatTensor(X_ctrl_train),
        torch.FloatTensor(y_train),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_phys_val),
        torch.FloatTensor(X_ctrl_val),
        torch.FloatTensor(y_val),
    )

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False)

    model = ContinuousSurrogateModel(
        num_plot_vars=num_plot_vars,
        num_monitor_vars=num_monitor_vars,
        num_controls=num_controls,
        control_embed_dim=CONFIG["control_embed_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_layers=CONFIG["num_layers"],
        num_heads=CONFIG["num_heads"],
        dropout=CONFIG["dropout"],
        prediction_horizon=CONFIG["prediction_horizon"],
    ).to(CONFIG["device"])

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    criterion = SmoothnessPenalizedLoss(
        num_plot_vars=num_plot_vars,
        num_monitor_vars=num_monitor_vars,
        clinical_weight=CONFIG["clinical_loss_weight"],
        prediction_horizon=CONFIG["prediction_horizon"],
        beta=CONFIG["huber_beta"],
        diff_weight=CONFIG["diff_weight"],
    )

    print("\nStarting optimization...")
    best_val_loss = float("inf")

    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_loss, train_wave_loss, train_clin_loss, train_diff_loss = 0, 0, 0, 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']} [Train]")

        for batch_X, batch_ctrl, batch_y in train_pbar:
            batch_X = batch_X.to(CONFIG["device"])
            batch_ctrl = batch_ctrl.to(CONFIG["device"])
            batch_y = batch_y.to(CONFIG["device"])

            # Extract the final step of the sequence to anchor the derivative calculation
            x_last = batch_X[:, -1, :].clone()

            batch_X_noisy = add_exposure_noise(batch_X, CONFIG["noise_std"], CONFIG["device"])

            optimizer.zero_grad()
            preds = model(batch_X_noisy, batch_ctrl)

            loss, wave_loss, clin_loss, diff_loss = criterion(preds, batch_y, x_last)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_wave_loss += wave_loss.item()
            train_clin_loss += clin_loss.item()
            train_diff_loss += diff_loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss, val_wave_loss, val_clin_loss, val_diff_loss = 0, 0, 0, 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']} [Val]")

        with torch.no_grad():
            for batch_X, batch_ctrl, batch_y in val_pbar:
                batch_X = batch_X.to(CONFIG["device"])
                batch_ctrl = batch_ctrl.to(CONFIG["device"])
                batch_y = batch_y.to(CONFIG["device"])

                x_last = batch_X[:, -1, :].clone()
                preds = model(batch_X, batch_ctrl)

                loss, wave_loss, clin_loss, diff_loss = criterion(preds, batch_y, x_last)

                val_loss += loss.item()
                val_wave_loss += wave_loss.item()
                val_clin_loss += clin_loss.item()
                val_diff_loss += diff_loss.item()
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)
        avg_val_diff_loss = val_diff_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss:     {avg_train_loss:.6f}")
        print(f"  Val Total Loss: {avg_val_loss:.6f}")
        print(f"  Val Diff Loss:  {avg_val_diff_loss:.6f}\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": CONFIG,
                    "num_plot_vars": num_plot_vars,
                    "num_monitor_vars": num_monitor_vars,
                    "num_controls": num_controls,
                },
                CONFIG["model_save_path"],
            )
            print(f"Model checkpoint saved with validation loss: {best_val_loss:.6f}\n")

    print("\n" + "=" * 60)
    print("Training sequence completed.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
