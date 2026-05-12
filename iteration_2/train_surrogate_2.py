import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ..utils.utils import get_device

# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------

CONFIG = {
    # Paths
    "data_dir": "./data",
    "model_save_path": "surrogate_model_v3.pth",
    "scalers_save_path": "scalers_v3.pkl",
    # Data Params
    "window_size": 90,  # Lookback window (3 seconds at 30Hz)
    "stride": 1,
    "test_split": 0.2,
    # Model Params
    "hidden_dim": 256,
    "num_layers": 4,
    "num_heads": 8,
    "state_embed_dim": 32,  # Learnable state embedding dimension
    "dropout": 0.1,
    "batch_size": 64,
    "epochs": 50,
    "learning_rate": 1e-3,
    "prediction_horizon": 1,  # Configurable: how many steps to predict
    "clinical_loss_weight": 2.0,  # Configurable: weight for clinical metrics loss
    "device": get_device(),
    # Variable Definitions
    "plot_vars": ["lungs.q_in[1].p", "lungs.q_in[1].m_flow", "Ecg.ecg", "EithaPressure.pressure"],
    "monitor_vars": [
        "filter.y",
        "currentHeartReat.y",
        "HRAdd.y",
        "arterialPressure.systolic",
        "arterialPressure.diastolic",
        "arterial.sO2",
        "arterial.pO2",
        "arterial.pCO2",
        "arterial.pH",
        "tissueUnit[1].pH",
        "venous.pH",
    ],
    # Map State ID
    "state_files": {0: "normal.csv", 1: "pneumonia.csv", 2: "ventilated.csv", 3: "stabilized.csv"},
    "state_names": {0: "Normal", 1: "Pneumonia", 2: "Ventilated", 3: "Stabilized"},
}


# ---------------------------------------------
# MODEL DEFINITION
# ---------------------------------------------


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for temporal sequences"""

    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :]


class WaveformHead(nn.Module):
    """Prediction head for fast waveform signals (ECG, pressure, flow)"""

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
    """Prediction head for slow clinical metrics with residual connection"""

    def __init__(self, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        # Residual projection for preserving slow-changing baseline values
        self.residual = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.head(x) + 0.3 * self.residual(x)


class DualHeadSurrogateModelV3(nn.Module):
    """
    Transformer-based dual-head architecture with learnable state embeddings.

    Key improvements over V2:
    - Learnable state embeddings instead of one-hot encoding
    - Better handling of state-specific dynamics
    - Per-state normalized inputs
    """

    def __init__(
        self,
        num_plot_vars,
        num_monitor_vars,
        num_states,
        state_embed_dim,
        hidden_dim,
        num_layers,
        num_heads,
        dropout,
        prediction_horizon,
    ):
        super().__init__()
        self.num_plot_vars = num_plot_vars
        self.num_monitor_vars = num_monitor_vars
        self.num_states = num_states
        self.state_embed_dim = state_embed_dim
        self.prediction_horizon = prediction_horizon
        self.hidden_dim = hidden_dim

        # Learnable state embedding: maps state_id to dense vector
        self.state_embedding = nn.Embedding(num_states, state_embed_dim)

        # Input = 15 physiological vars + 32 state embed + 4 temporal features
        input_dim = num_plot_vars + num_monitor_vars + state_embed_dim + 4

        # Input projection to hidden dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Learnable positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=200)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        # Separate prediction heads for different temporal scales
        self.waveform_head = WaveformHead(hidden_dim, num_plot_vars * prediction_horizon, dropout)
        self.clinical_head = ClinicalHead(
            hidden_dim, num_monitor_vars * prediction_horizon, dropout
        )

    def forward(self, x, state_ids):
        """
        Args:
            x: (Batch, Window, Features) - physiological vars + temporal encoding
            state_ids: (Batch,) - integer state IDs for embedding lookup
        Returns:
            predictions: (Batch, (num_plot + num_monitor) * horizon)
        """
        batch_size, seq_len, _ = x.shape

        # Get state embeddings and broadcast to sequence length
        state_embed = self.state_embedding(state_ids)  # (Batch, state_embed_dim)
        state_embed = state_embed.unsqueeze(1).expand(
            -1, seq_len, -1
        )  # (Batch, Window, state_embed_dim)

        # Concatenate state embedding with input features
        x = torch.cat([x, state_embed], dim=-1)  # (Batch, Window, input_dim)

        # Project to hidden dimension
        x = self.input_proj(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Apply transformer encoder
        encoded = self.encoder(x)

        # Extract last timestep encoding
        last_step = encoded[:, -1, :]

        # Predict through separate heads
        waveform_pred = self.waveform_head(last_step)
        clinical_pred = self.clinical_head(last_step)

        return torch.cat([waveform_pred, clinical_pred], dim=1)


# ---------------------------------------------
# WEIGHTED LOSS FUNCTION
# ---------------------------------------------


class WeightedMSELoss(nn.Module):
    """MSE loss with configurable weight for clinical variables"""

    def __init__(self, num_plot_vars, num_monitor_vars, clinical_weight, prediction_horizon):
        super().__init__()
        self.num_plot_vars = num_plot_vars
        self.num_monitor_vars = num_monitor_vars
        self.clinical_weight = clinical_weight
        self.prediction_horizon = prediction_horizon

    def forward(self, predictions, targets):
        batch_size = predictions.shape[0]

        # Reshape to (Batch, horizon, num_vars)
        preds = predictions.view(batch_size, self.prediction_horizon, -1)
        targs = targets.view(batch_size, self.prediction_horizon, -1)

        # Split into waveform and clinical
        waveform_preds = preds[:, :, : self.num_plot_vars]
        waveform_targs = targs[:, :, : self.num_plot_vars]

        clinical_preds = preds[:, :, self.num_plot_vars :]
        clinical_targs = targs[:, :, self.num_plot_vars :]

        # Compute separate MSE
        waveform_mse = torch.mean((waveform_preds - waveform_targs) ** 2)
        clinical_mse = torch.mean((clinical_preds - clinical_targs) ** 2)

        # Weighted combination
        total_loss = waveform_mse + self.clinical_weight * clinical_mse

        return total_loss, waveform_mse, clinical_mse


# ---------------------------------------------
# DATA PREPROCESSING WITH PER-STATE SCALERS
# ---------------------------------------------


def load_and_process_data(config):
    """
    Load data with per-state StandardScaler normalization.
    This prevents cross-state contamination (e.g., ventilated's extreme HRAdd values
    affecting the scaling of other states).
    """
    print("Loading data...")
    all_targets = config["plot_vars"] + config["monitor_vars"]
    num_targets = len(all_targets)
    num_plot_vars = len(config["plot_vars"])

    # Per-state scalers
    scalers = {}
    data_sequences = []

    # Step A: Load CSVs and fit per-state scalers
    for state_id, filename in config["state_files"].items():
        path = os.path.join(config["data_dir"], filename)
        if not os.path.exists(path):
            print(f"Warning: File {path} not found. Skipping.")
            continue

        df = pd.read_csv(path)

        # Extract target columns
        try:
            target_data = df[all_targets].values
        except KeyError as e:
            raise KeyError(f"Column missing in {filename}: {e}")

        # Check for NaNs
        if np.isnan(target_data).any():
            nan_count = np.sum(np.isnan(target_data))
            print(
                f"Warning: {nan_count} NaNs found in {filename}. Filling with forward/backward fill."
            )
            # Use pandas for better NaN handling
            target_df = pd.DataFrame(target_data, columns=all_targets)
            target_df = target_df.ffill().bfill().fillna(0)
            target_data = target_df.values

        # Fit per-state StandardScaler
        scaler = StandardScaler()
        scaler.fit(target_data)
        scalers[state_id] = scaler

        print(f"  State {state_id} ({config['state_names'][state_id]}): {len(target_data)} samples")
        data_sequences.append((target_data, state_id))

    # Save scalers
    joblib.dump(scalers, config["scalers_save_path"])
    print(f"Saved {len(scalers)} per-state scalers to {config['scalers_save_path']}")

    # Step B: Create Windows with per-state normalization
    X_list, y_list, state_ids_list = [], [], []

    print("Creating sequences...")
    horizon = config["prediction_horizon"]

    for raw_data, state_id in data_sequences:
        # Normalize with per-state scaler
        scaled_data = scalers[state_id].transform(raw_data)

        # Sliding Window
        num_samples = len(scaled_data) - config["window_size"] - horizon + 1
        stride = config["stride"]

        for i in range(0, num_samples, stride):
            # Input window: physiological vars only (state embedding added in model)
            window_data = scaled_data[i : i + config["window_size"]]  # (window_size, 15)

            # Temporal encoding (4 features: 30Hz + cardiac cycle)
            temporal_encoding = np.zeros((config["window_size"], 4))

            for t in range(config["window_size"]):
                abs_timestep = i + t

                # 30Hz sampling cycle
                phase_30hz = 2 * np.pi * (abs_timestep % 30) / 30
                temporal_encoding[t, 0] = np.sin(phase_30hz)
                temporal_encoding[t, 1] = np.cos(phase_30hz)

                # Cardiac cycle (~73.5 bpm = ~24.5 samples at 30Hz)
                cardiac_period = 24.5
                phase_cardiac = 2 * np.pi * (abs_timestep % cardiac_period) / cardiac_period
                temporal_encoding[t, 2] = np.sin(phase_cardiac)
                temporal_encoding[t, 3] = np.cos(phase_cardiac)

            # Concatenate: (window_size, 19) = 15 vars + 4 temporal
            # State embedding will be added in the model's forward pass
            full_window = np.hstack([window_data, temporal_encoding])

            # Multi-step targets
            targets = scaled_data[i + config["window_size"] : i + config["window_size"] + horizon]
            targets_flat = targets.flatten()

            X_list.append(full_window)
            y_list.append(targets_flat)
            state_ids_list.append(state_id)

    X = np.array(X_list)
    y = np.array(y_list)
    state_ids = np.array(state_ids_list)

    print(f"Total sequences: {len(X)}")
    for sid in config["state_files"].keys():
        count = np.sum(state_ids == sid)
        print(f"    State {sid} ({config['state_names'][sid]}): {count} sequences")

    return X, y, state_ids, num_targets, num_plot_vars, scalers


# ---------------------------------------------
# TRAINING LOOP WITH PER-STATE VALIDATION
# ---------------------------------------------


def compute_per_state_loss(model, dataloader, criterion, state_ids_tensor, config, device):
    """Compute validation loss per state for detailed monitoring."""
    model.eval()
    state_losses = {sid: [] for sid in config["state_files"].keys()}

    with torch.no_grad():
        start_idx = 0
        for batch_X, batch_y in dataloader:
            batch_size = batch_X.shape[0]
            batch_states = state_ids_tensor[start_idx : start_idx + batch_size].to(device)
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            preds = model(batch_X, batch_states)

            # Compute per-sample loss
            for i in range(batch_size):
                state_id = batch_states[i].item()
                sample_loss = torch.mean((preds[i] - batch_y[i]) ** 2).item()
                state_losses[state_id].append(sample_loss)

            start_idx += batch_size

    # Average per state
    avg_state_losses = {}
    for sid, losses in state_losses.items():
        if len(losses) > 0:
            avg_state_losses[sid] = np.mean(losses)
        else:
            avg_state_losses[sid] = float("nan")

    return avg_state_losses


def main():
    print("=" * 60)
    print("SURROGATE MODEL V3 - Per-State Scaling & State Embeddings")
    print("=" * 60)

    print(f"Using device: {CONFIG['device']}")

    try:
        X, y, state_ids, num_targets, num_plot_vars, scalers = load_and_process_data(CONFIG)
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback

        traceback.print_exc()
        return

    num_monitor_vars = num_targets - num_plot_vars

    # Stratified train/val split to ensure balanced state representation
    print("\nExecuting stratified train and validation split...")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=CONFIG["test_split"], random_state=42)
    train_idx, val_idx = next(splitter.split(X, state_ids))

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    state_ids_train, state_ids_val = state_ids[train_idx], state_ids[val_idx]

    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")

    # Create datasets and dataloaders
    train_ds = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train),
        torch.LongTensor(state_ids_train),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val),
        torch.LongTensor(state_ids_val),
    )

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False)

    # Model setup
    print("\nModel configuration:")
    print(
        f"  Input: {num_targets} vars + {CONFIG['state_embed_dim']} state embed + 4 temporal = {num_targets + CONFIG['state_embed_dim'] + 4} features"
    )
    print(f"  Hidden dim: {CONFIG['hidden_dim']}")
    print(f"  Layers: {CONFIG['num_layers']}, Heads: {CONFIG['num_heads']}")
    print(f"  Prediction horizon: {CONFIG['prediction_horizon']} step(s)")
    print(f"  Clinical loss weight: {CONFIG['clinical_loss_weight']}x")

    model = DualHeadSurrogateModelV3(
        num_plot_vars=num_plot_vars,
        num_monitor_vars=num_monitor_vars,
        num_states=4,
        state_embed_dim=CONFIG["state_embed_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_layers=CONFIG["num_layers"],
        num_heads=CONFIG["num_heads"],
        dropout=CONFIG["dropout"],
        prediction_horizon=CONFIG["prediction_horizon"],
    ).to(CONFIG["device"])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    criterion = WeightedMSELoss(
        num_plot_vars=num_plot_vars,
        num_monitor_vars=num_monitor_vars,
        clinical_weight=CONFIG["clinical_loss_weight"],
        prediction_horizon=CONFIG["prediction_horizon"],
    )

    print("\nStarting optimization...")
    best_val_loss = float("inf")

    for epoch in range(CONFIG["epochs"]):
        # Training phase
        model.train()
        train_loss = 0
        train_waveform_loss = 0
        train_clinical_loss = 0

        for batch_X, batch_y, batch_states in train_loader:
            batch_X = batch_X.to(CONFIG["device"])
            batch_y = batch_y.to(CONFIG["device"])
            batch_states = batch_states.to(CONFIG["device"])

            optimizer.zero_grad()
            preds = model(batch_X, batch_states)
            loss, wave_loss, clin_loss = criterion(preds, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_waveform_loss += wave_loss.item()
            train_clinical_loss += clin_loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_wave = train_waveform_loss / len(train_loader)
        avg_train_clin = train_clinical_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        val_waveform_loss = 0
        val_clinical_loss = 0

        with torch.no_grad():
            for batch_X, batch_y, batch_states in val_loader:
                batch_X = batch_X.to(CONFIG["device"])
                batch_y = batch_y.to(CONFIG["device"])
                batch_states = batch_states.to(CONFIG["device"])

                preds = model(batch_X, batch_states)
                loss, wave_loss, clin_loss = criterion(preds, batch_y)

                val_loss += loss.item()
                val_waveform_loss += wave_loss.item()
                val_clinical_loss += clin_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_wave = val_waveform_loss / len(val_loader)
        avg_val_clin = val_clinical_loss / len(val_loader)

        # Per-state validation loss (every 5 epochs to save time)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            # Create a simple loader without shuffling for per-state computation
            val_loader_ordered = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False)
            state_losses = {}
            state_counts = {sid: 0 for sid in CONFIG["state_files"].keys()}
            state_sums = {sid: 0.0 for sid in CONFIG["state_files"].keys()}

            with torch.no_grad():
                for batch_X, batch_y, batch_states in val_loader_ordered:
                    batch_X = batch_X.to(CONFIG["device"])
                    batch_y = batch_y.to(CONFIG["device"])
                    batch_states = batch_states.to(CONFIG["device"])

                    preds = model(batch_X, batch_states)

                    for i in range(batch_X.shape[0]):
                        sid = batch_states[i].item()
                        sample_loss = torch.mean((preds[i] - batch_y[i]) ** 2).item()
                        state_sums[sid] += sample_loss
                        state_counts[sid] += 1

            state_losses = {
                sid: state_sums[sid] / max(state_counts[sid], 1)
                for sid in CONFIG["state_files"].keys()
            }
            state_loss_str = " | ".join(
                [
                    f"{CONFIG['state_names'][sid]}: {state_losses[sid]:.4f}"
                    for sid in sorted(state_losses.keys())
                ]
            )
            print(f"  Per-State Val Loss: [{state_loss_str}]")

        # Update learning rate
        scheduler.step(avg_val_loss)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(
            f"  Train - Total: {avg_train_loss:.6f} | Wave: {avg_train_wave:.6f} | Clin: {avg_train_clin:.6f}"
        )
        print(
            f"  Val   - Total: {avg_val_loss:.6f} | Wave: {avg_val_wave:.6f} | Clin: {avg_val_clin:.6f}\n"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": CONFIG,
                    "num_plot_vars": num_plot_vars,
                    "num_monitor_vars": num_monitor_vars,
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
