import os
from glob import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class HemodynamicsDataset(Dataset):
    """
    A PyTorch Dataset for hemodynamics time series data.
    This dataset loads CSV files containing hemodynamics measurements and creates
    sequences suitable for time series prediction tasks. Each sequence includes
    the original features along with heart rate and optionally cardiac phase information.
    Args:
        csv_dir (str): Directory path containing CSV files with hemodynamics data.
        target_vars (list[str]): List of column names to use as target variables.
        sampling_rate (int): Sampling rate of the data in Hz.
        seq_len (int): Length of input sequences.
        pred_len (int): Length of prediction sequences.
        file_filter (str, optional): Glob pattern to filter CSV files. Defaults to "*.csv".
        include_phase (bool, optional): Whether to include cardiac phase information. Defaults to False.
    Attributes:
        sequences (list): List of tuples containing (input_sequence, target_sequence) pairs.
    Note:
        - CSV filenames should contain heart rate information in the format "*_<HR>_bpm.csv"
        - Data is normalized using z-score normalization across all files (global normalization)
        - Input sequences are augmented with heart rate (Hz) and optionally cardiac phase information
        - Only files containing all specified target variables are processed
    Returns:
        torch.Tensor: Input sequence with shape (seq_len, n_features + 1) or (seq_len, n_features + 2) if include_phase=True
        torch.Tensor: Target sequence with shape (pred_len, n_features)
    """

    def __init__(
        self,
        csv_dir: str,
        target_vars: list[str],
        sampling_rate: int,
        seq_len: int,
        pred_len: int,
        file_filter="*.csv",
        include_phase=False,
    ):
        self.sequences = []
        csv_files = glob(os.path.join(csv_dir, file_filter))

        # First pass: collect all data to calculate global statistics
        all_data = []
        file_hr_map = {}  # Store heart rates for each file

        for file in csv_files:
            df = pd.read_csv(file)
            if not all(col in df.columns for col in target_vars):
                continue

            hr = int(os.path.basename(file).split("_")[1])  # e.g., default_100_bpm.csv
            file_hr_map[file] = hr
            all_data.append(df[target_vars].values.astype(np.float32))

        if not all_data:
            return  # No valid files found

        # Concatenate all data and calculate global statistics
        combined_data = np.vstack(all_data)
        global_mean = combined_data.mean(0)
        global_std = combined_data.std(0) + 1e-6

        # Second pass: create sequences with global normalization
        for file in csv_files:
            if file not in file_hr_map:
                continue  # Skip files that were filtered out in the first pass

            df = pd.read_csv(file)
            hr = file_hr_map[file]
            hr_hz = hr / 60.0

            # Apply global normalization
            data = df[target_vars].values.astype(np.float32)
            data = (data - global_mean) / global_std

            if include_phase:
                dt = 1 / sampling_rate
                period = 60.0 / hr
                phases = np.array(
                    [(i * dt % period) / period for i in range(len(data))], dtype=np.float32
                ).reshape(-1, 1)

            for i in range(len(data) - seq_len - pred_len):
                x_seq = data[i : i + seq_len]
                y_seq = data[i + seq_len : i + seq_len + pred_len]
                hr_seq = np.full((seq_len, 1), hr_hz, dtype=np.float32)

                if include_phase:
                    phase_seq = phases[i : i + seq_len]
                    full_input = np.hstack([x_seq, hr_seq, phase_seq])
                else:
                    full_input = np.hstack([x_seq, hr_seq])

                self.sequences.append((full_input, y_seq))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x, y = self.sequences[idx]
        return torch.tensor(x), torch.tensor(y)

    def __str__(self):
        if len(self.sequences) == 0:
            return "Empty HemodynamicsDataset"

        sample_x, sample_y = self.sequences[0]
        return (
            f"HemodynamicsDataset:\n"
            f"  Total sequences: {len(self.sequences)}\n"
            f"  Input shape: {sample_x.shape}\n"
            f"  Target shape: {sample_y.shape}"
        )
