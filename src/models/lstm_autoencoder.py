import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models"
)


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for anomaly detection.
    Trained ONLY on normal hours (is_anomaly == 0).
    Anomaly score = reconstruction error on unseen data.
    High error = pattern doesn't match what the model learned as normal.

    Architecture:
      Encoder: LSTM(input_dim, hidden_dim=32, num_layers=1, batch_first=True)
      Decoder: LSTM(hidden_dim, hidden_dim=32, num_layers=1, batch_first=True)
               Linear(hidden_dim, input_dim)
    Sequence length: 24 hours (one full day context)
    """

    SEQUENCE_LEN = 24

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        seq_len = x.size(1)

        _, (h_n, _) = self.encoder(x)
        # h_n: (1, batch, hidden_dim) → repeat across time axis
        decoder_input = h_n.squeeze(0).unsqueeze(1).repeat(1, seq_len, 1)

        out, _ = self.decoder(decoder_input)
        reconstruction = self.output_layer(out)
        return reconstruction


class LSTMAnomalyDetector:
    def __init__(self, sequence_len: int = 24, hidden_dim: int = 32):
        self.sequence_len = sequence_len
        self.hidden_dim = hidden_dim
        self.model: LSTMAutoencoder | None = None
        self.scaler = StandardScaler()
        self.threshold: float | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _make_sequences(self, X: np.ndarray) -> torch.Tensor:
        """Slide a fixed window over X → (n_sequences, seq_len, n_features)."""
        n = len(X)
        seqs = np.stack(
            [X[i : i + self.sequence_len] for i in range(n - self.sequence_len)],
            axis=0,
        )
        return torch.tensor(seqs, dtype=torch.float32)

    def _reconstruction_errors(self, X_scaled: np.ndarray) -> np.ndarray:
        """Per-timestep MSE (mean over features), padded to original length."""
        self.model.eval()
        sequences = self._make_sequences(X_scaled).to(self.device)
        errors = np.zeros(len(X_scaled))

        with torch.no_grad():
            dataset = TensorDataset(sequences)
            loader = DataLoader(dataset, batch_size=256, shuffle=False)
            seq_errors = []
            for (batch,) in loader:
                recon = self.model(batch)
                # MSE over features for each (seq, timestep)
                mse = ((batch - recon) ** 2).mean(dim=2)  # (batch, seq_len)
                # Take error at the LAST timestep as the label for that position
                seq_errors.append(mse[:, -1].cpu().numpy())

        seq_errors = np.concatenate(seq_errors)  # length = n - seq_len
        # Pad front with zeros — no prediction possible for first seq_len rows
        errors[self.sequence_len :] = seq_errors
        return errors

    def fit(
        self,
        X_train: np.ndarray,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 0.001,
    ) -> None:
        X_scaled = self.scaler.fit_transform(X_train)

        sequences = self._make_sequences(X_scaled).to(self.device)
        dataset = TensorDataset(sequences)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        input_dim = X_train.shape[1]
        self.model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=self.hidden_dim).to(
            self.device
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                recon = self.model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch)

            epoch_loss /= len(sequences)
            if epoch % 5 == 0 or epoch == 1:
                logger.info("Epoch %3d/%d | loss=%.6f", epoch, epochs, epoch_loss)

        # Threshold = 95th percentile of per-timestep errors on training data
        train_errors = self._reconstruction_errors(X_scaled)
        # Exclude padded zeros from percentile computation
        non_zero = train_errors[self.sequence_len :]
        self.threshold = float(np.percentile(non_zero, 95))

        os.makedirs(_MODELS_DIR, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "scaler_mean": self.scaler.mean_,
                "scaler_scale": self.scaler.scale_,
                "threshold": self.threshold,
                "sequence_len": self.sequence_len,
                "hidden_dim": self.hidden_dim,
                "input_dim": input_dim,
            },
            os.path.join(_MODELS_DIR, "lstm_autoencoder.pt"),
        )
        logger.info(
            "LSTM Autoencoder trained | threshold=%.4f", self.threshold
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        errors = self._reconstruction_errors(X_scaled)
        return (errors > self.threshold).astype(int)

    def get_scores(self, X: np.ndarray) -> np.ndarray:
        """Return raw reconstruction errors — higher = more anomalous."""
        X_scaled = self.scaler.transform(X)
        return self._reconstruction_errors(X_scaled)

    def fit_predict(self, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        X_all = df[feature_cols].values
        X_normal = df.loc[df["is_anomaly"] == 0, feature_cols].values

        n_normal = len(X_normal)
        n_total = len(X_all)
        logger.info(
            "Training on %d normal hours, predicting on %d total hours",
            n_normal,
            n_total,
        )

        self.fit(X_normal)
        preds = self.predict(X_all)

        n = int(preds.sum())
        pct = 100 * n / n_total
        logger.info("LSTM Autoencoder flagged %d anomalies (%.1f%%)", n, pct)

        return preds
