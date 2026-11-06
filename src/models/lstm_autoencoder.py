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
    Trained without labels on a chronological training period.
    Anomaly score = reconstruction error on unseen data.
    High error = pattern differs from what the model learned in training.

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
    def __init__(self, sequence_len: int = 24, hidden_dim: int = 32, random_state: int = 42):
        self.sequence_len = sequence_len
        self.hidden_dim = hidden_dim
        self.model: LSTMAutoencoder | None = None
        self.scaler = StandardScaler()
        self.threshold: float | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state

    def _make_sequences(self, X: np.ndarray) -> torch.Tensor:
        """Slide a fixed window over X → (n_sequences, seq_len, n_features)."""
        n = len(X)
        if n < self.sequence_len:
            raise ValueError(
                f"Need at least {self.sequence_len} rows to build a sequence; got {n}"
            )
        seqs = np.stack(
            [X[i : i + self.sequence_len] for i in range(n - self.sequence_len + 1)],
            axis=0,
        )
        return torch.tensor(seqs, dtype=torch.float32)

    def _reconstruction_errors(self, X_scaled: np.ndarray) -> np.ndarray:
        """Per-timestep MSE (mean over features), padded to original length."""
        if self.model is None:
            raise RuntimeError("Model must be fitted before scoring")
        self.model.eval()
        sequences = self._make_sequences(X_scaled).to(self.device)
        errors = np.full(len(X_scaled), np.nan)

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

        seq_errors = np.concatenate(seq_errors)
        # Pad the front with NaN where a complete window is unavailable.
        errors[self.sequence_len - 1 :] = seq_errors
        return errors

    def fit(
        self,
        X_train: np.ndarray,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 0.001,
        save_artifact: bool = True,
    ) -> None:
        if X_train.ndim != 2 or len(X_train) < self.sequence_len:
            raise ValueError("X_train must be a 2D array with at least sequence_len rows")
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
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
        valid_errors = train_errors[np.isfinite(train_errors)]
        self.threshold = float(np.percentile(valid_errors, 95))

        if save_artifact:
            self.save(input_dim)
        logger.info(
            "LSTM Autoencoder trained | threshold=%.4f", self.threshold
        )

    def save(self, input_dim: int | None = None) -> None:
        if self.model is None or self.threshold is None:
            raise RuntimeError("Model and threshold must be available before saving")
        input_dim = input_dim or self.model.input_dim
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

    @classmethod
    def load(cls, path: str) -> "LSTMAnomalyDetector":
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        detector = cls(
            sequence_len=int(artifact["sequence_len"]),
            hidden_dim=int(artifact["hidden_dim"]),
        )
        detector.device = torch.device("cpu")
        detector.model = LSTMAutoencoder(
            input_dim=int(artifact["input_dim"]), hidden_dim=detector.hidden_dim
        ).to(detector.device)
        detector.model.load_state_dict(artifact["model_state"])
        detector.model.eval()
        detector.scaler.mean_ = np.asarray(artifact["scaler_mean"])
        detector.scaler.scale_ = np.asarray(artifact["scaler_scale"])
        detector.scaler.var_ = detector.scaler.scale_ ** 2
        detector.scaler.n_features_in_ = int(artifact["input_dim"])
        detector.scaler.n_samples_seen_ = 1
        detector.threshold = float(artifact["threshold"])
        return detector

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        errors = self._reconstruction_errors(X_scaled)
        return (errors > self.threshold).astype(int)

    def get_scores(self, X: np.ndarray) -> np.ndarray:
        """Return raw reconstruction errors — higher = more anomalous."""
        X_scaled = self.scaler.transform(X)
        return self._reconstruction_errors(X_scaled)

    def fit_predict(self, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        X_all = df[feature_cols].values
        n_total = len(X_all)
        logger.info("Training and predicting on %d supplied hours", n_total)

        self.fit(X_all)
        preds = self.predict(X_all)

        n = int(preds.sum())
        pct = 100 * n / n_total
        logger.info("LSTM Autoencoder flagged %d anomalies (%.1f%%)", n, pct)

        return preds
