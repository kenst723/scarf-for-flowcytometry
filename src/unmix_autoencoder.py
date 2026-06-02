import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.unmix_spectral import PoissonUnmixer


def fix_seed(seed=42):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class AF_AutoEncoder(nn.Module):
    """AutoEncoder for Autofluorescence spectrum reconstruction.

    Parameters
    ----------
    input_dim : int
        Number of spectral channels (e.g. 34 for Sony SA3800).
    hidden_dim : int
        Width of the hidden layers (default: 128).
    bottleneck_dim : int
        Dimension of the latent bottleneck layer (default: 16).
    dropout : float
        Dropout probability applied after each hidden layer (default: 0.1).
    """
    def __init__(self, input_dim=34, hidden_dim=128, bottleneck_dim=16, dropout=0.1):
        super(AF_AutoEncoder, self).__init__()
        mid_dim = hidden_dim // 2  # e.g. 64 for hidden_dim=128

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.LeakyReLU(0.1)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class SpectralLoss(nn.Module):
    """Combined loss: MSE + weighted cosine similarity loss.

    Cosine similarity encourages preservation of spectral shape,
    while MSE ensures accurate intensity reconstruction.

    Parameters
    ----------
    alpha : float
        Weight for the cosine similarity loss term (default: 0.1).
    """
    def __init__(self, alpha=0.1):
        super(SpectralLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        # Cosine similarity: 1.0 means perfect match, so loss = 1 - sim
        cos_sim = F.cosine_similarity(pred, target, dim=1).mean()
        return mse + self.alpha * (1.0 - cos_sim)


class AutoEncoderUnmixer:
    """AutoEncoder-based spectral unmixer.

    Trains an autoencoder on negative (unstained) samples to learn the
    autofluorescence manifold, then uses the learned model to predict
    and subtract autofluorescence from stained samples.

    Parameters
    ----------
    epochs : int
        Maximum number of training epochs (default: 100).
    batch_size : int
        Training batch size (default: 256).
    lr : float
        Initial learning rate for Adam optimizer (default: 0.001).
    hidden_dim : int
        Width of AE hidden layers (default: 128).
    bottleneck_dim : int
        Dimension of AE bottleneck (default: 16).
    dropout : float
        Dropout probability (default: 0.1).
    alpha : float
        Weight for cosine similarity in SpectralLoss (default: 0.1).
    patience : int
        Early stopping patience in epochs (default: 10).
    val_ratio : float
        Fraction of training data used for validation (default: 0.2).
    model_save_path : str or None
        Path to save the trained model checkpoint.
    seed : int
        Random seed for reproducibility (default: 42).
    """
    def __init__(self, epochs=100, batch_size=256, lr=0.001,
                 hidden_dim=128, bottleneck_dim=16, dropout=0.1,
                 alpha=0.1, patience=10, val_ratio=0.2,
                 model_save_path=None, seed=42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.dropout = dropout
        self.alpha = alpha
        self.patience = patience
        self.val_ratio = val_ratio
        self.seed = seed
        self.model = None
        self.val_min = 0.0
        self.val_max = 1.0
        self.cofactor = 150.0
        self.model_save_path = model_save_path
        self.S_AF = None
        self.S_Stain = None
        self.slope = 0.0
        self.bg = 0.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X_neg, X_stain):
        """Train the AutoEncoder on negative control data.

        Parameters
        ----------
        X_neg : ndarray, shape (n_neg, n_channels)
            Negative (unstained) spectral data.
        X_stain : ndarray, shape (n_stain, n_channels)
            Stained spectral data (used for PoissonUnmixer reference fitting).

        Returns
        -------
        self
        """
        fix_seed(self.seed)

        # 1. 従来のパラメータ（S_AF, S_Stain, slope）を計算するため、PoissonUnmixer を利用
        poisson = PoissonUnmixer()
        poisson.fit(X_neg, X_stain)
        self.S_AF = poisson.S_AF
        self.S_Stain = poisson.S_Stain
        self.slope = poisson.slope
        self.bg = poisson.bg

        # 2. AutoEncoder の学習
        input_dim = X_neg.shape[1]
        self.model = AF_AutoEncoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            bottleneck_dim=self.bottleneck_dim,
            dropout=self.dropout
        ).to(self.device)

        # 前処理 (ArcSinh + MinMax)
        X_neg_arcsinh = np.arcsinh(X_neg / self.cofactor)
        X_stain_arcsinh = np.arcsinh(X_stain / self.cofactor)

        self.val_min = np.min(X_neg_arcsinh)
        self.val_max = max(np.max(X_neg_arcsinh), np.max(X_stain_arcsinh))

        X_neg_scaled = (X_neg_arcsinh - self.val_min) / (self.val_max - self.val_min)
        X_neg_scaled = np.clip(X_neg_scaled, 0, 1)

        # Train / Validation split
        n_samples = len(X_neg_scaled)
        n_val = max(int(n_samples * self.val_ratio), 1)
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        X_train = X_neg_scaled[train_indices]
        X_val = X_neg_scaled[val_indices]

        tensor_train = torch.FloatTensor(X_train)
        tensor_val = torch.FloatTensor(X_val).to(self.device)
        dataset = TensorDataset(tensor_train, tensor_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = SpectralLoss(alpha=self.alpha)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Early stopping state
        best_val_loss = float('inf')
        best_state_dict = None
        epochs_without_improvement = 0

        print(f"    [AutoEncoder] Training on {self.device} for up to {self.epochs} epochs "
              f"(patience={self.patience}, val={n_val}/{n_samples})...")
        print(f"    [AutoEncoder] Architecture: {input_dim}→{self.hidden_dim}→"
              f"{self.hidden_dim // 2}→{self.bottleneck_dim}→...→{input_dim}")

        for epoch in range(self.epochs):
            # --- Training ---
            self.model.train()
            train_loss = 0.0
            n_batches = 0
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_x)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
                n_batches += 1

            train_loss /= len(X_train)

            # --- Validation ---
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(tensor_val)
                val_loss = criterion(val_pred, tensor_val).item()

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            # Log every 10 epochs or at the end
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"      Epoch {epoch + 1:4d}/{self.epochs} | "
                      f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                      f"LR: {current_lr:.2e}")

            # --- Early Stopping ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    print(f"      Early stopping at epoch {epoch + 1} "
                          f"(best val loss: {best_val_loss:.6f})")
                    break

        # Restore best model
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            self.model.to(self.device)

        self.model.eval()

        if self.model_save_path:
            self._save_checkpoint(self.model_save_path, input_dim)
            print(f"    [AutoEncoder] Checkpoint saved to {self.model_save_path}")

        return self

    def _save_checkpoint(self, path, input_dim):
        """Save full checkpoint including model weights and preprocessing params."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'input_dim': input_dim,
            'hidden_dim': self.hidden_dim,
            'bottleneck_dim': self.bottleneck_dim,
            'dropout': self.dropout,
            'val_min': self.val_min,
            'val_max': self.val_max,
            'cofactor': self.cofactor,
            'S_AF': self.S_AF,
            'S_Stain': self.S_Stain,
            'slope': self.slope,
            'bg': self.bg,
        }
        torch.save(checkpoint, path)

    def load_model(self, path):
        """Load model from a full checkpoint file.

        Restores architecture, weights, and all preprocessing parameters.

        Parameters
        ----------
        path : str
            Path to the checkpoint file (.pth).
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Restore preprocessing parameters
        self.val_min = checkpoint.get('val_min', self.val_min)
        self.val_max = checkpoint.get('val_max', self.val_max)
        self.cofactor = checkpoint.get('cofactor', self.cofactor)
        self.S_AF = checkpoint.get('S_AF', self.S_AF)
        self.S_Stain = checkpoint.get('S_Stain', self.S_Stain)
        self.slope = checkpoint.get('slope', self.slope)
        self.bg = checkpoint.get('bg', self.bg)

        # Restore model architecture and weights
        input_dim = checkpoint.get('input_dim', 34)
        hidden_dim = checkpoint.get('hidden_dim', 128)
        bottleneck_dim = checkpoint.get('bottleneck_dim', 16)
        dropout = checkpoint.get('dropout', 0.1)

        self.model = AF_AutoEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def _predict_af(self, X):
        """Predict autofluorescence component from raw spectral data.

        Parameters
        ----------
        X : ndarray, shape (N, n_channels)
            Raw spectral data.

        Returns
        -------
        pred_raw : ndarray, shape (N, n_channels)
            Predicted autofluorescence spectrum in raw scale.
        """
        if self.model is None:
            raise ValueError("Model is not fitted. Call fit() first.")

        X_arcsinh = np.arcsinh(X / self.cofactor)
        X_scaled = (X_arcsinh - self.val_min) / (self.val_max - self.val_min)
        X_scaled = np.clip(X_scaled, 0, 1)

        tensor_X = torch.FloatTensor(X_scaled).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(tensor_X).cpu().numpy()

        pred_arcsinh = pred_scaled * (self.val_max - self.val_min) + self.val_min
        pred_raw = np.sinh(pred_arcsinh) * self.cofactor
        return pred_raw

    def transform(self, X):
        """アンミキシングを実行し、漏れ込み補正済みの係数を返す."""
        C = self._unmix(X)
        C_af = C[:, 0]
        C_stain = C[:, 1]
        C_stain_corrected = C_stain - self.slope * C_af - self.bg
        return C_af, C_stain_corrected

    def get_raw_coefficients(self, X):
        """補正前の生の係数 [c_af, c_stain] を返す."""
        return self._unmix(X)

    def remove_stain_component(self, X):
        """色素成分を除去し、純粋な自家蛍光(AF)スペクトルを返す"""
        return self._predict_af(X)

    def _unmix(self, X):
        """
        AutoEncoder を用いて C_af と C_stain の係数を計算する。
        S_AF, S_Stain に射影（OLS）することで、従来のポアソンモデルとスケールを一致させる。
        """
        pred_af = self._predict_af(X)
        pure_stain = X - pred_af
        pure_stain = np.maximum(pure_stain, 0)

        # 係数への変換 (OLS: y = c * x  => c = y*x / x*x)
        denom_af = np.dot(self.S_AF, self.S_AF) + 1e-9
        c_af = pred_af @ self.S_AF / denom_af

        denom_stain = np.dot(self.S_Stain, self.S_Stain) + 1e-9
        c_stain = pure_stain @ self.S_Stain / denom_stain

        return np.column_stack((c_af, c_stain))
