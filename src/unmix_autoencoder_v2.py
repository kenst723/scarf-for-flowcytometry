"""
unmix_autoencoder_v2.py — Experimental AutoEncoder Architectures

This module defines two alternative AutoEncoder models for autofluorescence
spectrum reconstruction, intended to be compared against the baseline
AF_AutoEncoder in unmix_autoencoder.py via hp_search_ae_v2.py.

It also provides TransformerAutoEncoderUnmixer — a drop-in replacement for
AutoEncoderUnmixer that uses TransformerAF_AutoEncoder as its backbone and
can be integrated into the pipeline via --method transformer.

Models
------
DeeperAF_AutoEncoder
    A 4-layer MLP AutoEncoder with residual (skip) connections.
    Wider and deeper than the baseline, better suited to capture
    non-linear spectral structure.

TransformerAF_AutoEncoder
    Treats each spectral channel as a token and applies multi-head
    self-attention (TransformerEncoder) to capture inter-channel
    dependencies before compressing to a bottleneck.

SpectralLossV2
    Extended spectral loss with optional Huber loss and spectral
    gradient (finite-difference) regularisation term.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    事前活性化を行うクラス
    1. batchNorm 平均0 分散1にすることでデータをきれいに
    2. LeakyReLu 0以下の数値を許容する
    3. Linear 線形変換
    4. Dropout ドロップアウト(データの間引き)
    5. skip 接続(出力に対して変換前のデータをそのまま足し合わせる)

    Parameters
    ----------
    in_dim : int
        入力次元.
    out_dim : int
        出力次元.
    dropout : float
        ドロップアウト率.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim)
        self.act  = nn.LeakyReLU(0.1)
        self.fc   = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(dropout)

        # Project skip connection when dimensions change
        self.skip = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.drop(self.fc(self.act(self.norm(x))))
        return out + residual


# ---------------------------------------------------------------------------
# Option A: Deeper / Wider MLP AutoEncoder
# ---------------------------------------------------------------------------

class DeeperAF_AutoEncoder(nn.Module):
    """4-layer MLP AutoEncoder with residual connections.

    Architecture (encoder side):
        input → hidden → hidden//2 → hidden//4 → hidden//8 → bottleneck

    Each encoder stage uses a ResidualBlock to ease gradient flow.
    The decoder is the mirror image without skip connections (simpler).

    Parameters
    ----------
    input_dim : int
        Number of spectral channels (e.g. 34 for Sony SA3800).
    hidden_dim : int
        Width of the first (widest) hidden layer (default: 256).
    bottleneck_dim : int
        Dimension of the latent bottleneck (default: 16).
    dropout : float
        Dropout probability inside each block (default: 0.1).
    use_residual : bool
        Whether to use skip connections in the encoder. If False,
        plain Linear + BN + LeakyReLU blocks are used instead.
    """

    def __init__(
        self,
        input_dim: int = 34,
        hidden_dim: int = 256,
        bottleneck_dim: int = 16,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual

        d1 = hidden_dim          # e.g. 256
        d2 = max(hidden_dim // 2, bottleneck_dim * 2)   # e.g. 128
        d3 = max(hidden_dim // 4, bottleneck_dim * 2)   # e.g.  64
        d4 = max(hidden_dim // 8, bottleneck_dim * 2)   # e.g.  32

        # ---- Encoder ----
        self.enc_in = nn.Sequential(
            nn.Linear(input_dim, d1),
            nn.BatchNorm1d(d1),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )
        if use_residual:
            self.enc_block1 = ResidualBlock(d1, d2, dropout)
            self.enc_block2 = ResidualBlock(d2, d3, dropout)
            self.enc_block3 = ResidualBlock(d3, d4, dropout)
        else:
            self.enc_block1 = nn.Sequential(nn.Linear(d1, d2), nn.BatchNorm1d(d2), nn.LeakyReLU(0.1), nn.Dropout(dropout))
            self.enc_block2 = nn.Sequential(nn.Linear(d2, d3), nn.BatchNorm1d(d3), nn.LeakyReLU(0.1), nn.Dropout(dropout))
            self.enc_block3 = nn.Sequential(nn.Linear(d3, d4), nn.BatchNorm1d(d4), nn.LeakyReLU(0.1), nn.Dropout(dropout))

        self.enc_out = nn.Sequential(
            nn.Linear(d4, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.LeakyReLU(0.1),
        )

        # ---- Decoder (plain MLP, mirror) ----
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, d4),
            nn.BatchNorm1d(d4),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(d4, d3),
            nn.BatchNorm1d(d3),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(d3, d2),
            nn.BatchNorm1d(d2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(d2, d1),
            nn.BatchNorm1d(d1),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(d1, input_dim),
            nn.Sigmoid(),
        )

        # Store for logging
        self._dims = (input_dim, d1, d2, d3, d4, bottleneck_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc_in(x)
        x = self.enc_block1(x)
        x = self.enc_block2(x)
        x = self.enc_block3(x)
        return self.enc_out(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decoder(z)

    def arch_str(self) -> str:
        d = self._dims
        res = " [ResBlock]" if self.use_residual else ""
        return f"{d[0]}→{d[1]}→{d[2]}→{d[3]}→{d[4]}{res}→[{d[5]}]→...→{d[0]}"


# ---------------------------------------------------------------------------
# Option B: Transformer-based AutoEncoder
# ---------------------------------------------------------------------------

class _LearnablePositionalEncoding(nn.Module):
    """Learnable positional embedding for a fixed sequence length."""

    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe


class TransformerAF_AutoEncoder(nn.Module):
    """Transformer-based AutoEncoder for spectral unmixing.

    Each of the ``input_dim`` spectral channels is treated as a separate
    token.  A learnable linear projection maps the scalar intensity of each
    channel to a ``d_model``-dimensional embedding; learnable positional
    encodings are added to preserve channel order.  The token sequence is
    processed by a stack of ``TransformerEncoder`` layers, after which the
    output tokens are flattened and projected to the bottleneck.

    The decoder is a lightweight MLP (no cross-attention) for efficiency.

    Parameters
    ----------
    input_dim : int
        Number of spectral channels (sequence length). Default: 34.
    d_model : int
        Transformer embedding dimension. Must be divisible by ``nhead``.
        Default: 64.
    nhead : int
        Number of self-attention heads. Default: 4.
    num_transformer_layers : int
        Number of TransformerEncoder layers. Default: 2.
    dim_feedforward : int
        Inner dimension of the Transformer FFN sublayer. Default: 128.
    bottleneck_dim : int
        Dimension of the latent bottleneck. Default: 16.
    dropout : float
        Dropout probability in both Transformer and MLP decoder. Default: 0.1.
    """

    def __init__(
        self,
        input_dim: int = 34,
        d_model: int = 64,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        dim_feedforward: int = 128,
        bottleneck_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % nhead == 0, (
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        )
        self.input_dim = input_dim
        self.d_model = d_model
        self._meta = dict(
            input_dim=input_dim, d_model=d_model, nhead=nhead,
            num_transformer_layers=num_transformer_layers,
            dim_feedforward=dim_feedforward, bottleneck_dim=bottleneck_dim,
        )

        # ---- Token embedding: scalar → d_model ----
        # Each channel value is first projected independently
        self.channel_embed = nn.Linear(1, d_model)

        # Learnable positional encoding (one embedding per channel position)
        self.pos_enc = _LearnablePositionalEncoding(input_dim, d_model)

        # ---- Transformer Encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # (B, seq, d_model)
            norm_first=True,    # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
            enable_nested_tensor=False,
        )

        # ---- Bottleneck projection ----
        flat_dim = input_dim * d_model
        self.bottleneck = nn.Sequential(
            nn.Linear(flat_dim, bottleneck_dim * 4),
            nn.LayerNorm(bottleneck_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim * 4, bottleneck_dim),
        )

        # ---- MLP Decoder ----
        mid = max(bottleneck_dim * 4, 64)
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, mid),
            nn.BatchNorm1d(mid),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(mid, mid * 2),
            nn.BatchNorm1d(mid * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(mid * 2, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode (B, input_dim) → (B, bottleneck_dim)."""
        B = x.size(0)
        # (B, C) → (B, C, 1) → (B, C, d_model)
        tokens = self.channel_embed(x.unsqueeze(-1))
        tokens = self.pos_enc(tokens)                   # (B, C, d_model)
        tokens = self.transformer(tokens)               # (B, C, d_model)
        flat   = tokens.reshape(B, -1)                  # (B, C * d_model)
        return self.bottleneck(flat)                    # (B, bottleneck_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decoder(z)

    def arch_str(self) -> str:
        m = self._meta
        return (
            f"{m['input_dim']}ch × d_model={m['d_model']} "
            f"→ TransformerEncoder(nhead={m['nhead']}, layers={m['num_transformer_layers']}, "
            f"ffn={m['dim_feedforward']}) → [{m['bottleneck_dim']}] → MLP → {m['input_dim']}"
        )


# ---------------------------------------------------------------------------
# SpectralLossV2
# ---------------------------------------------------------------------------

class SpectralLossV2(nn.Module):
    """Extended spectral reconstruction loss.

    Loss = α_mse * MSE  +  α_cos * (1 - CosSim)  +  α_grad * GradMSE

    where ``GradMSE`` penalises differences in the finite-difference
    gradient (channel-to-channel slope) of the predicted vs. true spectrum.
    This encourages the model to reproduce spectral *shape* not just
    per-channel intensities.

    Parameters
    ----------
    alpha_cos : float
        Weight for the cosine similarity term (default: 0.1).
    alpha_grad : float
        Weight for the spectral gradient MSE term (default: 0.0 = disabled).
    use_huber : bool
        If True, replace MSE with Huber loss (delta=0.1) for robustness to
        outlier cells (default: False).
    """

    def __init__(
        self,
        alpha_cos: float = 0.1,
        alpha_grad: float = 0.0,
        use_huber: bool = False,
    ):
        super().__init__()
        self.alpha_cos  = alpha_cos
        self.alpha_grad = alpha_grad
        self.use_huber  = use_huber

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Reconstruction loss
        if self.use_huber:
            recon = F.huber_loss(pred, target, delta=0.1)
        else:
            recon = F.mse_loss(pred, target)

        # Cosine similarity term
        cos_sim = F.cosine_similarity(pred, target, dim=1).mean()
        cos_loss = self.alpha_cos * (1.0 - cos_sim)

        # Spectral gradient term
        if self.alpha_grad > 0.0:
            grad_pred   = pred[:, 1:] - pred[:, :-1]
            grad_target = target[:, 1:] - target[:, :-1]
            grad_loss   = self.alpha_grad * F.mse_loss(grad_pred, grad_target)
        else:
            grad_loss = torch.tensor(0.0, device=pred.device)

        return recon + cos_loss + grad_loss


# ---------------------------------------------------------------------------
# Pipeline-ready Unmixer: TransformerAutoEncoderUnmixer
# ---------------------------------------------------------------------------

class TransformerAutoEncoderUnmixer:
    """Pipeline-compatible unmixer backed by TransformerAF_AutoEncoder.

    Drop-in replacement for AutoEncoderUnmixer. Exposes the same public
    interface (fit / transform / get_raw_coefficients / remove_stain_component)
    so it can be used in run_unmixing_group without changing call-sites.

    Default hyperparameters reflect the best configuration found by
    hp_search_ae_v2.py (100 trials, nhead=8, d_model=128, layers=3).

    Parameters
    ----------
    epochs : int
        Maximum training epochs (default: 150).
    batch_size : int
        Training batch size (default: 128).
    lr : float
        Adam learning rate (default: 0.0018).
    d_model : int
        Transformer embedding dimension per channel token (default: 128).
    nhead : int
        Number of self-attention heads; must divide d_model (default: 8).
    num_transformer_layers : int
        Number of TransformerEncoder layers (default: 3).
    dim_feedforward : int
        Inner FFN dimension of each Transformer layer (default: 128).
    bottleneck_dim : int
        Latent bottleneck dimension (default: 12).
    dropout : float
        Dropout probability in Transformer and MLP decoder (default: 0.30).
    alpha_cos : float
        Cosine similarity loss weight (default: 0.031).
    alpha_grad : float
        Spectral gradient MSE loss weight (default: 0.002).
    use_huber : bool
        Use Huber loss instead of MSE for reconstruction (default: True).
    patience : int
        Early-stopping patience in epochs (default: 15).
    val_ratio : float
        Fraction of training data used for validation (default: 0.2).
    model_save_path : str or None
        Path to save the trained checkpoint (.pth).
    seed : int
        Random seed for reproducibility (default: 42).
    """

    def __init__(
        self,
        epochs=150,
        batch_size=128,
        lr=0.0018,
        d_model=128,
        nhead=8,
        num_transformer_layers=3,
        dim_feedforward=128,
        bottleneck_dim=12,
        dropout=0.30,
        alpha_cos=0.031,
        alpha_grad=0.002,
        use_huber=True,
        patience=15,
        val_ratio=0.2,
        model_save_path=None,
        seed=42,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.d_model = d_model
        self.nhead = nhead
        self.num_transformer_layers = num_transformer_layers
        self.dim_feedforward = dim_feedforward
        self.bottleneck_dim = bottleneck_dim
        self.dropout = dropout
        self.alpha_cos = alpha_cos
        self.alpha_grad = alpha_grad
        self.use_huber = use_huber
        self.patience = patience
        self.val_ratio = val_ratio
        self.model_save_path = model_save_path
        self.seed = seed

        self.model = None
        self.cofactor = 150.0
        self.val_min = 0.0
        self.val_max = 1.0

        # Interface compatibility with AutoEncoderUnmixer
        self.S_AF = None
        self.S_Stain = None
        self.slope = 0.0
        self.bg = 0.0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X_neg, X_stain):
        """Train on negative control data.

        Parameters
        ----------
        X_neg : ndarray, shape (n_neg, n_channels)
            Negative (unstained) spectral data.
        X_stain : ndarray, shape (n_stain, n_channels)
            Stained spectral data (for PoissonUnmixer reference fitting).

        Returns
        -------
        self
        """
        from src.unmix_autoencoder import fix_seed
        from src.unmix_spectral import PoissonUnmixer

        fix_seed(self.seed)

        # 1. Derive S_AF, S_Stain, slope from PoissonUnmixer
        poisson = PoissonUnmixer()
        poisson.fit(X_neg, X_stain)
        self.S_AF    = poisson.S_AF
        self.S_Stain = poisson.S_Stain
        self.slope   = poisson.slope
        self.bg      = poisson.bg

        # 2. Build model
        input_dim = X_neg.shape[1]
        self.model = TransformerAF_AutoEncoder(
            input_dim=input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_transformer_layers=self.num_transformer_layers,
            dim_feedforward=self.dim_feedforward,
            bottleneck_dim=self.bottleneck_dim,
            dropout=self.dropout,
        ).to(self.device)

        # 3. Denoising Data Augmentation
        # X_stainの強さ分布から、最大強度を推定して人工的な色素ノイズを作る
        c_stain_raw = poisson.get_raw_coefficients(X_stain)[:, 1]
        c_max = float(np.percentile(c_stain_raw, 99))
        
        np.random.seed(self.seed)
        random_c = np.random.uniform(0, c_max, size=(len(X_neg), 1))
        X_synthetic = X_neg + random_c * self.S_Stain[None, :]
        
        # 入力(Noisy)と正解(Clean)のペアを作成
        X_input_raw  = np.vstack([X_neg, X_synthetic])
        X_target_raw = np.vstack([X_neg, X_neg])

        # 4. Preprocessing: ArcSinh + MinMax -> [0, 1]
        X_input_arc  = np.arcsinh(X_input_raw  / self.cofactor)
        X_target_arc = np.arcsinh(X_target_raw / self.cofactor)
        X_stain_arc  = np.arcsinh(X_stain / self.cofactor)
        
        self.val_min = float(np.min(X_input_arc))
        self.val_max = float(max(np.max(X_input_arc), np.max(X_stain_arc)))
        
        X_input_scaled  = np.clip((X_input_arc  - self.val_min) / (self.val_max - self.val_min), 0.0, 1.0)
        X_target_scaled = np.clip((X_target_arc - self.val_min) / (self.val_max - self.val_min), 0.0, 1.0)

        # 5. Train / validation split
        n = len(X_input_scaled)
        n_val = max(int(n * self.val_ratio), 1)
        idx = np.random.permutation(n)
        
        X_train_in  = X_input_scaled[idx[n_val:]]
        X_train_tgt = X_target_scaled[idx[n_val:]]
        X_val_in    = X_input_scaled[idx[:n_val]]
        X_val_tgt   = X_target_scaled[idx[:n_val]]

        t_val_in  = torch.FloatTensor(X_val_in).to(self.device)
        t_val_tgt = torch.FloatTensor(X_val_tgt).to(self.device)
        
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train_in), torch.FloatTensor(X_train_tgt)),
            batch_size=self.batch_size,
            shuffle=True,
        )

        criterion = SpectralLossV2(
            alpha_cos=self.alpha_cos,
            alpha_grad=self.alpha_grad,
            use_huber=self.use_huber,
        )
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=7
        )

        best_val   = float("inf")
        best_state = None
        no_improve = 0

        n_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"    [TransformerAE] Training on {self.device} -- "
            f"params={n_params:,}  epochs<={self.epochs}  patience={self.patience}"
        )
        print(f"    [TransformerAE] {self.model.arch_str()}")

        import sys
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            num_batches = len(loader)
            for i, (bx, by) in enumerate(loader):
                bx = bx.to(self.device)
                by = by.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(bx), by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * bx.size(0)
                
                # Print batch progress
                sys.stdout.write(f"\r      Epoch {epoch+1:4d}/{self.epochs} | Batch {i+1:3d}/{num_batches:3d}")
                sys.stdout.flush()
                
            train_loss /= len(X_train_in)

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(t_val_in)
                val_loss = criterion(val_pred, t_val_tgt).item()
            scheduler.step(val_loss)

            # Append epoch summary to the current progress line and break to a new line
            lr_now = optimizer.param_groups[0]["lr"]
            sys.stdout.write(f" | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr_now:.2e}\n")
            sys.stdout.flush()

            if val_loss < best_val:
                best_val   = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"      Early stopping at epoch {epoch+1}  (best val={best_val:.6f})")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        self.model.eval()

        if self.model_save_path:
            self._save_checkpoint(self.model_save_path, input_dim)
            print(f"    [TransformerAE] Checkpoint saved to {self.model_save_path}")

        return self

    def _save_checkpoint(self, path, input_dim):
        """Save full checkpoint including model weights and preprocessing params."""
        torch.save({
            "model_state_dict":        self.model.state_dict(),
            "input_dim":               input_dim,
            "d_model":                 self.d_model,
            "nhead":                   self.nhead,
            "num_transformer_layers":  self.num_transformer_layers,
            "dim_feedforward":         self.dim_feedforward,
            "bottleneck_dim":          self.bottleneck_dim,
            "dropout":                 self.dropout,
            "val_min":                 self.val_min,
            "val_max":                 self.val_max,
            "cofactor":                self.cofactor,
            "S_AF":                    self.S_AF,
            "S_Stain":                 self.S_Stain,
            "slope":                   self.slope,
            "bg":                      self.bg,
        }, path)

    def load_model(self, path):
        """Load model from checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.val_min  = ckpt.get("val_min",  self.val_min)
        self.val_max  = ckpt.get("val_max",  self.val_max)
        self.cofactor = ckpt.get("cofactor", self.cofactor)
        self.S_AF     = ckpt.get("S_AF",     self.S_AF)
        self.S_Stain  = ckpt.get("S_Stain",  self.S_Stain)
        self.slope    = ckpt.get("slope",    self.slope)
        self.bg       = ckpt.get("bg",       self.bg)
        self.model = TransformerAF_AutoEncoder(
            input_dim=ckpt.get("input_dim", 34),
            d_model=ckpt.get("d_model", 128),
            nhead=ckpt.get("nhead", 8),
            num_transformer_layers=ckpt.get("num_transformer_layers", 3),
            dim_feedforward=ckpt.get("dim_feedforward", 128),
            bottleneck_dim=ckpt.get("bottleneck_dim", 12),
            dropout=ckpt.get("dropout", 0.30),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def _predict_af(self, X):
        """Predict autofluorescence component (raw scale)."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        X_arc    = np.arcsinh(X / self.cofactor)
        X_scaled = np.clip(
            (X_arc - self.val_min) / (self.val_max - self.val_min), 0.0, 1.0
        )
        t = torch.FloatTensor(X_scaled).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(t).cpu().numpy()
        pred_arc = pred_scaled * (self.val_max - self.val_min) + self.val_min
        return np.sinh(pred_arc) * self.cofactor

    def _unmix(self, X):
        """OLS projection onto S_AF / S_Stain (same as AutoEncoderUnmixer)."""
        pred_af    = self._predict_af(X)
        pure_stain = np.maximum(X - pred_af, 0)
        denom_af    = np.dot(self.S_AF,    self.S_AF)    + 1e-9
        denom_stain = np.dot(self.S_Stain, self.S_Stain) + 1e-9
        c_af    = pred_af    @ self.S_AF    / denom_af
        c_stain = pure_stain @ self.S_Stain / denom_stain
        return np.column_stack((c_af, c_stain))

    def transform(self, X):
        """Return (C_af, C_stain_corrected) -- leakage-corrected coefficients."""
        C = self._unmix(X)
        return C[:, 0], C[:, 1] - self.slope * C[:, 0] - self.bg

    def get_raw_coefficients(self, X):
        """Return raw [c_af, c_stain] before leakage correction."""
        return self._unmix(X)

    def remove_stain_component(self, X):
        """Return the pure autofluorescence spectrum (stain component removed)."""
        return self._predict_af(X)
