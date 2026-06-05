"""
Hyperparameter Search v2 — Multi-Architecture AutoEncoder Comparison

Compares three AutoEncoder architectures for autofluorescence unmixing
using Optuna (TPE Bayesian optimisation):

  mlp_baseline   — original AF_AutoEncoder (2-layer MLP, baseline)
  mlp_deeper     — DeeperAF_AutoEncoder (4-layer MLP + optional ResBlocks)
  transformer    — TransformerAF_AutoEncoder (self-attention encoder + MLP decoder)

Usage
-----
  # Quick smoke test (5 trials each arch, 5 % of data)
  python -m learning.hp_search_ae_v2 --n-trials 5 --subset-ratio 0.05

  # Full search (100 trials, 1 % of data — faster per trial)
  python -m learning.hp_search_ae_v2 --n-trials 100 --subset-ratio 0.01

Results are saved to:
  learning/results/hp_search_ae_v2_results.csv
  learning/results/hp_search_ae_v2_best.txt
"""

import os
import sys
import argparse
import random as py_random

import numpy as np
import pandas as pd
import torch
import optuna
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import COFACTOR
from src.unmix_autoencoder import AF_AutoEncoder, SpectralLoss, fix_seed
from src.unmix_autoencoder_v2 import (
    DeeperAF_AutoEncoder,
    TransformerAF_AutoEncoder,
    SpectralLossV2,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Data loading (reused from hp_search_ae.py)
# ---------------------------------------------------------------------------

def load_negative_data():
    """Load all negative-control spectral data found in analysis/results/."""
    import glob
    from config import RESULTS_DIR, EXPERIMENTS
    from src.unmix_spectral import get_spectral_features

    X_list = []
    for exp_folder, date_str in EXPERIMENTS.items():
        results_dir = os.path.join(RESULTS_DIR, date_str)
        patterns = [
            os.path.join(results_dir, "Negative_*", "*.csv"),
            os.path.join(results_dir, "negative_*", "*.csv"),
        ]
        csv_paths = []
        for p in patterns:
            csv_paths.extend(glob.glob(p))
        csv_paths = sorted(set(p for p in csv_paths if "scarf_embeddings" not in p))

        for path in csv_paths:
            df = pd.read_csv(path)
            wl_features = get_spectral_features(df)
            X_list.append(df[wl_features].values)

    return np.vstack(X_list) if X_list else None


def preprocess(X_raw, cofactor=None):
    """ArcSinh + MinMax → [0, 1]."""
    if cofactor is None:
        cofactor = COFACTOR
    X_arc = np.arcsinh(X_raw / cofactor)
    vmin, vmax = X_arc.min(), X_arc.max()
    X_scaled = np.clip((X_arc - vmin) / (vmax - vmin), 0.0, 1.0)
    return X_scaled, vmin, vmax


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def _make_model(arch: str, input_dim: int, params: dict) -> torch.nn.Module:
    """Instantiate the selected architecture from an Optuna params dict."""
    if arch == "mlp_baseline":
        return AF_AutoEncoder(
            input_dim=input_dim,
            hidden_dim=params["hidden_dim"],
            bottleneck_dim=params["bottleneck_dim"],
            dropout=params["dropout"],
        )
    elif arch == "mlp_deeper":
        return DeeperAF_AutoEncoder(
            input_dim=input_dim,
            hidden_dim=params["hidden_dim"],
            bottleneck_dim=params["bottleneck_dim"],
            dropout=params["dropout"],
            use_residual=params["use_residual"],
        )
    elif arch == "transformer":
        return TransformerAF_AutoEncoder(
            input_dim=input_dim,
            d_model=params["d_model"],
            nhead=params["nhead"],
            num_transformer_layers=params["num_transformer_layers"],
            dim_feedforward=params["dim_feedforward"],
            bottleneck_dim=params["bottleneck_dim"],
            dropout=params["dropout"],
        )
    else:
        raise ValueError(f"Unknown arch: {arch}")


def _make_criterion(arch: str, params: dict) -> torch.nn.Module:
    """Return the loss function for this trial."""
    alpha_cos  = params.get("alpha_cos",  params.get("alpha", 0.1))
    alpha_grad = params.get("alpha_grad", 0.0)
    use_huber  = params.get("use_huber",  False)
    if arch == "mlp_baseline":
        # Keep original SpectralLoss for apples-to-apples comparison
        return SpectralLoss(alpha=alpha_cos)
    return SpectralLossV2(alpha_cos=alpha_cos, alpha_grad=alpha_grad, use_huber=use_huber)


def train_and_eval(
    X_scaled: np.ndarray,
    arch: str,
    params: dict,
    epochs: int = 50,
    patience: int = 10,
    val_ratio: float = 0.2,
) -> float:
    """Train one trial and return best validation loss."""
    input_dim = X_scaled.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train / val split
    n = len(X_scaled)
    n_val = max(int(n * val_ratio), 1)
    idx = np.random.permutation(n)
    X_train = X_scaled[idx[n_val:]]
    X_val   = X_scaled[idx[:n_val]]

    t_train = torch.FloatTensor(X_train)
    t_val   = torch.FloatTensor(X_val).to(device)
    loader  = DataLoader(
        TensorDataset(t_train, t_train),
        batch_size=params["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    if len(loader) == 0:
        return float("inf")

    model     = _make_model(arch, input_dim, params).to(device)
    criterion = _make_criterion(arch, params)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val = float("inf")
    no_improve = 0

    for _ in range(epochs):
        model.train()
        for bx, _ in loader:
            bx = bx.to(device)
            optimizer.zero_grad()
            criterion(model(bx), bx).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(t_val), t_val).item()
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_val


# ---------------------------------------------------------------------------
# Optuna objective (arch-conditioned search space)
# ---------------------------------------------------------------------------

def build_objective(X_scaled: np.ndarray, epochs: int, patience: int):
    """Return the Optuna objective function (closure over data)."""

    def objective(trial: optuna.Trial) -> float:
        arch = trial.suggest_categorical(
            "arch", ["mlp_baseline", "mlp_deeper", "transformer"]
        )

        # --- Shared hyperparameters ---
        lr         = trial.suggest_float("lr",    1e-4, 1e-2, log=True)
        dropout    = trial.suggest_float("dropout", 0.0, 0.4)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
        alpha_cos  = trial.suggest_float("alpha_cos", 0.0, 0.5)

        params = dict(lr=lr, dropout=dropout, batch_size=batch_size, alpha_cos=alpha_cos)

        # --- Architecture-specific hyperparameters ---
        if arch == "mlp_baseline":
            params["hidden_dim"]     = trial.suggest_int("hidden_dim",     32, 256, log=True)
            params["bottleneck_dim"] = trial.suggest_int("bottleneck_dim",  4,  64, log=True)

        elif arch == "mlp_deeper":
            params["hidden_dim"]     = trial.suggest_int("deeper_hidden_dim",     64, 512, log=True)
            params["bottleneck_dim"] = trial.suggest_int("deeper_bottleneck_dim",  4,  64, log=True)
            params["use_residual"]   = trial.suggest_categorical("use_residual", [True, False])
            params["alpha_grad"]     = trial.suggest_float("alpha_grad", 0.0, 0.2)
            params["use_huber"]      = trial.suggest_categorical("use_huber", [True, False])

        elif arch == "transformer":
            # nhead must divide d_model — use a safe pair list
            nhead_dmodel = trial.suggest_categorical(
                "nhead_dmodel", ["2_32", "4_64", "4_128", "8_64", "8_128"]
            )
            nhead_str, dmodel_str = nhead_dmodel.split("_")
            params["nhead"]   = int(nhead_str)
            params["d_model"] = int(dmodel_str)
            params["num_transformer_layers"] = trial.suggest_int("num_transformer_layers", 1, 4)
            params["dim_feedforward"]        = trial.suggest_categorical(
                "dim_feedforward", [64, 128, 256]
            )
            params["bottleneck_dim"] = trial.suggest_int("tf_bottleneck_dim", 4, 32, log=True)
            params["alpha_grad"]     = trial.suggest_float("tf_alpha_grad", 0.0, 0.2)
            params["use_huber"]      = trial.suggest_categorical("tf_use_huber", [True, False])

        try:
            return train_and_eval(X_scaled, arch, params, epochs=epochs, patience=patience)
        except Exception as exc:
            print(f"  [Trial {trial.number}] ERROR ({arch}): {exc}")
            raise optuna.TrialPruned()

    return objective


# ---------------------------------------------------------------------------
# Run search
# ---------------------------------------------------------------------------

def run_search(
    X_raw: np.ndarray,
    n_trials: int = 60,
    subset_ratio: float = 0.01,
    search_epochs: int = 50,
    patience: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Run Optuna search and return the results DataFrame."""
    fix_seed(seed)

    n_total  = len(X_raw)
    n_subset = max(int(n_total * subset_ratio), 128)
    idx      = np.random.choice(n_total, size=n_subset, replace=False)
    X_sub, _, _ = preprocess(X_raw[idx])

    print("=" * 70)
    print("  AutoEncoder Architecture Search v2  (Optuna TPE)")
    print("=" * 70)
    print(f"  Total events     : {n_total:,}")
    print(f"  Subset ratio     : {subset_ratio:.1%}  →  {n_subset:,} events")
    print(f"  Epochs per trial : {search_epochs}")
    print(f"  Patience         : {patience}")
    print(f"  Total trials     : {n_trials}")
    print(f"  Architectures    : mlp_baseline | mlp_deeper | transformer")
    print("=" * 70)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        build_objective(X_sub, search_epochs, patience),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # ---- Report ----
    df = study.trials_dataframe()
    df = df.sort_values("value").reset_index(drop=True)

    print("\n" + "=" * 70)
    print("  Results by architecture")
    print("=" * 70)
    for arch in ["mlp_baseline", "mlp_deeper", "transformer"]:
        sub = df[df.get("params_arch", pd.Series(dtype=str)) == arch]
        if len(sub) == 0:
            continue
        best_row = sub.iloc[0]
        print(f"\n  [{arch}]  best val loss = {best_row['value']:.6f}"
              f"  (trial #{int(best_row['number'])})")

    print("\n" + "=" * 70)
    best = study.best_trial
    print(f"  ★ Overall best: arch={best.params.get('arch', 'N/A')}"
          f"  val loss={best.value:.6f}")
    for k, v in best.params.items():
        if k != "arch":
            print(f"      {k:<30s} = {v}")
    print("=" * 70)

    # ---- Save ----
    learning_dir  = os.path.dirname(os.path.abspath(__file__))
    result_dir    = os.path.join(learning_dir, "results")
    os.makedirs(result_dir, exist_ok=True)

    csv_path = os.path.join(result_dir, "hp_search_ae_v2_results.csv")
    df.to_csv(csv_path, index=False)

    txt_path = os.path.join(result_dir, "hp_search_ae_v2_best.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Best architecture: {best.params.get('arch', 'N/A')}\n")
        f.write(f"Best val loss    : {best.value:.6f}\n\n")
        f.write("Parameters:\n")
        for k, v in best.params.items():
            f.write(f"  {k:<30s} = {v}\n")

    print(f"\n  Full results → {csv_path}")
    print(f"  Best params  → {txt_path}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-architecture AutoEncoder HP search (v2)"
    )
    parser.add_argument("--n-trials",     type=int,   default=60,   help="Number of Optuna trials")
    parser.add_argument("--subset-ratio", type=float, default=0.01, help="Fraction of data used per trial")
    parser.add_argument("--epochs",       type=int,   default=50,   help="Max epochs per trial")
    parser.add_argument("--patience",     type=int,   default=10,   help="Early stopping patience")
    parser.add_argument("--seed",         type=int,   default=42,   help="Random seed")
    args = parser.parse_args()

    print("Loading negative control data...")
    X_neg = load_negative_data()
    if X_neg is None or len(X_neg) == 0:
        print("No negative control data found. Run the pipeline first.")
        sys.exit(1)
    print(f"Loaded {len(X_neg):,} events  ({X_neg.shape[1]} channels)\n")

    run_search(
        X_raw=X_neg,
        n_trials=args.n_trials,
        subset_ratio=args.subset_ratio,
        search_epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
