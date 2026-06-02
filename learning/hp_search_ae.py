"""
Hyperparameter Search for AutoEncoder (Autofluorescence Unmixing)

hp_search.py (SCARF用) と同様の構造で、Optuna (TPE) による
AutoEncoder のハイパーパラメータ探索を行います。

探索対象:
    - bottleneck_dim: ボトルネック次元 [4, 64]
    - hidden_dim: 隠れ層の幅 [32, 256]
    - lr: 学習率 [1e-4, 1e-2]
    - dropout: ドロップアウト率 [0.0, 0.5]
    - alpha: コサイン類似度損失の重み [0.0, 0.5]
    - batch_size: バッチサイズ [64, 128, 256, 512]

Usage:
    python -m learning.hp_search_ae
"""

import os
import sys
import random as py_random

import numpy as np
import pandas as pd
import torch
import optuna
from torch.utils.data import DataLoader, TensorDataset

# Add project root to system path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import COFACTOR
from src.unmix_autoencoder import AF_AutoEncoder, SpectralLoss, fix_seed


def load_negative_data():
    """
    Automatically discover and load negative (unstained) spectral data
    from the analysis results directory.

    Returns
    -------
    X_neg : ndarray, shape (N, n_channels)
        Raw spectral data from negative control samples.
    """
    import glob
    from config import RESULTS_DIR, EXPERIMENTS
    from src.unmix_spectral import get_spectral_features

    X_neg_list = []

    for exp_folder, date_str in EXPERIMENTS.items():
        results_dir = os.path.join(RESULTS_DIR, date_str)
        neg_csv_paths = glob.glob(os.path.join(results_dir, "Negative_*", "*.csv")) + \
                        glob.glob(os.path.join(results_dir, "negative_*", "*.csv"))
        neg_csv_paths = sorted(list(set([p for p in neg_csv_paths if "scarf_embeddings" not in p])))

        for path in neg_csv_paths:
            df = pd.read_csv(path)
            wl_features = get_spectral_features(df)
            X_neg_list.append(df[wl_features].values)

    if not X_neg_list:
        return None

    return np.vstack(X_neg_list)


def preprocess_for_ae(X_raw, cofactor=None):
    """Apply ArcSinh + MinMax scaling for AutoEncoder input.

    Parameters
    ----------
    X_raw : ndarray, shape (N, n_channels)
        Raw spectral data.
    cofactor : float or None
        ArcSinh cofactor. Defaults to config.COFACTOR.

    Returns
    -------
    X_scaled : ndarray, shape (N, n_channels)
        Scaled data in [0, 1].
    val_min : float
        Minimum value after ArcSinh (for inverse transform).
    val_max : float
        Maximum value after ArcSinh (for inverse transform).
    """
    if cofactor is None:
        cofactor = COFACTOR

    X_arcsinh = np.arcsinh(X_raw / cofactor)
    val_min = np.min(X_arcsinh)
    val_max = np.max(X_arcsinh)
    X_scaled = (X_arcsinh - val_min) / (val_max - val_min)
    X_scaled = np.clip(X_scaled, 0, 1)
    return X_scaled, val_min, val_max


def train_one_config(X_scaled, bottleneck_dim=16, hidden_dim=128,
                     lr=0.001, dropout=0.1, alpha=0.1,
                     batch_size=256, epochs=50, patience=10,
                     val_ratio=0.2):
    """Train AutoEncoder with a specific configuration and return best validation loss.

    Parameters
    ----------
    X_scaled : ndarray, shape (N, n_channels)
        Preprocessed (ArcSinh+MinMax scaled) negative control data.
    bottleneck_dim : int
        Latent bottleneck dimension.
    hidden_dim : int
        Hidden layer width.
    lr : float
        Learning rate.
    dropout : float
        Dropout probability.
    alpha : float
        Weight for cosine similarity loss.
    batch_size : int
        Training batch size.
    epochs : int
        Maximum training epochs.
    patience : int
        Early stopping patience.
    val_ratio : float
        Fraction of data used for validation.

    Returns
    -------
    best_val_loss : float
        Best validation loss achieved during training.
    """
    input_dim = X_scaled.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train / Validation split
    n_samples = len(X_scaled)
    n_val = max(int(n_samples * val_ratio), 1)
    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    X_train = X_scaled[train_indices]
    X_val = X_scaled[val_indices]

    tensor_train = torch.FloatTensor(X_train)
    tensor_val = torch.FloatTensor(X_val).to(device)
    dataset = TensorDataset(tensor_train, tensor_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    if len(loader) == 0:
        return float('inf')

    model = AF_AutoEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
        dropout=dropout
    ).to(device)

    criterion = SpectralLoss(alpha=alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # Training
        model.train()
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(tensor_val)
            val_loss = criterion(val_pred, tensor_val).item()

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    return best_val_loss


def run_search(X_raw, n_iterations=100, subset_ratio=0.01, search_epochs=50,
               patience=10, seed=42):
    """
    Run Bayesian optimization over AutoEncoder hyperparameters using Optuna (TPE).

    Parameters
    ----------
    X_raw : ndarray, shape (N, n_channels)
        Raw negative control spectral data.
    n_iterations : int
        Number of Optuna trials.
    subset_ratio : float
        Fraction of data to use for search (for speed).
    search_epochs : int
        Maximum epochs per trial.
    patience : int
        Early stopping patience per trial.
    seed : int
        Random seed.

    Returns
    -------
    df_results : DataFrame
        Optuna trial results sorted by loss.
    """
    fix_seed(seed)

    # --- 1. Extract subset ---
    n_total = len(X_raw)
    n_subset = max(int(n_total * subset_ratio), 64)  # At least 64 events
    indices = np.random.choice(n_total, size=n_subset, replace=False)
    X_subset_raw = X_raw[indices]

    # Preprocess
    X_subset_scaled, _, _ = preprocess_for_ae(X_subset_raw)

    print(f"=== AutoEncoder Hyperparameter Search (Optuna TPE) ===")
    print(f"Total events: {n_total}")
    print(f"Subset ratio: {subset_ratio:.1%} → {n_subset} events")
    print(f"Search epochs per trial: {search_epochs}")
    print(f"Early stopping patience: {patience}")
    print(f"Number of trials: {n_iterations}")
    print(f"Search space:")
    print(f"  bottleneck_dim ∈ [4, 64]    (log)")
    print(f"  hidden_dim     ∈ [32, 256]   (log)")
    print(f"  lr             ∈ [1e-4, 1e-2] (log)")
    print(f"  dropout        ∈ [0.0, 0.5]")
    print(f"  alpha          ∈ [0.0, 0.5]")
    print(f"  batch_size     ∈ {{64, 128, 256, 512}}")
    print(f"{'='*60}")

    def objective(trial):
        bottleneck_dim = trial.suggest_int('bottleneck_dim', 4, 64, log=True)
        hidden_dim = trial.suggest_int('hidden_dim', 32, 256, log=True)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        alpha = trial.suggest_float('alpha', 0.0, 0.5)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])

        try:
            best_val_loss = train_one_config(
                X_subset_scaled,
                bottleneck_dim=bottleneck_dim,
                hidden_dim=hidden_dim,
                lr=lr,
                dropout=dropout,
                alpha=alpha,
                batch_size=batch_size,
                epochs=search_epochs,
                patience=patience,
            )
        except Exception as e:
            print(f"  Trial {trial.number}: ERROR: {e}")
            raise optuna.TrialPruned()

        return best_val_loss

    # --- 2. Run Optuna Optimization ---
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_iterations)

    # --- 3. Report results ---
    df_results = study.trials_dataframe()
    df_results = df_results.sort_values('value').reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"=== Search Complete ===")
    print(f"{'='*60}")
    print(f"\nTop 10 configurations:")
    cols_to_show = [
        'number', 'params_bottleneck_dim', 'params_hidden_dim',
        'params_lr', 'params_dropout', 'params_alpha',
        'params_batch_size', 'value', 'duration'
    ]
    available_cols = [c for c in cols_to_show if c in df_results.columns]
    print(df_results[available_cols].head(10).to_string(index=False))

    best_trial = study.best_trial
    best_bd = best_trial.params['bottleneck_dim']
    best_hd = best_trial.params['hidden_dim']
    best_lr = best_trial.params['lr']
    best_do = best_trial.params['dropout']
    best_al = best_trial.params['alpha']
    best_bs = best_trial.params['batch_size']

    print(f"\n{'='*60}")
    print(f"★ Best configuration:")
    print(f"   bottleneck_dim = {best_bd}")
    print(f"   hidden_dim     = {best_hd}")
    print(f"   lr             = {best_lr:.5f}")
    print(f"   dropout        = {best_do:.3f}")
    print(f"   alpha          = {best_al:.3f}")
    print(f"   batch_size     = {best_bs}")
    print(f"   Val Loss       = {best_trial.value:.6f}")
    print(f"{'='*60}")
    print(f"\nTo use these params in the pipeline, update the AutoEncoderUnmixer call:")
    print(f"  AutoEncoderUnmixer(")
    print(f"      bottleneck_dim={best_bd}, hidden_dim={best_hd},")
    print(f"      lr={best_lr:.5f}, dropout={best_do:.3f},")
    print(f"      alpha={best_al:.3f}, batch_size={best_bs}")
    print(f"  )")

    # Save results CSV
    learning_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(learning_dir, "results")
    os.makedirs(result_dir, exist_ok=True)
    results_path = os.path.join(result_dir, "hp_search_ae_results.csv")
    df_results.to_csv(results_path, index=False)
    print(f"\nFull results saved to: {results_path}")

    return df_results


def main():
    print("Loading negative control data for AutoEncoder HP search...")
    X_neg = load_negative_data()

    if X_neg is None or len(X_neg) == 0:
        print("No negative control data found. Please run the data pipeline first.")
        sys.exit(1)

    print(f"Loaded {len(X_neg)} negative events ({X_neg.shape[1]} channels)")

    run_search(
        X_raw=X_neg,
        n_iterations=100,
        subset_ratio=0.01,
        search_epochs=50,
        patience=10,
        seed=42
    )


if __name__ == '__main__':
    main()
