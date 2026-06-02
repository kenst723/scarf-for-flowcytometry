"""
Hyperparameter Search for Periodic Encoding (k, σ)

論文の手法に従い、以下のプロセスでハイパーパラメータを探索します:
1. 全データの1%をサブセットとして抽出
2. k (num_frequencies) と σ (sigma) のランダムな組み合わせを100回試行
3. 各組み合わせで短いエポック数の学習を行い、最終Lossを記録
4. 最もLossが低かった組み合わせを「最適値」として報告

Usage:
    python -m learning.hp_search
"""

import os
import sys
import time
import random as py_random

import numpy as np
import pandas as pd
import fcsparser
import torch
import optuna
from sklearn.preprocessing import StandardScaler

# Add project root and pytorch-scarf directory to system path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "pytorch-scarf"))

from config import COFACTOR
from scarf.loss import NTXent
from scarf.model import SCARF
from scarf.dataset import SCARFDataset


def fix_seed(seed):
    """Fix random seeds for reproducibility."""
    py_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_and_preprocess(csv_path, fcs_path, cofactor=None):
    """Load data and apply preprocessing (shared with scarf.py)."""
    if cofactor is None:
        cofactor = COFACTOR

    df_sraw = pd.read_csv(csv_path)
    _, df_fcs = fcsparser.parse(fcs_path, reformat_meta=True)
    assert len(df_sraw) == len(df_fcs), "Event counts do not match!"

    # Feature selection
    wl_features = [c for c in df_sraw.columns if c.startswith('Area_') and c.endswith('nm')]
    X_spectral = df_sraw[wl_features].values
    scatter_features = ['FSC - Area', 'SSC - Area']
    X_scatter = df_fcs[scatter_features].values
    X_combined = np.hstack((X_scatter, X_spectral))

    # ArcSinh + StandardScaler (ArcSinh is skipped)
    # X_arcsinh = np.arcsinh(X_combined / cofactor)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    return X_scaled


def train_one_config(X_scaled, num_frequencies, sigma, epochs=50,
                     batch_size=32, lr=0.001, temperature=1.0,
                     dim_hidden_encoder=16, num_hidden_encoder=4,
                     dim_hidden_head=16, num_hidden_head=2,
                     corruption_rate=0.6, dropout=0.0, embedding_dim=16):
    """Train SCARF with a specific (k, σ) and return final loss."""
    dummy_targets = np.zeros(len(X_scaled))
    dataset = SCARFDataset(X_scaled, dummy_targets)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    if len(dataloader) == 0:
        return float('inf')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SCARF(
        input_dim=dataset.shape[1],
        features_low=dataset.features_low,
        features_high=dataset.features_high,
        dim_hidden_encoder=dim_hidden_encoder,
        num_hidden_encoder=num_hidden_encoder,
        dim_hidden_head=dim_hidden_head,
        num_hidden_head=num_hidden_head,
        corruption_rate=corruption_rate,
        dropout=dropout,
        num_frequencies=num_frequencies,
        sigma=sigma,
        embedding_dim=embedding_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = NTXent(temperature=temperature)

    # Train
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x in dataloader:
            x = x.to(device)
            emb_anchor, emb_positive = model(x)
            loss = criterion(emb_anchor, emb_positive)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

    # Return average loss of last epoch
    final_loss = epoch_loss / len(dataloader)
    return final_loss


def run_search(X_full, n_iterations=100, subset_ratio=0.01, search_epochs=50,
               seed=42):
    """
    Run Bayesian optimization over (k, σ) hyperparameters using Optuna (TPE).

    Search space:
        k (num_frequencies): integer, sampled log-uniformly from [1, 128]
        σ (sigma):           float,   sampled log-uniformly from [0.001, 100]
    """
    fix_seed(seed)

    # --- 1. Extract subset ---
    n_total = len(X_full)
    n_subset = max(int(n_total * subset_ratio), 64)  # At least 64 events
    indices = np.random.choice(n_total, size=n_subset, replace=False)
    X_subset = X_full[indices]

    print(f"=== Hyperparameter Search (Optuna TPE) ===")
    print(f"Total events: {n_total}")
    print(f"Subset ratio: {subset_ratio:.1%} → {n_subset} events")
    print(f"Search epochs per trial: {search_epochs}")
    print(f"Number of trials: {n_iterations}")
    print(f"Search space: k ∈ [1, 128], σ ∈ [0.001, 100], lr ∈ [1e-4, 1e-2], corruption_rate ∈ [0.1, 0.8], dropout ∈ [0.0, 0.5]")
    print(f"{'='*60}")

    def objective(trial):
        k = trial.suggest_int('k', 1, 128, log=True)
        sigma = trial.suggest_float('sigma', 0.001, 100.0, log=True)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        corruption_rate = trial.suggest_float('corruption_rate', 0.1, 0.8)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        embedding_dim = trial.suggest_categorical('embedding_dim', [8, 16, 24, 32, 64])

        try:
            final_loss = train_one_config(
                X_subset,
                num_frequencies=k,
                sigma=sigma,
                lr=lr,
                corruption_rate=corruption_rate,
                dropout=dropout,
                embedding_dim=embedding_dim,
                epochs=search_epochs,
                batch_size=32,
            )
        except Exception as e:
            print(f"  Trial {trial.number}: k={k}, σ={sigma:.4f} → ERROR: {e}")
            raise optuna.TrialPruned()

        return final_loss

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
    cols_to_show = ['number', 'params_k', 'params_sigma', 'params_lr', 'params_corruption_rate', 'params_dropout', 'params_embedding_dim', 'value', 'duration']
    print(df_results[cols_to_show].head(10).to_string(index=False))

    best_trial = study.best_trial
    best_k = best_trial.params['k']
    best_sigma = best_trial.params['sigma']
    best_lr = best_trial.params['lr']
    best_cr = best_trial.params['corruption_rate']
    best_do = best_trial.params['dropout']
    best_ed = best_trial.params['embedding_dim']

    print(f"\n{'='*60}")
    print(f"★ Best configuration:")
    print(f"   k (num_frequencies) = {best_k}")
    print(f"   σ (sigma)           = {best_sigma}")
    print(f"   lr                  = {best_lr:.5f}")
    print(f"   corruption_rate     = {best_cr:.3f}")
    print(f"   dropout             = {best_do:.3f}")
    print(f"   embedding_dim       = {best_ed}")
    print(f"   Loss                = {best_trial.value}")
    print(f"{'='*60}")
    print(f"\nRun full training with best params:")
    print(f"  python -m learning.scarf --num-frequencies {best_k} --sigma {best_sigma} --lr {best_lr} --corruption-rate {best_cr} --dropout {best_do} --embedding-dim {best_ed} --epochs 200")

    # Save results CSV
    scarf_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(scarf_dir, "result")
    os.makedirs(result_dir, exist_ok=True)
    results_path = os.path.join(result_dir, "hp_search_results_optuna.csv")
    df_results.to_csv(results_path, index=False)
    print(f"\nFull results saved to: {results_path}")

    return df_results


def main():
    from learning.scarf import discover_samples

    print("Scanning for sample data...")
    samples = discover_samples()

    if not samples:
        print("No samples found. Please run the data pipeline first.")
        sys.exit(1)

    # Use the first sample for hyperparameter search
    sample = samples[0]
    print(f"Using sample: {sample['sample_label']}")
    print(f"  CSV: {os.path.basename(sample['csv'])}")
    print(f"  FCS: {os.path.basename(sample['fcs'])}")

    X_scaled = load_and_preprocess(sample['csv'], sample['fcs'])

    run_search(
        X_full=X_scaled,
        n_iterations=100,
        subset_ratio=0.01,
        search_epochs=50,
        seed=42
    )


if __name__ == '__main__':
    main()
