import os
import sys
import pandas as pd
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import COFACTOR

def plot_synthetic_mixture(neg_csv_path, stain_csv_path, output_png="synthetic_mixture_umap.png"):
    print("Loading CSVs...")
    df_neg = pd.read_csv(neg_csv_path)
    df_stain = pd.read_csv(stain_csv_path)
    
    # Extract spectral columns
    wl_features = [
        c for c in df_neg.columns
        if c.startswith('Area_') and c.endswith('nm') and '638.6nm' not in c
    ]
    
    X_neg_full = df_neg[wl_features].values
    X_stain_full = df_stain[wl_features].values
    
    # Create synthetic mixture: 2900 Negative + 2000 Calcein
    print("Creating Synthetic Mixture (2900 Neg + 2000 Calcein)...")
    np.random.seed(42)
    idx_neg = np.random.choice(len(X_neg_full), 2900, replace=False)
    idx_stain = np.random.choice(len(X_stain_full), 2000, replace=False)
    
    X_neg_mix = X_neg_full[idx_neg]
    X_stain_mix = X_stain_full[idx_stain]
    
    X_mix = np.vstack([X_neg_mix, X_stain_mix])
    
    # Ground truth labels for coloring (0 = Negative, 1 = Calcein)
    labels = np.array([0]*2900 + [1]*2000)
    
    # Calculate Reference Spectra using the mixture
    print("Calculating References...")
    S_AF = np.median(X_neg_full, axis=0)
    S_AF = S_AF / np.sum(S_AF)
    
    total_intensity = np.sum(X_mix, axis=1)
    bright_cells_total = X_mix[total_intensity >= np.percentile(total_intensity, 95)]
    S_bright_total = np.median(bright_cells_total, axis=0)
    ratios = S_bright_total / (S_AF + 1e-9)
    peak_idx = np.argmax(ratios)
    
    peak_values = X_mix[:, peak_idx]
    stained_cells = X_mix[peak_values >= np.percentile(peak_values, 95)]
    S_Stain = np.median(stained_cells, axis=0)
    S_Stain = np.maximum(S_Stain - (S_AF * np.min(S_Stain / (S_AF + 1e-9))), 0)
    S_Stain = S_Stain / np.sum(S_Stain)
    
    # Perform WLSM Unmixing to extract AF
    def perform_wlsm(X, S, epsilon=1.0):
        N, M = X.shape
        num_fluor = S.shape[1]
        C = np.zeros((N, num_fluor))
        for i in range(N):
            w = 1.0 / np.maximum(X[i], epsilon)
            St_W = S.T * w
            try:
                inv = np.linalg.inv(St_W @ S)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(St_W @ S)
            C[i] = inv @ St_W @ X[i]
        return C

    print("Performing WLSM Unmixing...")
    S = np.column_stack((S_AF, S_Stain))
    C_mix = perform_wlsm(X_mix, S)
    raw_calcein_coef = C_mix[:, 1]
    
    # Reconstruct AF-only (subtracting Calcein component)
    X_mix_unmixed_af = X_mix - raw_calcein_coef[:, None] * S_Stain[None, :]
    
    print("Computing UMAP 1: Raw Mixture...")
    scaler_raw = StandardScaler()
    X_mix_scaled = scaler_raw.fit_transform(np.arcsinh(X_mix / COFACTOR))
    reducer_raw = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3, random_state=42)
    umap_raw = reducer_raw.fit_transform(X_mix_scaled)
    
    print("Computing UMAP 2: Negative Control...")
    X_neg_mix = X_mix[labels == 0]
    scaler_neg = StandardScaler()
    X_neg_scaled = scaler_neg.fit_transform(np.arcsinh(X_neg_mix / COFACTOR))
    reducer_neg = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3, random_state=42)
    umap_neg = reducer_neg.fit_transform(X_neg_scaled)
    
    print("Computing UMAP 3 & 4: Unmixed AF...")
    scaler_af = StandardScaler()
    X_af_scaled = scaler_af.fit_transform(np.arcsinh(X_mix_unmixed_af / COFACTOR))
    reducer_af = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3, random_state=42)
    umap_af = reducer_af.fit_transform(X_af_scaled)
    
    # Plotting
    print("Generating 2x2 plot...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=200)
    
    colors_label = np.array(['gray', 'green'])
    labels_text = ['Negative Cells', 'Calcein Cells']
    
    # Plot 1: Raw Synthetic Mixture
    for i in [0, 1]:
        mask = (labels == i)
        axes[0, 0].scatter(umap_raw[mask, 0], umap_raw[mask, 1], c=colors_label[i], label=labels_text[i], s=5, alpha=0.6)
    axes[0, 0].set_title('1. Raw Synthetic Data UMAP', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    
    # Plot 2: Control (Negative) UMAP
    axes[0, 1].scatter(umap_neg[:, 0], umap_neg[:, 1], c='gray', s=5, alpha=0.6)
    axes[0, 1].set_title('2. Control (Negative) UMAP\n(Computed only on Negative cells)', fontsize=14, fontweight='bold')
    
    # Plot 3: Unmixed AF UMAP
    for i in [0, 1]:
        mask = (labels == i)
        axes[1, 0].scatter(umap_af[mask, 0], umap_af[mask, 1], c=colors_label[i], label=labels_text[i], s=5, alpha=0.6)
    axes[1, 0].set_title('3. Unmixed Autofluorescence UMAP\n(After Calcein Subtraction)', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    
    # Plot 4: Unmixed AF UMAP colored by Calcein
    # Apply arcsinh scaling to calcein coefficient for coloring
    calcein_arcsinh = np.arcsinh(raw_calcein_coef / COFACTOR)
    sc = axes[1, 1].scatter(umap_af[:, 0], umap_af[:, 1], c=calcein_arcsinh, cmap='viridis', s=5, alpha=0.6)
    axes[1, 1].set_title('4. Unmixed AF UMAP\n(Colored by Unmixed Calcein Coefficient)', fontsize=14, fontweight='bold')
    fig.colorbar(sc, ax=axes[1, 1], label='Calcein Coefficient (ArcSinh)')
    
    plt.tight_layout()
    output_path = r'analysis\results\2026-05-27\synthetic_mixture_umap_2x2.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

if __name__ == '__main__':
    neg_csv = r"analysis\results\2026-05-27\negative_B01\B01 Well - B01_20260527_110441.csv"
    calcein_csv = r"analysis\results\2026-05-27\Calcein_A01\A01 Well - A01_20260527_110439.csv"
    out_png = r"analysis\results\2026-05-27\synthetic_mixture_umap.png"
    
    plot_synthetic_mixture(neg_csv, calcein_csv, output_png=out_png)
