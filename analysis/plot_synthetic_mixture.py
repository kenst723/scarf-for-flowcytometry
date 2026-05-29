import os
import sys
import pandas as pd
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from config import COFACTOR
from src.unmix_spectral import PoissonUnmixer

def plot_synthetic_mixture(neg_csv_path, stain_csv_path, output_png="synthetic_mixture_umap.png"):
    print("Loading CSVs...")
    df_neg = pd.read_csv(neg_csv_path)
    df_stain = pd.read_csv(stain_csv_path)
    
    wl_features = [c for c in df_neg.columns if c.startswith('Area_') and c.endswith('nm') and '638.6nm' not in c]
    
    X_neg_full = df_neg[wl_features].values
    X_stain_full = df_stain[wl_features].values
    
    print("Creating Synthetic Mixture (2500 Neg + 2000 Calcein)...")
    np.random.seed(42)
    idx_neg = np.random.choice(len(X_neg_full), 2500, replace=False)
    idx_stain = np.random.choice(len(X_stain_full), 2000, replace=False)
    
    X_neg_mix = X_neg_full[idx_neg]
    X_stain_mix = X_stain_full[idx_stain]
    X_mix = np.vstack([X_neg_mix, X_stain_mix])
    labels = np.array([0]*2500 + [1]*2000)
    
    print("Fitting PoissonUnmixer...")
    unmixer = PoissonUnmixer().fit(X_neg_full, X_stain_full)
    
    print("Performing IRLS Poisson Unmixing...")
    C_mix = unmixer.get_raw_coefficients(X_mix)
    raw_calcein_coef = C_mix[:, 1]
    
    X_mix_unmixed_af = unmixer.remove_stain_component(X_mix)
    
    print("Computing UMAPs...")
    scaler_raw = StandardScaler()
    umap_raw = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3, random_state=42).fit_transform(scaler_raw.fit_transform(np.arcsinh(X_mix / COFACTOR)))
    
    X_neg_mix_only = X_mix[labels == 0]
    scaler_neg = StandardScaler()
    umap_neg = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3, random_state=42).fit_transform(scaler_neg.fit_transform(np.arcsinh(X_neg_mix_only / COFACTOR)))
    
    scaler_af = StandardScaler()
    umap_af = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3, random_state=42).fit_transform(scaler_af.fit_transform(np.arcsinh(X_mix_unmixed_af / COFACTOR)))
    
    print("Generating plot...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=200)
    colors_label = np.array(['gray', 'green'])
    labels_text = ['Negative Cells', 'Calcein Cells']
    
    for i in [0, 1]:
        mask = (labels == i)
        axes[0, 0].scatter(umap_raw[mask, 0], umap_raw[mask, 1], c=colors_label[i], label=labels_text[i], s=5, alpha=0.6)
    axes[0, 0].set_title('1. Raw Synthetic Data UMAP', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    
    axes[0, 1].scatter(umap_neg[:, 0], umap_neg[:, 1], c='gray', s=5, alpha=0.6)
    axes[0, 1].set_title('2. Control (Negative) UMAP\n(Computed only on Negative cells)', fontsize=14, fontweight='bold')
    
    for i in [0, 1]:
        mask = (labels == i)
        axes[1, 0].scatter(umap_af[mask, 0], umap_af[mask, 1], c=colors_label[i], label=labels_text[i], s=5, alpha=0.6)
    axes[1, 0].set_title('3. Unmixed Autofluorescence UMAP\n(After Calcein Subtraction)', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    
    calcein_arcsinh = np.arcsinh(raw_calcein_coef / COFACTOR)
    sc = axes[1, 1].scatter(umap_af[:, 0], umap_af[:, 1], c=calcein_arcsinh, cmap='viridis', s=5, alpha=0.6)
    axes[1, 1].set_title('4. Unmixed AF UMAP\n(Colored by Unmixed Calcein Coefficient)', fontsize=14, fontweight='bold')
    fig.colorbar(sc, ax=axes[1, 1], label='Calcein Coefficient (ArcSinh)')
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_png}")

if __name__ == '__main__':
    neg_csv = r"analysis\results\2026-05-27\negative_B01\B01 Well - B01_20260529_134834.csv"
    calcein_csv = r"analysis\results\2026-05-27\Calcein_A01\A01 Well - A01_20260529_134433.csv"
    out_png = r"analysis\results\2026-05-27\synthetic_mixture_umap_2x2.png"
    plot_synthetic_mixture(neg_csv, calcein_csv, output_png=out_png)
