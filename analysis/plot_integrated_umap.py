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

def plot_integrated(neg_csv_path, stain_csv_path, stain_name="Calcein", output_html="integrated_umap.html", output_png="integrated_umap.png"):
    print("Loading CSVs...")
    df_neg = pd.read_csv(neg_csv_path)
    df_stain = pd.read_csv(stain_csv_path)
    
    print(f"Negative events: {len(df_neg)}")
    print(f"Stained events: {len(df_stain)}")
    
    # Extract spectral columns
    wl_features = [
        c for c in df_neg.columns
        if c.startswith('Area_') and c.endswith('nm') and '638.6nm' not in c
    ]
    
    X_neg = df_neg[wl_features].values
    X_stain = df_stain[wl_features].values
    
    # Calculate Reference Spectra to subtract PI/Calcein
    print("Reconstructing pure Autofluorescence spectra for Stained samples...")
    S_AF = np.median(X_neg, axis=0)
    S_AF = S_AF / np.sum(S_AF)
    
    total_intensity = np.sum(X_stain, axis=1)
    bright_cells_total = X_stain[total_intensity >= np.percentile(total_intensity, 99)]
    S_bright_total = np.median(bright_cells_total, axis=0)
    ratios = S_bright_total / (S_AF + 1e-9)
    peak_idx = np.argmax(ratios)
    
    peak_values = X_stain[:, peak_idx]
    stained_cells = X_stain[peak_values >= np.percentile(peak_values, 98)]
    S_Stain = np.median(stained_cells, axis=0)
    S_Stain = np.maximum(S_Stain - (S_AF * np.min(S_Stain / (S_AF + 1e-9))), 0)
    S_Stain = S_Stain / np.sum(S_Stain)
    
    # Subtract Stain component from raw spectra using WLSM
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
    unmixed_stain_intensity = df_stain[f'Unmixed_{stain_name}'].values
    
    S = np.column_stack((S_AF, S_Stain))
    C_stain = perform_wlsm(X_stain, S)
    raw_calcein_coef = C_stain[:, 1]
    
    # DO NOT use np.maximum(..., 0). Keep natural noise distribution around 0.
    X_to_umap = X_stain - raw_calcein_coef[:, None] * S_Stain[None, :]
    
    # Preprocessing for AF-only UMAP
    print("Preprocessing and running AF-only UMAP...")
    X_to_umap_arcsinh = np.arcsinh(X_to_umap / COFACTOR)
    scaler = StandardScaler()
    X_to_umap_scaled = scaler.fit_transform(X_to_umap_arcsinh)
    
    reducer_af = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3, random_state=42)
    umap_coords = reducer_af.fit_transform(X_to_umap_scaled)
    
    # Preprocessing for Raw Spectra UMAP
    print("Preprocessing and running Raw Spectra UMAP...")
    X_raw_arcsinh = np.arcsinh(X_stain / COFACTOR)
    scaler_raw = StandardScaler()
    X_raw_scaled = scaler_raw.fit_transform(X_raw_arcsinh)
    
    reducer_raw = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3, random_state=42)
    umap_coords_raw = reducer_raw.fit_transform(X_raw_scaled)
    
    # Preprocessing for Negative Sample UMAP
    print("Preprocessing and running Negative Sample UMAP...")
    X_neg_arcsinh = np.arcsinh(X_neg / COFACTOR)
    scaler_neg = StandardScaler()
    X_neg_scaled = scaler_neg.fit_transform(X_neg_arcsinh)
    
    reducer_neg = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3, random_state=42)
    umap_coords_neg = reducer_neg.fit_transform(X_neg_scaled)
    
    # Plotting
    print("Generating plots...")
    stain_intensity = np.arcsinh(unmixed_stain_intensity / COFACTOR)
    stain_label = f'Unmixed {stain_name} (ArcSinh)'
    vmin, vmax = np.percentile(stain_intensity, [1, 99])
    
    # Plotly
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        'Negative Sample UMAP',
        'Stained Sample - Autofluorescence UMAP',
        f'Stained Sample - AF UMAP (Colored by {stain_name})',
        f'Stained Sample - Raw Spectra UMAP (Colored by {stain_name})'
    ))
    
    fig.add_trace(go.Scatter(x=umap_coords_neg[:, 0], y=umap_coords_neg[:, 1], mode='markers', marker=dict(size=3, color='#808080', opacity=0.6), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=umap_coords[:, 0], y=umap_coords[:, 1], mode='markers', marker=dict(size=3, color='#d3d3d3', opacity=0.6), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=umap_coords[:, 0], y=umap_coords[:, 1], mode='markers', marker=dict(size=3, color=stain_intensity, cmin=vmin, cmax=vmax, colorscale='bluered', opacity=0.7), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=umap_coords_raw[:, 0], y=umap_coords_raw[:, 1], mode='markers', marker=dict(size=3, color=stain_intensity, cmin=vmin, cmax=vmax, colorscale='bluered', opacity=0.7, colorbar=dict(title=stain_label, x=1.0)), showlegend=False), row=2, col=2)
    
    fig.update_layout(title=f'Integrated Data (4900 events) - 4-way UMAP Comparison', width=1200, height=1000)
    fig.write_html(output_html)
    
    # Matplotlib
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig_mpl, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=200)
    
    # 1. Raw Spectra UMAP
    sc_raw = axes[0, 0].scatter(umap_coords_raw[:, 0], umap_coords_raw[:, 1], c=stain_intensity, vmin=vmin, vmax=vmax, cmap='coolwarm', s=2, alpha=0.5)
    axes[0, 0].set_title(f'1. Raw Data UMAP (4900 Stained Cells)\n(Colored by {stain_name})', fontsize=12, fontweight='bold')
    fig_mpl.colorbar(sc_raw, ax=axes[0, 0], label=stain_label)
    
    # 2. Control (Negative) UMAP
    axes[0, 1].scatter(umap_coords_neg[:, 0], umap_coords_neg[:, 1], c='#808080', s=2, alpha=0.5)
    axes[0, 1].set_title(f'2. Control (Negative) UMAP\n(4900 Negative Cells)', fontsize=12, fontweight='bold')
    
    # 3. Unmixed AF UMAP (Uncolored)
    axes[1, 0].scatter(umap_coords[:, 0], umap_coords[:, 1], c='#d3d3d3', s=2, alpha=0.5)
    axes[1, 0].set_title(f'3. Unmixed Autofluorescence UMAP\n(4900 Stained Cells, Uncolored)', fontsize=12, fontweight='bold')
    
    # 4. Unmixed AF UMAP (Colored)
    sc_af = axes[1, 1].scatter(umap_coords[:, 0], umap_coords[:, 1], c=stain_intensity, vmin=vmin, vmax=vmax, cmap='coolwarm', s=2, alpha=0.5)
    axes[1, 1].set_title(f'4. Unmixed Autofluorescence UMAP\n(Colored by {stain_name})', fontsize=12, fontweight='bold')
    fig_mpl.colorbar(sc_af, ax=axes[1, 1], label=stain_label)
    
    plt.tight_layout()
    plt.savefig(output_png, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_html} and {output_png}")

if __name__ == '__main__':
    neg_csv = r"analysis\results\2026-05-27\negative_B01\B01 Well - B01_20260527_110441.csv"
    calcein_csv = r"analysis\results\2026-05-27\Calcein_A01\A01 Well - A01_20260527_110439.csv"
    out_html = r"analysis\results\2026-05-27\integrated_umap_calcein.html"
    out_png = r"analysis\results\2026-05-27\integrated_umap_calcein.png"
    
    plot_integrated(neg_csv, calcein_csv, stain_name="Calcein", output_html=out_html, output_png=out_png)
