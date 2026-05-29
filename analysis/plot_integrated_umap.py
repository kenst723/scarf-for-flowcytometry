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
from src.unmix_spectral import PoissonUnmixer

def plot_integrated(neg_csv_path, stain_csv_path, stain_name="Calcein", output_html="integrated_umap.html", output_png="integrated_umap.png"):
    print("Loading CSVs...")
    df_neg = pd.read_csv(neg_csv_path)
    df_stain = pd.read_csv(stain_csv_path)
    
    wl_features = [c for c in df_neg.columns if c.startswith('Area_') and c.endswith('nm') and '638.6nm' not in c]
    X_neg = df_neg[wl_features].values
    X_stain = df_stain[wl_features].values
    
    print("Performing IRLS Poisson Unmixing...")
    unmixer = PoissonUnmixer().fit(X_neg, X_stain)
    _, unmixed_stain_intensity = unmixer.transform(X_stain)
    X_to_umap = unmixer.remove_stain_component(X_stain)
    
    print("Preprocessing and running AF-only UMAP...")
    reducer_af = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3, random_state=42)
    umap_coords = reducer_af.fit_transform(StandardScaler().fit_transform(np.arcsinh(X_to_umap / COFACTOR)))
    
    print("Preprocessing and running Raw Spectra UMAP...")
    reducer_raw = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3, random_state=42)
    umap_coords_raw = reducer_raw.fit_transform(StandardScaler().fit_transform(np.arcsinh(X_stain / COFACTOR)))
    
    print("Preprocessing and running Negative Sample UMAP...")
    reducer_neg = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3, random_state=42)
    umap_coords_neg = reducer_neg.fit_transform(StandardScaler().fit_transform(np.arcsinh(X_neg / COFACTOR)))
    
    print("Generating plots...")
    stain_intensity = np.arcsinh(unmixed_stain_intensity / COFACTOR)
    vmin, vmax = np.percentile(stain_intensity, [1, 99])
    stain_label = f'Unmixed {stain_name} (ArcSinh)'
    
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
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig_mpl, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=200)
    
    sc_raw = axes[0, 0].scatter(umap_coords_raw[:, 0], umap_coords_raw[:, 1], c=stain_intensity, vmin=vmin, vmax=vmax, cmap='coolwarm', s=2, alpha=0.5)
    axes[0, 0].set_title(f'1. Raw Data UMAP\n(Colored by {stain_name})', fontsize=12, fontweight='bold')
    fig_mpl.colorbar(sc_raw, ax=axes[0, 0], label=stain_label)
    
    axes[0, 1].scatter(umap_coords_neg[:, 0], umap_coords_neg[:, 1], c='#808080', s=2, alpha=0.5)
    axes[0, 1].set_title('2. Control (Negative) UMAP', fontsize=12, fontweight='bold')
    
    axes[1, 0].scatter(umap_coords[:, 0], umap_coords[:, 1], c='#d3d3d3', s=2, alpha=0.5)
    axes[1, 0].set_title('3. Unmixed Autofluorescence UMAP', fontsize=12, fontweight='bold')
    
    sc_af = axes[1, 1].scatter(umap_coords[:, 0], umap_coords[:, 1], c=stain_intensity, vmin=vmin, vmax=vmax, cmap='coolwarm', s=2, alpha=0.5)
    axes[1, 1].set_title(f'4. Unmixed Autofluorescence UMAP\n(Colored by {stain_name})', fontsize=12, fontweight='bold')
    fig_mpl.colorbar(sc_af, ax=axes[1, 1], label=stain_label)
    
    plt.tight_layout()
    plt.savefig(output_png, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_html} and {output_png}")

if __name__ == '__main__':
    neg_csv = r"analysis\results\2026-05-27\negative_B01\B01 Well - B01_20260529_134834.csv"
    calcein_csv = r"analysis\results\2026-05-27\Calcein_A01\A01 Well - A01_20260529_134433.csv"
    out_html = r"analysis\results\2026-05-27\integrated_umap_calcein.html"
    out_png = r"analysis\results\2026-05-27\integrated_umap_calcein.png"
    plot_integrated(neg_csv, calcein_csv, stain_name="Calcein", output_html=out_html, output_png=out_png)
