"""
SCARF Embeddings UMAP Projection (Autofluorescence-based)

SCARF特徴量（Negative）で UMAP 座標系を学習し、
マーカー染色サンプルの SCARF 特徴量をその空間に投影して、蛍光強度で色付けした 2D プロットを生成する。

Usage:
    python -m src.umap_embeddings_autofluor \
        --neg-emb "learning/results/2026-05-21/Negative_A02/Negative_A02_scarf_embeddings.csv" \
        --stain-emb "learning/results/2026-05-21/PI_A01/PI_A01_scarf_embeddings.csv" \
        --stain-csv "analysis/results/2026-05-21/PI_A01/A01 Well - A01_20260526_191729.csv" \
        --stain PI \
        --output "learning/results/2026-05-21/scarf_umap_PI.html"
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import umap
import fcsparser

# プロジェクトルートを path に追加
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from config import COFACTOR

def run_scarf_umap_autofluor(neg_emb_path, stain_emb_path, stain_csv_path, output_path, stain_name="PI", cofactor=None, seed=42, png_output_path=None):
    if cofactor is None:
        cofactor = COFACTOR

    print(f"Loading Negative SCARF embeddings from {os.path.basename(neg_emb_path)}...")
    df_neg_emb = pd.read_csv(neg_emb_path)
    X_neg = df_neg_emb.values

    print(f"Loading Stained SCARF embeddings from {os.path.basename(stain_emb_path)}...")
    df_stain_emb = pd.read_csv(stain_emb_path)
    X_stain = df_stain_emb.values

    print(f"Loading raw Stained CSV (for unmixed intensities) from {os.path.basename(stain_csv_path)}...")
    df_stain_csv = pd.read_csv(stain_csv_path)
    assert len(X_stain) == len(df_stain_csv), "Event counts do not match between Stained embeddings and raw CSV!"

    print("Running 2D UMAP (fit on Negative, transform Stained)...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.3,
        metric='euclidean',
        random_state=seed
    )

    print("  Fitting UMAP on Negative embeddings...")
    umap_neg = reducer.fit_transform(X_neg)

    print(f"  Transforming {stain_name} embeddings into Negative UMAP space...")
    umap_stain = reducer.transform(X_stain)

    # 蛍光強度の取得
    if 'Unmixed_AF' in df_stain_csv.columns and f'Unmixed_{stain_name}' in df_stain_csv.columns:
        af_intensity = np.arcsinh(df_stain_csv['Unmixed_AF'].values / cofactor)
        stain_intensity = np.arcsinh(df_stain_csv[f'Unmixed_{stain_name}'].values / cofactor)
        af_label = 'Unmixed AF (ArcSinh)'
        stain_label = f'Unmixed {stain_name} (ArcSinh)'
    else:
        print("Warning: Unmixed columns not found in CSV. Using zeros.")
        af_intensity = np.zeros(len(df_stain_csv))
        stain_intensity = np.zeros(len(df_stain_csv))
        af_label = 'Mean AF (Fallback)'
        stain_label = f'{stain_name} (Fallback)'

    print("Generating interactive 2D plot...")
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'Negative Control (Autofluorescence only)',
            f'Stained ({stain_name}) colored by {stain_label}'
        )
    )

    # Left panel: Negative Control (Autofluorescence only) - Uniform gray
    fig.add_trace(
        go.Scatter(
            x=umap_neg[:, 0], y=umap_neg[:, 1],
            mode='markers', name='Negative Control',
            marker=dict(
                size=3, color='#d3d3d3',
                opacity=0.6
            ),
            showlegend=False
        ), row=1, col=1
    )

    # Right panel: Stain Intensity
    fig.add_trace(
        go.Scatter(
            x=umap_neg[:, 0], y=umap_neg[:, 1],
            mode='markers', name=f'{stain_name} Intensity',
            marker=dict(
                size=3, color=stain_intensity,
                colorscale='Jet', opacity=0.7,
                colorbar=dict(title=stain_label, x=1.0)
            ),
            showlegend=False
        ), row=1, col=2
    )

    fig.update_layout(
        title=f'SCARF Embeddings + Autofluorescence UMAP Projection',
        width=1400, height=700,
        margin=dict(l=40, r=40, b=40, t=60),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    fig.update_xaxes(title_text='UMAP 1')
    fig.update_yaxes(title_text='UMAP 2')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.write_html(output_path)
    print(f"Interactive 2D plot saved to: {output_path}")

    if png_output_path:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig_mpl, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=200)
        
        # Left: Negative Control
        axes[0].scatter(
            umap_neg[:, 0], umap_neg[:, 1], 
            c='#d3d3d3', s=2, alpha=0.5
        )
        axes[0].set_title('Negative Control\n(Autofluorescence only, uncolored)', fontsize=12, fontweight='bold', pad=10)
        axes[0].set_xlabel('UMAP 1', fontsize=10)
        axes[0].set_ylabel('UMAP 2', fontsize=10)
        
        # Right: Stain (PI)
        sc2 = axes[1].scatter(
            umap_neg[:, 0], umap_neg[:, 1], 
            c=stain_intensity, cmap='jet', s=2, alpha=0.5
        )
        axes[1].set_title(f'Stained ({stain_name}) - Target Dye Intensity\n({stain_label})', fontsize=12, fontweight='bold', pad=10)
        axes[1].set_xlabel('UMAP 1', fontsize=10)
        axes[1].set_ylabel('UMAP 2', fontsize=10)
        fig_mpl.colorbar(sc2, ax=axes[1], label=stain_label)
        
        fig_mpl.suptitle(f'SCARF Embeddings + Autofluorescence UMAP Projection', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        os.makedirs(os.path.dirname(png_output_path) or '.', exist_ok=True)
        plt.savefig(png_output_path, bbox_inches='tight')
        plt.close()
        print(f"Static 2D plot saved to: {png_output_path}")


def main():
    parser = argparse.ArgumentParser(description='SCARF UMAP Projection')
    parser.add_argument('--neg-emb', type=str, required=True, help='Negative SCARF embeddings CSV')
    parser.add_argument('--stain-emb', type=str, required=True, help='Stained SCARF embeddings CSV')
    parser.add_argument('--stain-csv', type=str, required=True, help='Raw Stained CSV containing Unmixed columns')
    parser.add_argument('--stain', type=str, default='PI', help='Stain name')
    parser.add_argument('--output', type=str, required=True, help='Output HTML path')
    parser.add_argument('--png-output', type=str, default=None, help='Output PNG path')
    args = parser.parse_args()

    run_scarf_umap_autofluor(
        neg_emb_path=args.neg_emb,
        stain_emb_path=args.stain_emb,
        stain_csv_path=args.stain_csv,
        output_path=args.output,
        stain_name=args.stain,
        png_output_path=args.png_output
    )

if __name__ == '__main__':
    main()
