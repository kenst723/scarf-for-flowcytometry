"""
SCARF Embeddings UMAP Projection (Autofluorescence-based)

SCARF特徴量（Negative）で UMAP 座標系を学習し、
マーカー染色サンプルの SCARF 特徴量をその空間に投影して、蛍光強度で色付けした 2D プロットを生成する。

Usage:
    python -m learning.umap_embeddings_autofluor \
        --neg-emb "analysis/results/2026-05-21/Negative_A02/A02_scarf_embeddings.csv" \
        --stain-emb "analysis/results/2026-05-21/PI_A01/A01_scarf_embeddings.csv" \
        --fcs "data/Experiment 2026!05!21 15!59/24 Tube Rack (5mL) - 1/PI/A01 Well - A01.fcs" \
        --stain PI \
        --output "analysis/results/2026-05-21/scarf_umap_PI.html"
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

def run_scarf_umap_autofluor(neg_emb_path, stain_emb_path, fcs_path, output_path, stain_name="PI", cofactor=None, seed=42):
    if cofactor is None:
        cofactor = COFACTOR

    print(f"Loading Negative SCARF embeddings from {os.path.basename(neg_emb_path)}...")
    df_neg_emb = pd.read_csv(neg_emb_path)
    X_neg = df_neg_emb.values

    print(f"Loading Stained SCARF embeddings from {os.path.basename(stain_emb_path)}...")
    df_stain_emb = pd.read_csv(stain_emb_path)
    X_stain = df_stain_emb.values

    print(f"Loading FCS data for coloring from {os.path.basename(fcs_path)}...")
    _, df_fcs = fcsparser.parse(fcs_path, reformat_meta=True)
    assert len(X_stain) == len(df_fcs), "Event counts do not match between Stained embeddings and FCS!"

    print(f"Running 2D UMAP (fit on Negative, transform Stained)...")
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
    stain_col = None
    for col in df_fcs.columns:
        if stain_name.lower() in col.lower() and 'area' in col.lower():
            stain_col = col
            break

    if stain_col is not None:
        stain_intensity = np.arcsinh(df_fcs[stain_col].values / cofactor)
        stain_label = f'{stain_col} (ArcSinh)'
    else:
        stain_intensity = np.zeros(len(df_fcs))
        stain_label = 'Intensity Not Found'
        print(f"  Warning: '{stain_name}' channel not found in FCS.")

    print("Generating interactive 2D plot...")
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'SCARF: Negative + {stain_name} (Sample Type)',
            f'SCARF: {stain_name} colored by {stain_label}'
        )
    )

    # Left panel: Sample Type
    fig.add_trace(
        go.Scatter(
            x=umap_neg[:, 0], y=umap_neg[:, 1],
            mode='markers', name='Negative',
            marker=dict(size=3, color='gray', opacity=0.3),
            showlegend=True
        ), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=umap_stain[:, 0], y=umap_stain[:, 1],
            mode='markers', name=stain_name,
            marker=dict(size=3, color='red', opacity=0.5),
            showlegend=True
        ), row=1, col=1
    )

    # Right panel: Stain Intensity
    fig.add_trace(
        go.Scatter(
            x=umap_stain[:, 0], y=umap_stain[:, 1],
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


def main():
    parser = argparse.ArgumentParser(description='SCARF UMAP Projection')
    parser.add_argument('--neg-emb', type=str, required=True, help='Negative SCARF embeddings CSV')
    parser.add_argument('--stain-emb', type=str, required=True, help='Stained SCARF embeddings CSV')
    parser.add_argument('--fcs', type=str, required=True, help='Stained FCS file for coloring')
    parser.add_argument('--stain', type=str, default='PI', help='Stain name')
    parser.add_argument('--output', type=str, required=True, help='Output HTML path')
    args = parser.parse_args()

    run_scarf_umap_autofluor(
        neg_emb_path=args.neg_emb,
        stain_emb_path=args.stain_emb,
        fcs_path=args.fcs,
        output_path=args.output,
        stain_name=args.stain
    )

if __name__ == '__main__':
    main()
