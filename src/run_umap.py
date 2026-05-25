"""
UMAP 次元圧縮解析

スペクトルデータ + FCS散布光データを統合して UMAP 可視化を行う。

Usage:
    python -m src.run_umap --csv <sraw_csv> --fcs <fcs_file> --output <output_image>
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import umap
import fcsparser
from sklearn.preprocessing import StandardScaler

# プロジェクトルートを path に追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COFACTOR


def run_umap_analysis(sraw_csv, fcs_file, output_image, stain_name=None, cofactor=None):
    """
    UMAP 次元圧縮解析を実行し、プロットを保存する。

    Parameters
    ----------
    sraw_csv : str
        変換済み CSV ファイルのパス
    fcs_file : str
        対応する .fcs ファイルのパス
    output_image : str
        出力画像のパス (.png)
    stain_name : str, optional
        染色名 (例: "PI", "Calcein")。FCS から対応するチャネルを探して色付けに使用。
    cofactor : float, optional
        ArcSinh 変換の cofactor。None の場合は config.COFACTOR を使用。
    """
    if cofactor is None:
        cofactor = COFACTOR

    print(f"Loading spectral data from {os.path.basename(sraw_csv)}...")
    df_sraw = pd.read_csv(sraw_csv)

    print(f"Loading scatter data from {os.path.basename(fcs_file)}...")
    _, df_fcs = fcsparser.parse(fcs_file, reformat_meta=True)

    # Verify 1:1 mapping
    assert len(df_sraw) == len(df_fcs), "Event counts do not match between CSV and FCS!"

    # --- 1. Feature Selection ---
    # 33 Wavelength channels (excluding 638.6nm laser noise)
    wl_features = [c for c in df_sraw.columns if c.startswith('Area_') and c.endswith('nm') and '638.6nm' not in c]
    X_spectral = df_sraw[wl_features].values

    # FSC and SSC (Area)
    scatter_features = ['FSC - Area', 'SSC - Area']
    X_scatter = df_fcs[scatter_features].values

    print(f"Selected {len(wl_features)} spectral channels + {len(scatter_features)} scatter channels.")

    # Combine them into one feature matrix
    X_combined = np.hstack((X_scatter, X_spectral))

    # --- 2. Preprocessing ---
    print("Applying ArcSinh transformation...")
    X_arcsinh = np.arcsinh(X_combined / cofactor)

    # Important: Because FSC/SSC values have a vastly different dynamic range
    # than fluorescence, we MUST standardize (Z-score) all columns so that
    # FSC/SSC don't completely dominate the UMAP distances.
    print("Applying StandardScaler (Z-score normalization)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arcsinh)

    # --- 3. UMAP Dimensionality Reduction ---
    print("Running UMAP on combined features (this may take 10-20 seconds)...")
    reducer = umap.UMAP(
        n_neighbors=20,
        min_dist=0.3,
        metric='euclidean',
        random_state=42
    )
    embedding = reducer.fit_transform(X_scaled)

    # --- 4. Plotting ---
    print("Generating plot...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Colored by FSC to see where the large cells are
    sc1 = axes[0].scatter(
        embedding[:, 0], embedding[:, 1],
        c=df_fcs['FSC - Area'], cmap='jet',
        s=2, alpha=0.7, edgecolors='none'
    )
    axes[0].set_title('UMAP (Colored by FSC - Cell Size)', fontsize=14, fontweight='bold')
    axes[0].set_xticks([]); axes[0].set_yticks([])
    fig.colorbar(sc1, ax=axes[0], label='FSC - Area')

    # Plot 2: Colored by stain channel (from FCS) or fallback to 450nm
    stain_col = None
    if stain_name and stain_name.lower() != 'negative':
        # FCS カラム名から染色チャネルを検索 (例: "PI" → "PI - Area", "Calcein" → "Calcein AM - Area")
        for col in df_fcs.columns:
            if stain_name.lower() in col.lower() and 'area' in col.lower():
                stain_col = col
                break

    if stain_col is not None:
        stain_values = np.arcsinh(df_fcs[stain_col].values / cofactor)
        stain_label = f'{stain_col} (ArcSinh)'
        stain_title = f'UMAP (Colored by {stain_col})'
        print(f"  Coloring by FCS channel: {stain_col}")
    else:
        # フォールバック: 450nm スペクトルチャネル
        stain_values = np.arcsinh(df_sraw["Area_450.0nm"] / 500)
        stain_label = '450nm (ArcSinh)'
        stain_title = 'UMAP (Colored by 450nm)'
        if stain_name:
            print(f"  Stain '{stain_name}' not found in FCS columns, falling back to 450nm")

    sc2 = axes[1].scatter(
        embedding[:, 0], embedding[:, 1],
        c=stain_values, cmap='jet',
        s=2, alpha=0.7, edgecolors='none'
    )
    axes[1].set_title(stain_title, fontsize=14, fontweight='bold')
    axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.colorbar(sc2, ax=axes[1], label=stain_label)

    fig.suptitle('UMAP of Spectral + Scatter Data (Standardized)', fontsize=16, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_image) or '.', exist_ok=True)
    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {output_image}")


def main():
    parser = argparse.ArgumentParser(description='UMAP 次元圧縮解析')
    parser.add_argument('--csv', type=str, required=True, help='変換済み CSV ファイルのパス')
    parser.add_argument('--fcs', type=str, required=True, help='対応する .fcs ファイルのパス')
    parser.add_argument('--output', type=str, default=None, help='出力画像のパス (.png)')
    parser.add_argument('--stain', type=str, default=None, help='染色名 (例: PI, Calcein)')
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(args.csv)[0]
        args.output = base + '_umap.png'

    run_umap_analysis(args.csv, args.fcs, args.output, stain_name=args.stain)


if __name__ == '__main__':
    main()
