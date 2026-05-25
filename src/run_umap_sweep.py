"""
UMAP n_neighbors パラメータスイープ

複数の n_neighbors 値で UMAP を実行し、結果をPDFにまとめる。

Usage:
    python -m src.run_umap_sweep --csv <sraw_csv> --fcs <fcs_file> --output <output_pdf>
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import umap
import fcsparser
from sklearn.preprocessing import StandardScaler

# プロジェクトルートを path に追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COFACTOR


def run_umap_sweep(sraw_csv, fcs_file, output_pdf, n_neighbors_list=None, cofactor=None):
    """
    複数の n_neighbors 値で UMAP を実行し、結果を PDF にまとめる。

    Parameters
    ----------
    sraw_csv : str
        変換済み CSV ファイルのパス
    fcs_file : str
        対応する .fcs ファイルのパス
    output_pdf : str
        出力 PDF ファイルのパス
    n_neighbors_list : list of int, optional
        テストする n_neighbors 値のリスト
    cofactor : float, optional
        ArcSinh 変換の cofactor
    """
    if n_neighbors_list is None:
        n_neighbors_list = [2, 5, 10, 15, 30, 40, 50, 60]
    if cofactor is None:
        cofactor = COFACTOR

    # --- 1. Load Data ---
    print(f"Loading spectral data from {os.path.basename(sraw_csv)}...")
    df_sraw = pd.read_csv(sraw_csv)

    print(f"Loading scatter data from {os.path.basename(fcs_file)}...")
    _, df_fcs = fcsparser.parse(fcs_file, reformat_meta=True)

    assert len(df_sraw) == len(df_fcs), "Event counts do not match between CSV and FCS!"

    # --- 2. Feature Selection ---
    wl_features = [c for c in df_sraw.columns if c.startswith('Area_') and c.endswith('nm') and '638.6nm' not in c]
    X_spectral = df_sraw[wl_features].values

    scatter_features = ['FSC - Area', 'SSC - Area']
    X_scatter = df_fcs[scatter_features].values

    print(f"Selected {len(wl_features)} spectral channels + {len(scatter_features)} scatter channels.")

    X_combined = np.hstack((X_scatter, X_spectral))

    # --- 3. Preprocessing ---
    print("Applying ArcSinh transformation...")
    X_arcsinh = np.arcsinh(X_combined / cofactor)

    print("Applying StandardScaler (Z-score normalization)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arcsinh)

    # --- 4. Run UMAP for each n_neighbors ---
    embeddings = {}
    for nn in n_neighbors_list:
        print(f"Running UMAP with n_neighbors={nn}...")
        reducer = umap.UMAP(
            n_neighbors=nn,
            min_dist=0.3,
            metric='euclidean',
            random_state=42
        )
        embeddings[nn] = reducer.fit_transform(X_scaled)
        print(f"  Done (n_neighbors={nn}).")

    # --- 5. Prepare color parameters ---
    color_params = []

    # FSC
    color_params.append({
        'name': 'FSC - Area',
        'values': df_fcs['FSC - Area'].values,
        'label': 'FSC - Area',
    })

    # SSC
    color_params.append({
        'name': 'SSC - Area',
        'values': df_fcs['SSC - Area'].values,
        'label': 'SSC - Area',
    })

    # 33 wavelength channels (arcsinh-transformed for coloring)
    for wl in wl_features:
        wl_name = wl.replace('Area_', '')
        color_params.append({
            'name': wl_name,
            'values': np.arcsinh(df_sraw[wl].values / cofactor),
            'label': f'{wl_name} (ArcSinh)',
        })

    # --- 6. Generate PDF ---
    n_total_params = len(color_params)
    n_cols = 6
    n_rows = 6  # 6x6 = 36 slots for 35 params (1 empty)

    print(f"Generating PDF with {len(n_neighbors_list)} pages (one per n_neighbors)...")

    os.makedirs(os.path.dirname(output_pdf) or '.', exist_ok=True)

    with PdfPages(output_pdf) as pdf:
        for page_idx, nn in enumerate(n_neighbors_list):
            print(f"  Page {page_idx + 1}/{len(n_neighbors_list)}: n_neighbors={nn}...")

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 24))
            axes = axes.flatten()

            emb = embeddings[nn]

            for i, cp in enumerate(color_params):
                vmin = np.percentile(cp['values'], 1)
                vmax = np.percentile(cp['values'], 99)

                sc = axes[i].scatter(
                    emb[:, 0], emb[:, 1],
                    c=cp['values'], cmap='jet',
                    vmin=vmin, vmax=vmax,
                    s=1, alpha=0.5, edgecolors='none',
                    rasterized=True
                )
                axes[i].set_title(cp['name'], fontsize=10, fontweight='bold')
                axes[i].set_xticks([])
                axes[i].set_yticks([])
                fig.colorbar(sc, ax=axes[i], shrink=0.7)

            # Hide unused axes
            for j in range(n_total_params, n_rows * n_cols):
                axes[j].set_visible(False)

            fig.suptitle(f'UMAP — n_neighbors={nn}',
                         fontsize=18, fontweight='bold', y=0.99)

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig, dpi=100)
            plt.close(fig)

    print(f"PDF saved to: {output_pdf}")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description='UMAP n_neighbors パラメータスイープ')
    parser.add_argument('--csv', type=str, required=True, help='変換済み CSV ファイルのパス')
    parser.add_argument('--fcs', type=str, required=True, help='対応する .fcs ファイルのパス')
    parser.add_argument('--output', type=str, default=None, help='出力 PDF ファイルのパス')
    parser.add_argument('--n-neighbors', type=int, nargs='+', default=None,
                        help='n_neighbors の値リスト (例: 5 10 20 30)')
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(args.csv)[0]
        args.output = base + '_umap_sweep.pdf'

    run_umap_sweep(args.csv, args.fcs, args.output, n_neighbors_list=args.n_neighbors)


if __name__ == '__main__':
    main()
