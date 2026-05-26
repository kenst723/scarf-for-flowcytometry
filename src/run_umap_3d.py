"""
UMAP 3次元次元圧縮解析

スペクトルデータ + FCS散布光データを統合して 3D UMAP 可視化を行う。

Usage:
    python -m src.run_umap_3d --csv <sraw_csv> --fcs <fcs_file> --output <output_image>
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import umap
import fcsparser
from sklearn.preprocessing import StandardScaler

# プロジェクトルートを path に追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COFACTOR


def run_umap_analysis_3d(sraw_csv, fcs_file, output_image, stain_name=None, cofactor=None):
    """
    3D UMAP 次元圧縮解析を実行し、プロットを保存する。

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

    print("Applying StandardScaler (Z-score normalization)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arcsinh)

    # --- 3. UMAP Dimensionality Reduction (3D) ---
    print("Running 3D UMAP on combined features (this may take 10-30 seconds)...")
    reducer = umap.UMAP(
        n_components=3,  # Set to 3D
        n_neighbors=20,
        min_dist=0.3,
        metric='euclidean',
        random_state=42
    )
    embedding = reducer.fit_transform(X_scaled)

    # --- 4. Plotting (3D interactive via Plotly) ---
    print("Generating interactive 3D plot (HTML)...")

    # Determine stain column and values
    stain_col = None
    if stain_name and stain_name.lower() != 'negative':
        for col in df_fcs.columns:
            if stain_name.lower() in col.lower() and 'area' in col.lower():
                stain_col = col
                break

    if stain_col is not None:
        stain_values = np.arcsinh(df_fcs[stain_col].values / cofactor)
        stain_label = f'{stain_col} (ArcSinh)'
        stain_title = f'Colored by {stain_col}'
        print(f"  Coloring by FCS channel: {stain_col}")
    else:
        # Fallback
        stain_values = np.arcsinh(df_sraw["Area_450.0nm"] / 500)
        stain_label = '450nm (ArcSinh)'
        stain_title = 'Colored by 450nm'
        if stain_name:
            print(f"  Stain '{stain_name}' not found in FCS columns, falling back to 450nm")

    plot_df = pd.DataFrame({
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
        'UMAP3': embedding[:, 2],
        'FSC_Area': df_fcs['FSC - Area'].values,
        'Stain_Intensity': stain_values
    })

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=('Colored by FSC (Cell Size)', stain_title)
    )

    # Plot 1: FSC
    fig.add_trace(
        go.Scatter3d(
            x=plot_df['UMAP1'], y=plot_df['UMAP2'], z=plot_df['UMAP3'],
            mode='markers',
            marker=dict(
                size=2,
                color=plot_df['FSC_Area'],
                colorscale='Jet',
                opacity=0.7,
                colorbar=dict(title="FSC Area", x=0.45)
            )
        ),
        row=1, col=1
    )

    # Plot 2: Stain
    fig.add_trace(
        go.Scatter3d(
            x=plot_df['UMAP1'], y=plot_df['UMAP2'], z=plot_df['UMAP3'],
            mode='markers',
            marker=dict(
                size=2,
                color=plot_df['Stain_Intensity'],
                colorscale='Jet',
                opacity=0.7,
                colorbar=dict(title=stain_label, x=1.0)
            )
        ),
        row=1, col=2
    )

    fig.update_layout(
        title='3D UMAP of Spectral + Scatter Data (Standardized)',
        width=1600,
        height=800,
        margin=dict(l=0, r=0, b=0, t=50)
    )

    os.makedirs(os.path.dirname(output_image) or '.', exist_ok=True)
    fig.write_html(output_image)
    print(f"Interactive 3D Plot saved to: {output_image}")


def main():
    parser = argparse.ArgumentParser(description='UMAP 3次元次元圧縮解析')
    parser.add_argument('--csv', type=str, required=True, help='変換済み CSV ファイルのパス')
    parser.add_argument('--fcs', type=str, required=True, help='対応する .fcs ファイルのパス')
    parser.add_argument('--output', type=str, default=None, help='出力ファイルのパス (.html)')
    parser.add_argument('--stain', type=str, default=None, help='染色名 (例: PI, Calcein)')
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(args.csv)[0]
        args.output = base + '_umap_3d.html'

    run_umap_analysis_3d(args.csv, args.fcs, args.output, stain_name=args.stain)


if __name__ == '__main__':
    main()
