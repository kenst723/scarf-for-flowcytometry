"""
自家蛍光 UMAP + マーカー投影 (2D Plotly)

ネガティブコントロール（無染色）のスペクトルデータで UMAP 座標系を学習し、
マーカー染色サンプルをその空間に投影して蛍光強度で色付けした 2D プロットを生成する。

Usage:
    python -m src.run_umap_autofluor \
        --neg-dir "data/Experiment 2026!05!21 15!59/24 Tube Rack (5mL) - 1/Negative" \
        --stain-dir "data/Experiment 2026!05!21 15!59/24 Tube Rack (5mL) - 1/PI" \
        --stain PI \
        --output "analysis/results/2026-05-21/autofluor_umap.html"
"""

import os
import sys
import argparse
import glob

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import umap
import fcsparser
from sklearn.preprocessing import StandardScaler

# プロジェクトルートを path に追加
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import COFACTOR
from src.convert import convert_sraw_to_csv


def load_spectral_data(sraw_path, fcs_path, cofactor=None):
    """
    .sraw を CSV に変換し、スペクトルチャネル (Area_XXXnm) のみ抽出して返す。
    FCS ファイルからは scatter (FSC/SSC) や蛍光チャネルの値も取得する。

    Returns
    -------
    X_spectral : np.ndarray
        スペクトルチャネルの値 (num_events, num_spectral_channels)
    wl_features : list[str]
        スペクトルチャネルのカラム名リスト
    df_fcs : pd.DataFrame
        FCS データフレーム（色付け用チャネルの取得に使用）
    """
    if cofactor is None:
        cofactor = COFACTOR

    # .sraw → CSV 変換（一時的に result フォルダに出力）
    temp_dir = os.path.join(PROJECT_ROOT, "analysis", "results", "_temp_autofluor")
    os.makedirs(temp_dir, exist_ok=True)
    csv_path, df_sraw = convert_sraw_to_csv(sraw_path, output_dir=temp_dir)

    # FCS 読み込み
    _, df_fcs = fcsparser.parse(fcs_path, reformat_meta=True)
    assert len(df_sraw) == len(df_fcs), \
        f"Event counts do not match: sraw={len(df_sraw)}, fcs={len(df_fcs)}"

    # スペクトルチャネルのみ抽出 (638.6nm レーザーノイズを除外)
    wl_features = [
        c for c in df_sraw.columns
        if c.startswith('Area_') and c.endswith('nm') and '638.6nm' not in c
    ]
    X_spectral = df_sraw[wl_features].values

    # 一時ファイルの削除
    try:
        os.remove(csv_path)
    except OSError:
        pass

    return X_spectral, wl_features, df_fcs


def find_sraw_fcs_pairs(directory):
    """
    ディレクトリ内の .sraw ファイルと対応する .fcs ファイルのペアを返す。

    Returns
    -------
    pairs : list[tuple[str, str]]
        [(sraw_path, fcs_path), ...]
    """
    pairs = []
    sraw_files = sorted(glob.glob(os.path.join(directory, "*.sraw")))
    for sraw_path in sraw_files:
        base = os.path.splitext(sraw_path)[0]
        fcs_path = base + ".fcs"
        if os.path.isfile(fcs_path):
            pairs.append((sraw_path, fcs_path))
        else:
            print(f"  Warning: No matching .fcs for {os.path.basename(sraw_path)}, skipping.")
    return pairs


def run_umap_autofluor(neg_dir, stain_dir, output_path, stain_name="PI",
                       cofactor=None, seed=42):
    """
    自家蛍光 UMAP + マーカー投影パイプライン。

    Parameters
    ----------
    neg_dir : str
        ネガティブコントロール（無染色）のデータディレクトリ
    stain_dir : str
        マーカー染色サンプルのデータディレクトリ
    output_path : str
        出力 HTML ファイルのパス
    stain_name : str
        染色マーカー名 (例: "PI")
    cofactor : float, optional
        ArcSinh 変換の cofactor
    seed : int
        UMAP の乱数シード
    """
    if cofactor is None:
        cofactor = COFACTOR

    # =========================================================================
    # 1. Negative サンプルの読み込み
    # =========================================================================
    print("=" * 60)
    print("Step 1: Loading Negative (autofluorescence) samples...")
    print("=" * 60)

    neg_pairs = find_sraw_fcs_pairs(neg_dir)
    if not neg_pairs:
        print(f"Error: No .sraw/.fcs pairs found in {neg_dir}")
        sys.exit(1)

    neg_spectral_list = []
    neg_fcs_list = []
    wl_features = None

    for sraw_path, fcs_path in neg_pairs:
        print(f"  Loading {os.path.basename(sraw_path)}...")
        X_sp, wl_feat, df_fcs = load_spectral_data(sraw_path, fcs_path, cofactor)
        neg_spectral_list.append(X_sp)
        neg_fcs_list.append(df_fcs)
        if wl_features is None:
            wl_features = wl_feat

    X_neg = np.vstack(neg_spectral_list)
    df_neg_fcs = pd.concat(neg_fcs_list, ignore_index=True)
    print(f"  Total Negative events: {len(X_neg)} ({len(neg_pairs)} files)")
    print(f"  Spectral channels: {len(wl_features)}")

    # =========================================================================
    # 2. Stained サンプルの読み込み
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"Step 2: Loading {stain_name}-stained samples...")
    print("=" * 60)

    stain_pairs = find_sraw_fcs_pairs(stain_dir)
    if not stain_pairs:
        print(f"Error: No .sraw/.fcs pairs found in {stain_dir}")
        sys.exit(1)

    stain_spectral_list = []
    stain_fcs_list = []

    for sraw_path, fcs_path in stain_pairs:
        print(f"  Loading {os.path.basename(sraw_path)}...")
        X_sp, _, df_fcs = load_spectral_data(sraw_path, fcs_path, cofactor)
        stain_spectral_list.append(X_sp)
        stain_fcs_list.append(df_fcs)

    X_stain = np.vstack(stain_spectral_list)
    df_stain_fcs = pd.concat(stain_fcs_list, ignore_index=True)
    print(f"  Total {stain_name} events: {len(X_stain)} ({len(stain_pairs)} files)")

    # =========================================================================
    # 3. 前処理 (ArcSinh + StandardScaler)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Step 3: Preprocessing (ArcSinh + StandardScaler)...")
    print("=" * 60)

    # ArcSinh 変換
    X_neg_arcsinh = np.arcsinh(X_neg / cofactor)
    X_stain_arcsinh = np.arcsinh(X_stain / cofactor)

    # Negative データで Scaler を fit し、両方に適用
    scaler = StandardScaler()
    X_neg_scaled = scaler.fit_transform(X_neg_arcsinh)
    X_stain_scaled = scaler.transform(X_stain_arcsinh)

    print(f"  Scaler fitted on Negative data (mean, std per channel)")

    # =========================================================================
    # 4. 3D UMAP (fit on Negative, transform both)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Step 4: Running 2D UMAP (fit on Negative, transform both)...")
    print("=" * 60)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.3,
        metric='euclidean',
        random_state=seed
    )

    # Negative で fit + transform
    print("  Fitting UMAP on Negative data...")
    umap_neg = reducer.fit_transform(X_neg_scaled)

    # Stained を transform
    print(f"  Transforming {stain_name} data into Negative UMAP space...")
    umap_stain = reducer.transform(X_stain_scaled)

    # =========================================================================
    # 5. 蛍光強度の取得
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"Step 5: Extracting fluorescence intensity...")
    print("=" * 60)

    # FCS カラムからマーカーチャネルを検索
    stain_col = None
    for col in df_stain_fcs.columns:
        if stain_name.lower() in col.lower() and 'area' in col.lower():
            stain_col = col
            break

    if stain_col is not None:
        stain_intensity_stain = np.arcsinh(df_stain_fcs[stain_col].values / cofactor)
        
        neg_stain_col = stain_col if stain_col in df_neg_fcs.columns else None
        if neg_stain_col is None:
            for col in df_neg_fcs.columns:
                if stain_name.lower() in col.lower() and 'area' in col.lower():
                    neg_stain_col = col
                    break
                    
        if neg_stain_col is not None:
            stain_intensity_neg = np.arcsinh(df_neg_fcs[neg_stain_col].values / cofactor)
        else:
            stain_intensity_neg = np.zeros(len(df_neg_fcs))
            
        stain_label = f'{stain_col} (ArcSinh)'
        print(f"  Using FCS channel: {stain_col}")
        
        vmin = min(stain_intensity_neg.min(), stain_intensity_stain.min())
        vmax = max(stain_intensity_neg.max(), stain_intensity_stain.max())
    else:
        # フォールバック
        stain_intensity_stain = X_stain_arcsinh.mean(axis=1)
        stain_intensity_neg = X_neg_arcsinh.mean(axis=1)
        stain_label = 'Mean Spectral Intensity (ArcSinh)'
        vmin, vmax = 0, 1
        print(f"  Warning: '{stain_name}' channel not found in FCS. Using mean spectral intensity.")

    # =========================================================================
    # 6. Plotly 3D 可視化
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Step 6: Generating interactive 2D plot...")
    print("=" * 60)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'Negative Control colored by {stain_label}',
            f'Stained ({stain_name}) colored by {stain_label}'
        )
    )

    # --- Left panel: Negative Intensity ---
    fig.add_trace(
        go.Scatter(
            x=umap_neg[:, 0], y=umap_neg[:, 1],
            mode='markers',
            name='Negative',
            marker=dict(
                size=3, color=stain_intensity_neg,
                cmin=vmin, cmax=vmax,
                colorscale='Jet', opacity=0.7,
                colorbar=dict(title=stain_label, x=0.45)
            ),
            showlegend=False
        ),
        row=1, col=1
    )

    # --- Right panel: Stained only, colored by fluorescence intensity ---
    fig.add_trace(
        go.Scatter(
            x=umap_stain[:, 0], y=umap_stain[:, 1],
            mode='markers',
            name=f'{stain_name} Intensity',
            marker=dict(
                size=3,
                color=stain_intensity_stain,
                cmin=vmin, cmax=vmax,
                colorscale='Jet',
                opacity=0.7,
                colorbar=dict(title=stain_label, x=1.0)
            ),
            showlegend=False
        ),
        row=1, col=2
    )

    fig.update_layout(
        title=f'Autofluorescence UMAP + {stain_name} Projection',
        width=1400,
        height=700,
        margin=dict(l=40, r=40, b=40, t=60),
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01
        )
    )
    fig.update_xaxes(title_text='UMAP 1')
    fig.update_yaxes(title_text='UMAP 2')

    # 出力
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.write_html(output_path)
    print(f"\n  Interactive 2D plot saved to: {output_path}")

    # 一時フォルダの削除
    temp_dir = os.path.join(PROJECT_ROOT, "analysis", "results", "_temp_autofluor")
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description='自家蛍光 UMAP + マーカー投影 (2D Plotly)'
    )
    parser.add_argument('--neg-dir', type=str, required=True,
                        help='ネガティブコントロールのデータディレクトリ')
    parser.add_argument('--stain-dir', type=str, required=True,
                        help='マーカー染色サンプルのデータディレクトリ')
    parser.add_argument('--stain', type=str, default='PI',
                        help='染色マーカー名 (デフォルト: PI)')
    parser.add_argument('--output', type=str, default=None,
                        help='出力 HTML ファイルのパス')
    parser.add_argument('--cofactor', type=float, default=None,
                        help='ArcSinh cofactor')
    parser.add_argument('--seed', type=int, default=42,
                        help='UMAP の乱数シード')

    args = parser.parse_args()

    if args.output is None:
        results_dir = os.path.join(PROJECT_ROOT, "analysis", "results")
        os.makedirs(results_dir, exist_ok=True)
        args.output = os.path.join(results_dir, f"autofluor_umap_{args.stain}.html")

    run_umap_autofluor(
        neg_dir=args.neg_dir,
        stain_dir=args.stain_dir,
        output_path=args.output,
        stain_name=args.stain,
        cofactor=args.cofactor,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
