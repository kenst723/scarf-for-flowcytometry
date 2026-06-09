"""
アンミキシング前後比較スペクトル密度プロット

3パネル構成:
  左: Negative (自家蛍光のみ)
  中: 染色サンプル (生スペクトル)
  右: アンミキシング後の自家蛍光 (色素成分除去済み)

Usage:
    python -m src.plot_unmixing_comparison \
        --neg-csv <path> --stain-csv <path> \
        --stain Calcein --output <path>
"""

import os
import sys
import argparse
import glob
import copy

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# プロジェクトルートを path に追加
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.unmix_spectral import PoissonUnmixer, get_spectral_features
from src.unmix_factory import get_unmixer


def _render_density_panel(ax, data, wl_values, title, cmap, intensity_bins, vmax_global=None):
    """1パネル分のスペクトル密度ヒートマップを描画する."""
    num_channels = data.shape[1]
    num_intensity_bins = len(intensity_bins) - 1

    positive_data = data[data > 0]
    if len(positive_data) == 0:
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.text(0.5, 0.5, 'No positive data', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')
        return None

    # チャンネルごとのヒストグラム
    density = np.zeros((num_intensity_bins, num_channels))
    for i in range(num_channels):
        counts, _ = np.histogram(data[:, i], bins=intensity_bins)
        density[:, i] = counts

    # 0カウントは NaN (白表示)
    density[density == 0] = np.nan

    # チャンネル番号への幾何学的マッピング
    x_grid = np.arange(num_channels + 1)
    density_grid = np.full((num_intensity_bins, num_channels), np.nan)

    for i in range(num_channels):
        density_grid[:, i] = density[:, i]

    vmax = vmax_global if vmax_global else np.nanmax(density)
    im = ax.pcolormesh(x_grid, intensity_bins, density_grid,
                       cmap=cmap, norm=LogNorm(vmin=1, vmax=vmax))
    
    ax.set_xlabel('Channel Index', fontsize=10)
    ax.set_xlim(0, num_channels)
    
    # 5チャンネルごとにラベルを表示
    tick_positions = np.arange(0, num_channels, 5)
    ax.set_xticks(tick_positions + 0.5)
    ax.set_xticklabels(tick_positions + 1)
    
    ax.set_yscale('log')
    ax.set_ylabel('Intensity (Log10)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_facecolor('white')

    return im


def plot_unmixing_comparison(neg_csv_path, stain_csv_path, output_path,
                             stain_name="Calcein", method='poisson'):
    """3パネル比較スペクトル密度プロットを生成する.

    Parameters
    ----------
    neg_csv_path : str
        ネガティブコントロールの CSV ファイルパス.
    stain_csv_path : str
        染色サンプルの CSV ファイルパス.
    output_path : str
        出力画像ファイルのパス (.png).
    stain_name : str
        色素名 (タイトル表示用).
    method : str
        アンミキシング手法 ('poisson', 'scarf', 'autoencoder').
    """
    # --- データ読み込み ---
    df_neg = pd.read_csv(neg_csv_path)
    df_stain = pd.read_csv(stain_csv_path)

    wl_features = get_spectral_features(df_neg)
    wl_values = np.array([float(f.replace('Area_', '').replace('nm', ''))
                          for f in wl_features])

    X_neg = df_neg[wl_features].values
    X_stain = df_stain[wl_features].values

    # --- アンミキシング実行 ---
    parts_neg = os.path.normpath(neg_csv_path).split(os.sep)
    date_str = parts_neg[-3]
    neg_label = parts_neg[-2]
    
    parts_stain = os.path.normpath(stain_csv_path).split(os.sep)
    stain_label = parts_stain[-2]
    
    unmixer = get_unmixer(method, X_neg, X_stain, 
                          date_str=date_str, 
                          neg_label=neg_label, 
                          stain_label=stain_label)
                          
    print(f"Unmixing {len(X_stain)} cells...")
    X_unmixed_af = unmixer.remove_stain_component(X_stain)
    X_unmixed_af = np.maximum(X_unmixed_af, 0)

    # ---------------------------------------------------------
    # Helper for saving a plot
    # ---------------------------------------------------------
    def _save_plot(x_n, x_s, x_u, title_suffix, out_path):
        from scipy.signal import savgol_filter
        window_length = 7
        polyorder = 2
        
        # NaNs protection
        x_n = np.nan_to_num(x_n, nan=0.0, posinf=0.0, neginf=0.0)
        x_s = np.nan_to_num(x_s, nan=0.0, posinf=0.0, neginf=0.0)
        x_u = np.nan_to_num(x_u, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Smooth only the unmixed AF and the Negative reference for fair comparison
        x_n = savgol_filter(x_n, window_length, polyorder, axis=1)
        x_u = savgol_filter(x_u, window_length, polyorder, axis=1)
        
        x_n = np.maximum(x_n, 0)
        x_u = np.maximum(x_u, 0)

        cmap = copy.copy(plt.get_cmap('jet'))
        cmap.set_bad(color='white')

        fig, axes = plt.subplots(1, 3, figsize=(22, 6))

        all_data = np.concatenate([x_n, x_s, x_u], axis=0)
        positive_data = all_data[all_data > 0]
        if len(positive_data) > 0:
            intensity_min = max(positive_data.min(), 1e-1)
            intensity_max = all_data.max() * 1.5
            num_intensity_bins = 256
            intensity_bins = np.logspace(np.log10(intensity_min),
                                         np.log10(intensity_max),
                                         num_intensity_bins + 1)
        else:
            intensity_bins = np.logspace(np.log10(1e-1), np.log10(1e4), 257)

        im1 = _render_density_panel(
            axes[0], x_n, wl_values,
            f'Negative Control\n(Autofluorescence Only, n={len(x_n):,})',
            cmap, intensity_bins)

        im2 = _render_density_panel(
            axes[1], x_s, wl_values,
            f'{stain_name} Stained (Raw)\n(n={len(x_s):,})',
            cmap, intensity_bins)

        im3 = _render_density_panel(
            axes[2], x_u, wl_values,
            f'{stain_name} → Unmixed AF\n(Stain Component Removed, n={len(x_u):,})',
            cmap, intensity_bins)

        for im, ax in [(im1, axes[0]), (im2, axes[1]), (im3, axes[2])]:
            if im is not None:
                fig.colorbar(im, ax=ax, label='Event Count', pad=0.02, shrink=0.85)

        unmixer_name = "AutoEncoder" if method in ['autoencoder', 'transformer'] else ("SCARF-kNN" if method == 'scarf' else f"GLM Poisson ({method})")
        slope_str = f"slope={getattr(unmixer, 'slope', 0):.4f}"
        bg_str = f"bg={getattr(unmixer, 'bg', 0):.2f}"
        
        fig.suptitle(
            f'Spectral Unmixing Comparison — {stain_name} {title_suffix}\n'
            f'Unmixer: {unmixer_name}  |  {slope_str}  {bg_str}',
            fontsize=14, fontweight='bold', y=1.03)

        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_path}")

    # 1. Save Unfiltered Plot (All Events)
    _save_plot(X_neg, X_stain, X_unmixed_af, "", output_path)


def find_csv_in_dir(results_base_dir, prefix):
    """results ディレクトリ内から指定プレフィクスのサンプル CSV を1つ返す."""
    pattern = os.path.join(results_base_dir, f"{prefix}_*", "*.csv")
    csv_files = sorted(glob.glob(pattern))
    csv_files = [p for p in csv_files if "scarf_embeddings" not in p]
    if csv_files:
        return csv_files[0]
    return None


def main():
    parser = argparse.ArgumentParser(
        description='アンミキシング前後比較スペクトル密度プロット')
    parser.add_argument('--neg-csv', type=str, default=None,
                        help='Negative CSV のパス (省略時は results から自動検索)')
    parser.add_argument('--stain-csv', type=str, default=None,
                        help='染色サンプル CSV のパス (省略時は results から自動検索)')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='結果ベースディレクトリ (自動検索用)')
    parser.add_argument('--stain', type=str, default='Calcein',
                        help='色素名 (デフォルト: Calcein)')
    parser.add_argument('--output', type=str, default=None,
                        help='出力 PNG のパス')
    parser.add_argument('--method', type=str, choices=['poisson', 'poisson_glm', 'scarf', 'autoencoder', 'transformer'], default='poisson',
                        help='アンミキシング手法 (poisson, poisson_glm, scarf, autoencoder, transformer)')
    args = parser.parse_args()

    # CSV パスの解決
    neg_csv = args.neg_csv
    stain_csv = args.stain_csv

    if (neg_csv is None or stain_csv is None) and args.results_dir:
        if neg_csv is None:
            neg_csv = find_csv_in_dir(args.results_dir, "Negative")
        if stain_csv is None:
            stain_csv = find_csv_in_dir(args.results_dir, args.stain)

    if neg_csv is None or stain_csv is None:
        print("Error: --neg-csv と --stain-csv を指定するか、--results-dir を指定してください。")
        sys.exit(1)

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(stain_csv),
            f"unmixing_comparison_{args.stain}.png")

    plot_unmixing_comparison(neg_csv, stain_csv, args.output,
                             stain_name=args.stain, method=args.method)


if __name__ == '__main__':
    main()
