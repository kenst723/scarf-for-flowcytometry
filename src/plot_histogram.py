"""
蛍光強度ヒストグラム

FCS ファイルから指定染色チャネルの蛍光強度を読み込み、
横軸に蛍光強度、縦軸にイベント数のヒストグラムを生成する。

Usage:
    python -m src.plot_histogram --fcs <fcs_file> --stain PI --output <output_image>
"""

import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fcsparser

# プロジェクトルートを path に追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COFACTOR


def plot_histogram(fcs_file, output_image, stain_name, cofactor=None, bins=256):
    """
    FCS ファイルから蛍光強度ヒストグラムを生成する。

    Parameters
    ----------
    fcs_file : str
        .fcs ファイルのパス
    output_image : str
        出力画像のパス (.png)
    stain_name : str
        染色名 (例: "PI", "Calcein")。FCS カラムから対応チャネルを検索する。
    cofactor : float, optional
        ArcSinh 変換の cofactor。None の場合は config.COFACTOR を使用。
    bins : int, optional
        ヒストグラムのビン数 (デフォルト: 256)
    """
    if cofactor is None:
        cofactor = COFACTOR

    # --- 1. FCS データ読み込み ---
    meta, df_fcs = fcsparser.parse(fcs_file, reformat_meta=True)

    # --- 2. 染色チャネルを検索 ---
    stain_col = None
    if stain_name and stain_name.lower() != 'negative':
        for col in df_fcs.columns:
            if stain_name.lower() in col.lower() and 'area' in col.lower():
                stain_col = col
                break

    if stain_col is None:
        print(f"  Warning: Stain '{stain_name}' not found in FCS columns.")
        print(f"  Available columns: {list(df_fcs.columns)}")
        return None

    print(f"  Channel: {stain_col}")

    raw_values = df_fcs[stain_col].values

    # --- 3. ArcSinh 変換 ---
    transformed = np.arcsinh(raw_values / cofactor)

    # --- 4. プロット ---
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(transformed, bins=bins, color='#5B8BD4', edgecolor='none', alpha=0.85)

    ax.set_xlabel(f'{stain_col} (ArcSinh, cofactor={cofactor})', fontsize=12)
    ax.set_ylabel('Event Count', fontsize=12)

    sample_name = os.path.splitext(os.path.basename(fcs_file))[0]
    ax.set_title(f'Fluorescence Intensity Histogram — {sample_name}', fontsize=13, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=10)

    # イベント数とチャネル情報をアノテーション
    n_events = len(raw_values)
    ax.text(0.97, 0.95, f'n = {n_events:,}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=11, color='#444444',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.8))

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_image) or '.', exist_ok=True)
    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_image}")
    return output_image


def main():
    parser = argparse.ArgumentParser(description='蛍光強度ヒストグラム生成')
    parser.add_argument('--fcs', type=str, required=True, help='.fcs ファイルのパス')
    parser.add_argument('--stain', type=str, required=True, help='染色名 (例: PI, Calcein)')
    parser.add_argument('--output', type=str, default=None, help='出力画像のパス (.png)')
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(args.fcs)[0]
        args.output = base + '_histogram.png'

    plot_histogram(args.fcs, args.output, args.stain)


if __name__ == '__main__':
    main()
