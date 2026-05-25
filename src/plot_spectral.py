"""
スペクトル密度プロット

CSVファイルからスペクトル密度ヒートマップを生成する。

Usage:
    python -m src.plot_spectral --csv <path> --output <path>
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import copy


def plot_spectral_density(csv_path, output_path):
    """
    CSV ファイルからスペクトル密度プロットを生成する。

    Parameters
    ----------
    csv_path : str
        入力 CSV ファイルのパス
    output_path : str
        出力画像ファイルのパス (.png)
    """
    df = pd.read_csv(csv_path)

    area_ch_cols = [c for c in df.columns if c.startswith('Area_') and not c.endswith('nm')]
    area_wl_cols = [c for c in df.columns if c.startswith('Area_') and c.endswith('nm')]
    num_channels = len(area_ch_cols)

    data_ch = df[area_ch_cols].values
    data_wl = df[area_wl_cols].values

    ch_labels = [c.replace('Area_', '') for c in area_ch_cols]
    wl_values = [float(c.replace('Area_', '').replace('nm', '')) for c in area_wl_cols]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Custom colormap to make 0 counts transparent/white like the official software
    cmap = copy.copy(plt.get_cmap('jet'))
    cmap.set_bad(color='white')

    def _plot_density(ax, data, title):
        intensity_min = max(data[data > 0].min(), 1e-1)
        intensity_max = data.max() * 1.5
        num_intensity_bins = 256
        intensity_bins = np.logspace(np.log10(intensity_min), np.log10(intensity_max), num_intensity_bins + 1)

        # Standard binning per channel
        density = np.zeros((num_intensity_bins, num_channels))
        for i in range(num_channels):
            counts, _ = np.histogram(data[:, i], bins=intensity_bins)
            density[:, i] = counts

        # Set 0 counts to NaN so they render as white
        density[density == 0] = np.nan

        # Geometrical spacing: map each channel to its actual wavelength position
        x_grid = np.arange(400, 801, 1)
        density_grid = np.full((num_intensity_bins, len(x_grid)-1), np.nan)

        for i in range(num_channels):
            center = wl_values[i]
            w = 10 if i < 2 else (wl_values[i] - wl_values[i-1])
            start_idx = int(center - w/2 - 400)
            end_idx = int(center + w/2 - 400)
            start_idx = max(0, min(start_idx, len(x_grid)-2))
            end_idx = max(0, min(end_idx, len(x_grid)-1))
            for j in range(start_idx, end_idx):
                density_grid[:, j] = density[:, i]

        # Mask out the 638.6nm channel (laser noise)
        mask_idx = int(638.6 - 400)
        density_grid[:, mask_idx-1:mask_idx+2] = np.nan

        im = ax.pcolormesh(x_grid, intensity_bins, density_grid, cmap=cmap, norm=LogNorm(vmin=1, vmax=np.nanmax(density)))
        ax.set_xlabel('Wavelength (nm)', fontsize=11)
        ax.set_xlim(420, 800)
        ax.set_xticks([420, 515, 610, 705, 800])
        ax.set_xticklabels(['420', '515', '610', '705', '800'], fontsize=11)

        ax.set_yscale('log')
        ax.set_ylabel('Intensity', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')

        # White background
        ax.set_facecolor('white')

        fig.colorbar(im, ax=ax, label='Event Count', pad=0.02, shrink=0.9)

    _plot_density(ax1, data_ch, 'Area (Channel) — Geometrical Spacing')
    _plot_density(ax2, data_wl, 'Area (Wavelength) — Geometrical Spacing')

    sample_name = os.path.splitext(os.path.basename(csv_path))[0]
    fig.suptitle(f'Spectral Density — {sample_name}', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='スペクトル密度プロット生成')
    parser.add_argument('--csv', type=str, required=True, help='入力 CSV ファイルのパス')
    parser.add_argument('--output', type=str, default=None, help='出力画像のパス (.png)')
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(args.csv)[0]
        args.output = base + '_spectral_density.png'

    plot_spectral_density(args.csv, args.output)


if __name__ == '__main__':
    main()
