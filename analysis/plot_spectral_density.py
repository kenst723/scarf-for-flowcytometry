import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import copy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.unmix_spectral import PoissonUnmixer

def plot_cytek_style_density(ax, data, wl_values, title):
    num_channels = data.shape[1]
    
    # We want jet colormap with NaN as white
    cmap = copy.copy(plt.get_cmap('jet'))
    cmap.set_bad(color='white')
    
    intensity_min = max(data[data > 0].min(), 1e-1)
    intensity_max = data.max() * 1.5
    num_intensity_bins = 256
    intensity_bins = np.logspace(np.log10(intensity_min), np.log10(intensity_max), num_intensity_bins + 1)

    density = np.zeros((num_intensity_bins, num_channels))
    for i in range(num_channels):
        counts, _ = np.histogram(data[:, i], bins=intensity_bins)
        density[:, i] = counts

    density[density == 0] = np.nan

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
    ax.set_facecolor('white')

    return im

def main():
    import glob
    neg_dir = r"analysis\results\Experiment 2026!05!27 9!30\negative_B01"
    calcein_dir = r"analysis\results\Experiment 2026!05!27 9!30\Calcein_A01"
    output_png = r"analysis\results\Experiment 2026!05!27 9!30\spectral_density_plot.png"
    
    neg_csvs = glob.glob(os.path.join(neg_dir, "*.csv"))
    calcein_csvs = glob.glob(os.path.join(calcein_dir, "*.csv"))
    
    if not neg_csvs or not calcein_csvs:
        print("Could not find CSV files.")
        return
        
    neg_csv = neg_csvs[0]
    calcein_csv = calcein_csvs[0]
    
    print(f"Loading {neg_csv}...")
    df_neg = pd.read_csv(neg_csv)
    print(f"Loading {calcein_csv}...")
    df_stain = pd.read_csv(calcein_csv)
    
    wl_features = [c for c in df_neg.columns if c.startswith('Area_') and c.endswith('nm') and '638.6nm' not in c]
    wl_values = [float(c.replace('Area_', '').replace('nm', '')) for c in wl_features]
    
    X_neg = df_neg[wl_features].values
    X_stain = df_stain[wl_features].values
    
    print("Unmixing to get Reconstructed AF...")
    unmixer = PoissonUnmixer().fit(X_neg, X_stain)
    X_reconstructed_af = unmixer.remove_stain_component(X_stain)
    
    # Clip negative values to prevent LogNorm errors if any
    X_reconstructed_af = np.maximum(X_reconstructed_af, 1e-1)
    
    print("Plotting Cytek-style Spectral Density...")
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), dpi=150)
    
    im1 = plot_cytek_style_density(axes[0], X_neg, wl_values, "1. Negative Control")
    fig.colorbar(im1, ax=axes[0], label='Event Count', pad=0.02, shrink=0.9)
    
    im2 = plot_cytek_style_density(axes[1], X_stain, wl_values, "2. Calcein Sample (Raw)")
    fig.colorbar(im2, ax=axes[1], label='Event Count', pad=0.02, shrink=0.9)
    
    im3 = plot_cytek_style_density(axes[2], X_reconstructed_af, wl_values, "3. Unmixing (Reconstructed AF)")
    fig.colorbar(im3, ax=axes[2], label='Event Count', pad=0.02, shrink=0.9)
    
    plt.tight_layout()
    plt.savefig(output_png)
    print(f"Saved Spectral Density Plot to {output_png}")

if __name__ == "__main__":
    main()
