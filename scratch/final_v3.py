"""
Final v3: Use the fact that same cell population → c_af should be ~same as Neg median.
Instead of estimating c_af from spectral fitting, use a SIZE PROXY from the data itself.
The tail region (>700nm where Calcein doesn't emit) gives a direct measurement of each
cell's AF level independent of Calcein.
"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import copy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

def plot_density(ax, data, wl_values, title):
    num_channels = data.shape[1]
    cmap = copy.copy(plt.get_cmap('jet'))
    cmap.set_bad(color='white')
    
    intensity_min = max(data[data > 0].min(), 1e-1)
    intensity_max = data.max() * 1.5
    num_bins = 256
    intensity_bins = np.logspace(np.log10(intensity_min), np.log10(intensity_max), num_bins + 1)
    
    density = np.zeros((num_bins, num_channels))
    for i in range(num_channels):
        counts, _ = np.histogram(data[:, i], bins=intensity_bins)
        density[:, i] = counts
    density[density == 0] = np.nan
    
    x_grid = np.arange(400, 801, 1)
    density_grid = np.full((num_bins, len(x_grid)-1), np.nan)
    for i in range(num_channels):
        center = wl_values[i]
        w = 10 if i < 2 else (wl_values[i] - wl_values[i-1])
        s = max(0, min(int(center - w/2 - 400), len(x_grid)-2))
        e = max(0, min(int(center + w/2 - 400), len(x_grid)-1))
        for j in range(s, e):
            density_grid[:, j] = density[:, i]
    mask_idx = int(638.6 - 400)
    density_grid[:, mask_idx-1:mask_idx+2] = np.nan
    
    im = ax.pcolormesh(x_grid, intensity_bins, density_grid, cmap=cmap,
                       norm=LogNorm(vmin=1, vmax=np.nanmax(density)))
    ax.set_xlabel('Wavelength (nm)')
    ax.set_xlim(420, 800)
    ax.set_yscale('log')
    ax.set_ylabel('Intensity')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_facecolor('white')
    return im

def main():
    neg_dir = r"analysis\results\Experiment 2026!05!27 9!30\negative_B01"
    calcein_dir = r"analysis\results\Experiment 2026!05!27 9!30\Calcein_A01"
    neg_csv = glob.glob(os.path.join(neg_dir, "*.csv"))[0]
    calcein_csv = glob.glob(os.path.join(calcein_dir, "*.csv"))[0]

    df_neg = pd.read_csv(neg_csv)
    df_stain = pd.read_csv(calcein_csv)

    wl_features = [c for c in df_neg.columns if c.startswith('Area_') and c.endswith('nm') and '638.6nm' not in c]
    wl_values = np.array([float(c.replace('Area_', '').replace('nm', '')) for c in wl_features])
    X_neg = df_neg[wl_features].values
    X_stain = df_stain[wl_features].values
    
    median_neg = np.median(X_neg, axis=0)
    S_AF_unnorm = median_neg.copy()  # unnormalized AF shape
    
    # S_Stain: simple subtraction (correct for same cell population)
    total = np.sum(X_stain, axis=1)
    bright = X_stain[total >= np.percentile(total, 98)]
    median_bright = np.median(bright, axis=0)
    S_Stain_unnorm = median_bright - median_neg
    S_Stain_unnorm = np.maximum(S_Stain_unnorm, 0)
    
    # Normalize
    S_AF = S_AF_unnorm / (np.sum(S_AF_unnorm) + 1e-9)
    S_Stain = S_Stain_unnorm / (np.sum(S_Stain_unnorm) + 1e-9)
    
    # === Per-cell unmixing: direct AF estimation from tail ===
    # At tail channels (>700nm), Calcein contribution is tiny.
    # S_Stain_tail / S_AF_tail ratio tells us how much contamination exists.
    tail_mask = np.zeros(len(S_AF), dtype=bool)
    tail_mask[-8:] = True
    
    # For each cell, the AF at the tail gives us the cell's individual AF scale
    # X_tail = c_af * S_AF_tail + c_stain * S_Stain_tail
    # Since S_Stain_tail is very small relative to S_AF_tail:
    print(f"S_Stain_tail / S_AF_tail ratio: {np.sum(S_Stain[tail_mask]) / np.sum(S_AF[tail_mask]):.4f}")
    print(f"(If << 1, tail is dominated by AF and we can estimate c_af directly)")
    
    # Direct per-cell AF: scale = tail_intensity / median_neg_tail_intensity
    tail_neg_total = np.sum(median_neg[tail_mask])
    per_cell_af_scale = np.sum(X_stain[:, tail_mask], axis=1) / (tail_neg_total + 1e-9)
    
    # Per-cell AF reconstruction: scale * median_neg
    X_af_estimated = per_cell_af_scale[:, None] * median_neg[None, :]
    
    # Calcein component = Raw - AF
    X_calcein = X_stain - X_af_estimated
    
    # c_stain = projection onto S_Stain
    denom = np.dot(S_Stain, S_Stain) + 1e-9
    c_stain = X_calcein @ S_Stain / denom
    
    # Reconstructed AF = Raw - c_stain * S_Stain (normalized)
    X_recon = X_stain - c_stain[:, None] * S_Stain[None, :]
    X_recon = np.maximum(X_recon, 0.1)
    
    # Check ratios
    median_recon = np.median(X_recon, axis=0)
    ratio = median_recon / (median_neg + 1e-9)
    
    print("\n=== Direct AF estimation: Channel-by-channel ratio ===")
    for i in range(len(wl_values)):
        marker = ""
        if ratio[i] > 1.5: marker = " *** HIGH"
        elif ratio[i] < 0.5 and ratio[i] >= 0: marker = " *** LOW"
        print(f"  {wl_values[i]:>7.1f}nm  Neg={median_neg[i]:>8.1f}  Recon={median_recon[i]:>8.1f}  Ratio={ratio[i]:>6.2f}{marker}")
    
    print(f"\nc_stain: median={np.median(c_stain):.0f}")
    print(f"af_scale: median={np.median(per_cell_af_scale):.3f}")
    print(f"Negative c_stain: {np.sum(c_stain < 0)} / {len(c_stain)}")
    
    # Also test on Neg cells
    neg_af_scale = np.sum(X_neg[:, tail_mask], axis=1) / (tail_neg_total + 1e-9)
    neg_af_est = neg_af_scale[:, None] * median_neg[None, :]
    neg_calcein = X_neg - neg_af_est
    neg_c_stain = neg_calcein @ S_Stain / denom
    print(f"\nNeg cells: af_scale median={np.median(neg_af_scale):.3f}, c_stain median={np.median(neg_c_stain):.0f}")
    
    # Density plots
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), dpi=150)
    im1 = plot_density(axes[0], X_neg, wl_values, "1. Negative Control")
    fig.colorbar(im1, ax=axes[0], label='Count', pad=0.02, shrink=0.9)
    im2 = plot_density(axes[1], X_stain, wl_values, "2. Calcein (Raw)")
    fig.colorbar(im2, ax=axes[1], label='Count', pad=0.02, shrink=0.9)
    im3 = plot_density(axes[2], X_recon, wl_values, "3. Reconstructed AF")
    fig.colorbar(im3, ax=axes[2], label='Count', pad=0.02, shrink=0.9)
    
    plt.tight_layout()
    plt.savefig('scratch/final_v3_density.png', dpi=150)
    print("\nSaved final_v3_density.png")

if __name__ == "__main__":
    main()
