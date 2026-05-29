"""
Final approach: 
- S_Stain = simple subtraction (correct for same cell population)
- Sequential estimation (tail → c_af, then residual → c_stain)
- Key: S_Stain now includes the tail spreading, so we need to
  account for S_Stain contribution when estimating c_af from tail
"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import copy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

def unmix_sequential_v2(X, S_AF, S_Stain, n_iter=3):
    """Sequential estimation with iterative refinement.
    
    Since S_Stain has non-zero tail (optical spreading), we can't
    just use tail channels for c_af independently. Instead:
    1. Initial c_af from tail (ignoring small S_Stain tail)
    2. Initial c_stain from peak residual  
    3. Refine c_af: subtract c_stain * S_Stain from tail, re-estimate c_af
    4. Repeat 2-3
    """
    N = X.shape[0]
    
    # Define channel masks
    tail_mask = np.zeros(len(S_AF), dtype=bool)
    tail_mask[-8:] = True  # >700nm
    
    # All non-tail channels for c_stain estimation
    stain_mask = ~tail_mask
    
    S_AF_tail = S_AF[tail_mask]
    S_Stain_tail = S_Stain[tail_mask]
    S_Stain_stain = S_Stain[stain_mask]
    S_AF_stain = S_AF[stain_mask]
    
    # Step 1: Initial c_af from tail (ignore S_Stain tail for now)
    X_tail = X[:, tail_mask]
    denom_af = np.dot(S_AF_tail, S_AF_tail) + 1e-9
    c_af = X_tail @ S_AF_tail / denom_af
    
    for it in range(n_iter):
        # Step 2: c_stain from non-tail residual
        residual_stain = X[:, stain_mask] - c_af[:, None] * S_AF_stain[None, :]
        denom_stain = np.dot(S_Stain_stain, S_Stain_stain) + 1e-9
        c_stain = residual_stain @ S_Stain_stain / denom_stain
        
        # Step 3: Refine c_af from tail, subtracting S_Stain tail contribution
        residual_tail = X_tail - c_stain[:, None] * S_Stain_tail[None, :]
        c_af = residual_tail @ S_AF_tail / denom_af
    
    return np.column_stack((c_af, c_stain))

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
    
    # S_AF
    S_AF = median_neg / (np.sum(median_neg) + 1e-9)
    
    # S_Stain: simple subtraction (same cell population)
    total = np.sum(X_stain, axis=1)
    bright = X_stain[total >= np.percentile(total, 98)]
    median_bright = np.median(bright, axis=0)
    S_Stain = median_bright - median_neg
    S_Stain = np.maximum(S_Stain, 0)
    S_Stain /= (np.sum(S_Stain) + 1e-9)
    
    # Sequential unmixing v2
    C = unmix_sequential_v2(X_stain, S_AF, S_Stain, n_iter=3)
    c_af, c_stain = C[:, 0], C[:, 1]
    
    # Reconstructed AF
    X_recon = X_stain - c_stain[:, None] * S_Stain[None, :]
    X_recon = np.maximum(X_recon, 0.1)
    
    median_recon = np.median(X_recon, axis=0)
    ratio = median_recon / (median_neg + 1e-9)
    
    print("=== Sequential v2: Channel-by-channel ratio ===")
    for i in range(len(wl_values)):
        marker = ""
        if ratio[i] > 1.3: marker = " HIGH"
        elif ratio[i] < 0.7 and ratio[i] >= 0: marker = " LOW"
        print(f"  {wl_values[i]:>7.1f}nm  Ratio={ratio[i]:>6.2f}{marker}")
    
    print(f"\nc_af: median={np.median(c_af):.0f}")
    print(f"c_stain: median={np.median(c_stain):.0f}")
    
    # Verify on Negative cells
    C_neg = unmix_sequential_v2(X_neg, S_AF, S_Stain, n_iter=3)
    print(f"\nNeg: c_af median={np.median(C_neg[:, 0]):.0f}, c_stain median={np.median(C_neg[:, 1]):.0f}")
    
    # === Density plots ===
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), dpi=150)
    
    im1 = plot_density(axes[0], X_neg, wl_values, "1. Negative Control")
    fig.colorbar(im1, ax=axes[0], label='Count', pad=0.02, shrink=0.9)
    
    im2 = plot_density(axes[1], X_stain, wl_values, "2. Calcein (Raw)")
    fig.colorbar(im2, ax=axes[1], label='Count', pad=0.02, shrink=0.9)
    
    im3 = plot_density(axes[2], X_recon, wl_values, "3. Reconstructed AF (Sequential v2)")
    fig.colorbar(im3, ax=axes[2], label='Count', pad=0.02, shrink=0.9)
    
    plt.tight_layout()
    plt.savefig('scratch/final_density_test.png', dpi=150)
    print("\nSaved final_density_test.png")

if __name__ == "__main__":
    main()
