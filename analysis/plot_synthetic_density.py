import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import copy
from matplotlib.ticker import FuncFormatter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from config import COFACTOR

def perform_wlsm(X, S, epsilon=1.0):
    N, M = X.shape
    num_fluor = S.shape[1]
    C = np.zeros((N, num_fluor))
    for i in range(N):
        w = 1.0 / np.maximum(X[i], epsilon)
        St_W = S.T * w
        try:
            inv = np.linalg.inv(St_W @ S)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(St_W @ S)
        C[i] = inv @ St_W @ X[i]
    return C

def plot_density_arcsinh(ax, data, wl_values, title):
    # Use arcsinh scale for bins to properly handle negative values
    arcsinh_data = np.arcsinh(data / COFACTOR)
    
    val_min = np.min(arcsinh_data)
    val_max = np.max(arcsinh_data)
    
    num_intensity_bins = 256
    intensity_bins = np.linspace(val_min, val_max, num_intensity_bins + 1)
    
    num_channels = data.shape[1]
    density = np.zeros((num_intensity_bins, num_channels))
    for i in range(num_channels):
        counts, _ = np.histogram(arcsinh_data[:, i], bins=intensity_bins)
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
            
    mask_idx = int(638.6 - 400)
    density_grid[:, mask_idx-1:mask_idx+2] = np.nan
    
    cmap = copy.copy(plt.get_cmap('jet'))
    cmap.set_bad(color='white')
    
    im = ax.pcolormesh(x_grid, intensity_bins, density_grid, cmap=cmap, norm=LogNorm(vmin=1, vmax=np.nanmax(density)))
    
    ax.set_xlabel('Wavelength (nm)', fontsize=11)
    ax.set_xlim(420, 800)
    ax.set_xticks([420, 515, 610, 705, 800])
    ax.set_xticklabels(['420', '515', '610', '705', '800'], fontsize=11)

    # Format Y-axis to show original intensity values, not arcsinh values
    def arcsinh_formatter(y, pos):
        val = np.sinh(y) * COFACTOR
        if abs(val) >= 10000:
            return f"{val:.1e}"
        else:
            return f"{int(val)}"
            
    ax.yaxis.set_major_formatter(FuncFormatter(arcsinh_formatter))
    
    # Set y-ticks at typical decades in arcsinh space
    ticks_vals = [-1000, 0, 1000, 10000, 100000, 1000000, 10000000]
    ticks_arcsinh = np.arcsinh(np.array(ticks_vals) / COFACTOR)
    ax.set_yticks(ticks_arcsinh)
    
    ax.set_ylabel('Intensity (ArcSinh Scale)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_facecolor('white')
    return im

def main():
    neg_csv = r"analysis\results\2026-05-27\negative_B01\B01 Well - B01_20260527_110441.csv"
    calcein_csv = r"analysis\results\2026-05-27\Calcein_A01\A01 Well - A01_20260527_110439.csv"
    
    print("Loading data...")
    df_neg = pd.read_csv(neg_csv)
    df_stain = pd.read_csv(calcein_csv)
    
    wl_features = [c for c in df_neg.columns if c.startswith('Area_') and c.endswith('nm') and '638.6nm' not in c]
    wl_values = [float(c.replace('Area_', '').replace('nm', '')) for c in wl_features]
    
    X_neg_full = df_neg[wl_features].values
    X_stain_full = df_stain[wl_features].values
    
    print("Creating Synthetic Mixture...")
    np.random.seed(42)
    idx_neg = np.random.choice(len(X_neg_full), 2900, replace=False)
    idx_stain = np.random.choice(len(X_stain_full), 2000, replace=False)
    
    X_mix = np.vstack([X_neg_full[idx_neg], X_stain_full[idx_stain]])
    
    # Calculate References
    S_AF = np.median(X_neg_full, axis=0)
    S_AF = S_AF / np.sum(S_AF)
    
    total_intensity = np.sum(X_mix, axis=1)
    bright_cells = X_mix[total_intensity >= np.percentile(total_intensity, 95)]
    S_bright = np.median(bright_cells, axis=0)
    peak_idx = np.argmax(S_bright / (S_AF + 1e-9))
    
    stained_cells = X_mix[X_mix[:, peak_idx] >= np.percentile(X_mix[:, peak_idx], 95)]
    S_Stain = np.median(stained_cells, axis=0)
    S_Stain = np.maximum(S_Stain - (S_AF * np.min(S_Stain / (S_AF + 1e-9))), 0)
    S_Stain = S_Stain / np.sum(S_Stain)
    
    print("Unmixing...")
    S = np.column_stack((S_AF, S_Stain))
    C_mix = perform_wlsm(X_mix, S)
    X_mix_unmixed = X_mix - C_mix[:, 1:2] * S_Stain[None, :]
    
    print("Plotting...")
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    
    plot_density_arcsinh(axes[0], X_neg_full[idx_neg], wl_values, '1. True Negative Cells (2900 cells)')
    plot_density_arcsinh(axes[1], X_mix, wl_values, '2. Raw Synthetic Mixture (2900 Neg + 2000 Calcein)')
    im = plot_density_arcsinh(axes[2], X_mix_unmixed, wl_values, '3. WLSM Unmixed AF (Calcein Subtracted)')
    
    # Match Y-axis limits
    all_data = np.concatenate([X_neg_full[idx_neg], X_mix, X_mix_unmixed])
    val_min = np.min(np.arcsinh(all_data / COFACTOR))
    val_max = np.max(np.arcsinh(all_data / COFACTOR))
    
    for ax in axes:
        ax.set_ylim(val_min, val_max)
        
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Event Count (Log Scale)', pad=0.02, shrink=0.9)
    fig.suptitle('Spectral Density Verification of Unmixing (ArcSinh Scale)', fontsize=15, fontweight='bold', y=1.05)
    
    output_path = r'analysis\results\2026-05-27\synthetic_density_verification.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
