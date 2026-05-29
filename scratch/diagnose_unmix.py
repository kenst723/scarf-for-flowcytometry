"""
Diagnostic: Compare Negative (Panel 1) vs Reconstructed AF (Panel 3) quantitatively.
- Channel-by-channel median comparison
- Ratio of medians to identify where Calcein leaks through
"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.unmix_spectral import PoissonUnmixer

def main():
    neg_dir = r"analysis\results\Experiment 2026!05!27 9!30\negative_B01"
    calcein_dir = r"analysis\results\Experiment 2026!05!27 9!30\Calcein_A01"

    neg_csv = glob.glob(os.path.join(neg_dir, "*.csv"))[0]
    calcein_csv = glob.glob(os.path.join(calcein_dir, "*.csv"))[0]

    df_neg = pd.read_csv(neg_csv)
    df_stain = pd.read_csv(calcein_csv)

    wl_features_neg = [c for c in df_neg.columns if c.startswith('Area_') and c.endswith('nm') and '638.6nm' not in c]
    wl_features_stain = [c for c in df_stain.columns if c.startswith('Area_') and c.endswith('nm') and '638.6nm' not in c]
    print(f"Neg features length: {len(wl_features_neg)}, Stain features length: {len(wl_features_stain)}")
    print(f"Neg features: {wl_features_neg}")
    print(f"Are features identical?: {wl_features_neg == wl_features_stain}")

    wl_features = wl_features_neg
    wl_values = np.array([float(c.replace('Area_', '').replace('nm', '')) for c in wl_features])

    X_neg = df_neg[wl_features].values
    X_stain = df_stain[wl_features].values

    unmixer = PoissonUnmixer(max_iter=5)
    unmixer.fit(X_neg, X_stain)

    print(f"Raw median Neg   at 420nm: {np.median(X_neg[:, 0]):.1f}, at 515nm: {np.median(X_neg[:, 6]):.1f}, at 750nm: {np.median(X_neg[:, -3]):.1f}")
    print(f"Raw median Stain at 420nm: {np.median(X_stain[:, 0]):.1f}, at 515nm: {np.median(X_stain[:, 6]):.1f}, at 750nm: {np.median(X_stain[:, -3]):.1f}")

    # Get coefficients
    C = unmixer._unmix_sequential(X_stain)
    c_af = C[:, 0]
    c_stain = C[:, 1]

    # Reconstructed AF = X - c_stain * S_Stain
    X_recon_af = X_stain - c_stain[:, None] * unmixer.S_Stain[None, :]
    X_recon_af = np.maximum(X_recon_af, 0.1)  # clip for log

    # Channel-by-channel comparison
    median_neg = np.median(X_neg, axis=0)
    median_recon = np.median(X_recon_af, axis=0)
    ratio = median_recon / (median_neg + 1e-9)

    print("=== Channel-by-channel Median Comparison ===")
    print(f"{'Wavelength':>12} {'Neg Median':>12} {'Recon Median':>14} {'Ratio':>8}")
    for i in range(len(wl_values)):
        marker = " <-- HIGH" if ratio[i] > 2.0 else (" <-- LOW" if ratio[i] < 0.5 else "")
        print(f"{wl_values[i]:>12.1f} {median_neg[i]:>12.1f} {median_recon[i]:>14.1f} {ratio[i]:>8.2f}{marker}")

    print(f"\nOverall ratio (total): {np.sum(median_recon)/np.sum(median_neg):.3f}")
    print(f"Ratio at Calcein peak (~515nm): {ratio[np.argmin(np.abs(wl_values - 515))]:.3f}")
    print(f"Ratio at tail (~750nm): {ratio[np.argmin(np.abs(wl_values - 750))]:.3f}")

    # Also check: what does S_Stain look like?
    print("\n=== S_Stain reference spectrum (top 10 channels) ===")
    top_idx = np.argsort(unmixer.S_Stain)[::-1][:10]
    for idx in top_idx:
        print(f"  {wl_values[idx]:>8.1f}nm: {unmixer.S_Stain[idx]:.6f}")

    # Check af_scale
    tail_slice = slice(-5, None)
    median_stained_bright = np.median(X_stain[np.sum(X_stain, axis=1) >= np.percentile(np.sum(X_stain, axis=1), 98)], axis=0)
    af_scale = np.sum(median_stained_bright[tail_slice]) / (np.sum(median_neg[tail_slice]) + 1e-9)
    print(f"\nAF scale factor used: {af_scale:.2f}")

    # Check c_stain distribution
    print(f"\n=== c_stain distribution ===")
    print(f"  Mean: {np.mean(c_stain):.1f}")
    print(f"  Median: {np.median(c_stain):.1f}")
    print(f"  Min: {np.min(c_stain):.1f}")
    print(f"  Max: {np.max(c_stain):.1f}")
    print(f"  Negative c_stain count: {np.sum(c_stain < 0)} / {len(c_stain)}")

    # Plot comparison
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    ax = axes[0]
    ax.plot(wl_values, median_neg, 'b-o', ms=3, label='Negative Control (median)')
    ax.plot(wl_values, median_recon, 'r-s', ms=3, label='Reconstructed AF (median)')
    ax.set_yscale('log')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity')
    ax.set_title('Median Spectrum Comparison: Negative vs Reconstructed AF')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(wl_values, ratio, 'k-o', ms=3)
    ax.axhline(1.0, color='green', linestyle='--', label='Perfect match (1.0)')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Ratio (Recon / Neg)')
    ax.set_title('Ratio of Reconstructed AF to Negative Control (per channel)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 5)

    plt.tight_layout()
    plt.savefig('scratch/diagnostic_comparison.png', dpi=150)
    print("\nSaved diagnostic_comparison.png")

if __name__ == "__main__":
    main()
