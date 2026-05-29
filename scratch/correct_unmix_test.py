"""
Given: Same cell population (J774A.1 M0) on SA3800
Therefore: ANY difference between Neg and Calcein = Calcein-related (including optical spreading)
So: S_Stain = simple subtraction (no AF scaling needed)
Test: Does simple subtraction + OLS work now that we know the biology?
"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

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

    # === Correct S_Stain: simple 1:1 subtraction ===
    # Same cell population → all differences are Calcein-related
    median_neg = np.median(X_neg, axis=0)
    
    # Use top 2% bright cells for stain reference
    total_intensity = np.sum(X_stain, axis=1)
    bright = X_stain[total_intensity >= np.percentile(total_intensity, 98)]
    median_bright = np.median(bright, axis=0)
    
    S_Stain_simple = median_bright - median_neg
    S_Stain_simple = np.maximum(S_Stain_simple, 0)
    S_Stain_simple /= (np.sum(S_Stain_simple) + 1e-9)
    
    S_AF = median_neg / (np.sum(median_neg) + 1e-9)
    S = np.column_stack((S_AF, S_Stain_simple))
    
    print("=== S_Stain shape: simple subtraction ===")
    print("Channel    S_AF       S_Stain")
    for i in range(len(wl_values)):
        print(f"  {wl_values[i]:>7.1f}nm  {S_AF[i]:.6f}  {S_Stain_simple[i]:.6f}")
    
    # === Test: OLS unmixing with correct S_Stain ===
    C_ols = np.linalg.lstsq(S, X_stain.T, rcond=None)[0].T  # (N, 2)
    c_af_ols = C_ols[:, 0]
    c_stain_ols = C_ols[:, 1]
    
    # Reconstructed AF
    X_recon = X_stain - c_stain_ols[:, None] * S_Stain_simple[None, :]
    X_recon = np.maximum(X_recon, 0.1)
    
    median_recon = np.median(X_recon, axis=0)
    ratio = median_recon / (median_neg + 1e-9)
    
    print("\n=== OLS Unmixing: Channel-by-channel ratio ===")
    print(f"{'Wavelength':>12} {'Neg':>10} {'Recon':>10} {'Ratio':>8}")
    for i in range(len(wl_values)):
        print(f"{wl_values[i]:>12.1f} {median_neg[i]:>10.1f} {median_recon[i]:>10.1f} {ratio[i]:>8.2f}")
    
    print(f"\nOverall ratio: {np.sum(median_recon)/np.sum(median_neg):.3f}")
    print(f"Ratio at peak (~515nm): {ratio[np.argmin(np.abs(wl_values - 515))]:.3f}")
    print(f"Ratio at tail (~750nm): {ratio[np.argmin(np.abs(wl_values - 750))]:.3f}")
    
    print(f"\nc_af: mean={np.mean(c_af_ols):.0f}, median={np.median(c_af_ols):.0f}")
    print(f"c_stain: mean={np.mean(c_stain_ols):.0f}, median={np.median(c_stain_ols):.0f}")
    print(f"Negative c_stain: {np.sum(c_stain_ols < 0)} / {len(c_stain_ols)}")
    
    # Also test on Negative cells (c_stain should be ~0)
    C_neg_ols = np.linalg.lstsq(S, X_neg.T, rcond=None)[0].T
    print(f"\nNeg cells: c_af median={np.median(C_neg_ols[:, 0]):.0f}, c_stain median={np.median(C_neg_ols[:, 1]):.0f}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.plot(wl_values, S_AF, 'b-o', ms=3, label='S_AF')
    ax.plot(wl_values, S_Stain_simple, 'r-s', ms=3, label='S_Stain (simple sub)')
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 0.2)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Normalized intensity')
    ax.set_title('Reference Spectra (S_AF and S_Stain)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(wl_values, ratio, 'k-o', ms=3)
    ax.axhline(1.0, color='green', linestyle='--', label='Perfect (1.0)')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Ratio (Recon / Neg)')
    ax.set_title('Recon AF / Neg ratio (OLS)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 3)
    
    ax = axes[1, 0]
    ax.plot(wl_values, median_neg, 'b-o', ms=3, label='Negative (median)')
    ax.plot(wl_values, median_recon, 'r-s', ms=3, label='Recon AF (median)')
    ax.set_yscale('log')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity')
    ax.set_title('Median Spectrum: Neg vs Recon AF')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.hist(c_stain_ols, bins=50, alpha=0.7, label='Calcein cells')
    ax.hist(C_neg_ols[:, 1], bins=50, alpha=0.7, label='Neg cells')
    ax.set_xlabel('c_stain')
    ax.set_ylabel('Count')
    ax.set_title('c_stain distribution')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('scratch/correct_unmixing_test.png', dpi=150)
    print("\nSaved correct_unmixing_test.png")

if __name__ == "__main__":
    main()
