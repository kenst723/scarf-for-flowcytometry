"""
Deep diagnostic: Why is c_stain over-estimated?
Check the per-cell behavior - look at a typical cell and see what IRLS does.
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

    wl_features = [c for c in df_neg.columns if c.startswith('Area_') and c.endswith('nm') and '638.6nm' not in c]
    wl_values = np.array([float(c.replace('Area_', '').replace('nm', '')) for c in wl_features])

    X_neg = df_neg[wl_features].values
    X_stain = df_stain[wl_features].values

    unmixer = PoissonUnmixer(max_iter=5)
    unmixer.fit(X_neg, X_stain)

    S_AF = unmixer.S_AF
    S_Stain = unmixer.S_Stain
    S = unmixer.S

    # Pick a median-brightness Calcein cell
    total = np.sum(X_stain, axis=1)
    median_idx = np.argsort(total)[len(total)//2]
    x = X_stain[median_idx]

    print(f"=== Median cell analysis (index {median_idx}) ===")
    print(f"Total intensity: {np.sum(x):.0f}")

    # OLS fit (no weights)
    c_ols = np.linalg.lstsq(S, x, rcond=None)[0]
    recon_ols = x - c_ols[1] * S_Stain

    # Sequential fit (current PoissonUnmixer approach)
    C_seq = unmixer.get_raw_coefficients(x[None, :])
    c_seq = C_seq[0]
    recon_seq = x - c_seq[1] * S_Stain

    # "Ideal" fit: use tail to determine c_af, then peak to determine c_stain
    # Use channels > 650nm (Calcein-free) to fix c_af
    tail_mask = wl_values > 650
    peak_mask = (wl_values >= 490) & (wl_values <= 560)
    
    # c_af from tail: minimize sum((x[tail] - c_af * S_AF[tail])^2)
    c_af_tail = np.dot(x[tail_mask], S_AF[tail_mask]) / (np.dot(S_AF[tail_mask], S_AF[tail_mask]) + 1e-9)
    # c_stain from peak: minimize sum((x[peak] - c_af_tail * S_AF[peak] - c_stain * S_Stain[peak])^2)
    residual_peak = x[peak_mask] - c_af_tail * S_AF[peak_mask]
    c_stain_peak = np.dot(residual_peak, S_Stain[peak_mask]) / (np.dot(S_Stain[peak_mask], S_Stain[peak_mask]) + 1e-9)
    recon_ideal = x - c_stain_peak * S_Stain

    print(f"\nOLS:   c_af={c_ols[0]:.0f}, c_stain={c_ols[1]:.0f}")
    print(f"Seq:   c_af={c_seq[0]:.0f}, c_stain={c_seq[1]:.0f}")
    print(f"Ideal: c_af={c_af_tail:.0f}, c_stain={c_stain_peak:.0f}")

    # Check residual at tail for each method
    median_neg = np.median(X_neg, axis=0)
    print(f"\nResidual ratio at tail (700nm, should be ~cell_size_factor):")
    tail_ch = np.argmin(np.abs(wl_values - 700))
    print(f"  OLS:   {recon_ols[tail_ch] / median_neg[tail_ch]:.2f}")
    print(f"  Seq:   {recon_seq[tail_ch] / median_neg[tail_ch]:.2f}")
    print(f"  Ideal: {recon_ideal[tail_ch] / median_neg[tail_ch]:.2f}")

    peak_ch = np.argmin(np.abs(wl_values - 515))
    print(f"\nResidual ratio at peak (515nm, should be ~cell_size_factor):")
    print(f"  OLS:   {recon_ols[peak_ch] / median_neg[peak_ch]:.2f}")
    print(f"  Seq:   {recon_seq[peak_ch] / median_neg[peak_ch]:.2f}")
    print(f"  Ideal: {recon_ideal[peak_ch] / median_neg[peak_ch]:.2f}")

    # Plot all three reconstructions
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    ax = axes[0]
    neg_scaled_ols = c_ols[0] * S_AF * np.sum(median_neg)
    neg_scaled_seq = c_seq[0] * S_AF * np.sum(median_neg)  
    ax.plot(wl_values, x, 'k-', lw=2, label=f'Observed (cell {median_idx})')
    ax.plot(wl_values, recon_ols, 'b--', label=f'Recon AF (OLS, c_stain={c_ols[1]:.0f})')
    ax.plot(wl_values, recon_seq, 'r--', label=f'Recon AF (Seq, c_stain={c_seq[1]:.0f})')
    ax.plot(wl_values, recon_ideal, 'g--', lw=2, label=f'Recon AF (Ideal, c_stain={c_stain_peak:.0f})')
    ax.set_yscale('log')
    ax.set_ylim(1, None)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity')
    ax.set_title('Single cell reconstruction comparison')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(wl_values, recon_ols / (median_neg + 1), 'b-o', ms=3, label='OLS')
    ax.plot(wl_values, recon_seq / (median_neg + 1), 'r-s', ms=3, label='Seq')
    ax.plot(wl_values, recon_ideal / (median_neg + 1), 'g-^', ms=3, label='Ideal (tail→AF, peak→Stain)')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Ratio to Neg median')
    ax.set_title('Residual ratio per method')
    ax.set_ylim(-1, 6)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('scratch/single_cell_diagnostic.png', dpi=150)
    print("\nSaved single_cell_diagnostic.png")

if __name__ == "__main__":
    main()
