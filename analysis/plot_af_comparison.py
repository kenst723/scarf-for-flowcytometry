import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.unmix_spectral import PoissonUnmixer

def plot_af_comparison(neg_csv_path, stain_csv_path, stain_name="Calcein", output_png="af_comparison.png"):
    print("Loading data...")
    df_neg = pd.read_csv(neg_csv_path)
    df_stain = pd.read_csv(stain_csv_path)
    
    wl_features = [c for c in df_neg.columns if c.startswith('Area_') and c.endswith('nm') and '638.6nm' not in c]
    wavelengths = [float(c.replace('Area_', '').replace('nm', '')) for c in wl_features]
    
    X_neg = df_neg[wl_features].values
    X_stain = df_stain[wl_features].values
    
    print("Performing IRLS Poisson Unmixing...")
    unmixer = PoissonUnmixer().fit(X_neg, X_stain)
    
    X_stain_unmixed_af = unmixer.remove_stain_component(X_stain)
    
    print("Generating plot...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    
    for x in X_neg[::5]:
        axes[0].plot(wavelengths, x, color='gray', alpha=0.05)
    axes[0].plot(wavelengths, np.median(X_neg, axis=0), color='black', linewidth=2, label='Median AF')
    axes[0].set_title('1. Raw Negative Control Spectra', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Wavelength (nm)')
    axes[0].set_ylabel('Intensity')
    
    for x in X_stain_unmixed_af[::5]:
        axes[1].plot(wavelengths, x, color='purple', alpha=0.05)
    axes[1].plot(wavelengths, np.median(X_stain_unmixed_af, axis=0), color='indigo', linewidth=2, label='Median Reconstructed AF')
    axes[1].set_title(f'2. Reconstructed AF Spectra\n(From {stain_name} Stained Sample)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Wavelength (nm)')
    
    axes[2].plot(wavelengths, np.median(X_neg, axis=0), color='black', linewidth=2, label='True Negative AF')
    axes[2].plot(wavelengths, np.median(X_stain_unmixed_af, axis=0), color='indigo', linestyle='--', linewidth=2, label=f'Reconstructed AF (from {stain_name})')
    axes[2].set_title('3. Median Comparison', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Wavelength (nm)')
    axes[2].legend()
    
    for ax in axes:
        ax.set_ylim(-500, np.percentile(X_neg, 99.5))
        
    plt.tight_layout()
    plt.savefig(output_png)
    print(f"Saved plot to {output_png}")

if __name__ == '__main__':
    neg_csv = r"analysis\results\2026-05-27\negative_B01\B01 Well - B01_20260529_134834.csv"
    calcein_csv = r"analysis\results\2026-05-27\Calcein_A01\A01 Well - A01_20260529_134433.csv"
    out_png = r"analysis\results\2026-05-27\af_comparison_spectra.png"
    plot_af_comparison(neg_csv, calcein_csv, stain_name="Calcein", output_png=out_png)
