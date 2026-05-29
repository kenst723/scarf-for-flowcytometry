import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.unmix_spectral import PoissonUnmixer

def plot_unmixing_verification(neg_csv_path, stain_csv_path, stain_name="Calcein", output_png="unmixing_spectra.png"):
    print("Loading data...")
    df_neg = pd.read_csv(neg_csv_path)
    df_stain = pd.read_csv(stain_csv_path)
    
    wl_features = [c for c in df_neg.columns if c.startswith('Area_') and c.endswith('nm') and '638.6nm' not in c]
    wavelengths = [float(c.replace('Area_', '').replace('nm', '')) for c in wl_features]
    
    X_neg = df_neg[wl_features].values
    X_stain = df_stain[wl_features].values
    
    print("Computing Reference Spectra...")
    unmixer = PoissonUnmixer().fit(X_neg, X_stain)
    S_AF = unmixer.S_AF
    S_Stain = unmixer.S_Stain
    
    unmixed_af, unmixed_stain = unmixer.transform(X_stain)
    
    brightest_idx = np.argmax(unmixed_stain)
    X_cell = X_stain[brightest_idx]
    
    C_cell = unmixer.get_raw_coefficients(X_cell[None, :])[0]
    c_af = C_cell[0]
    c_stain = C_cell[1]
    pred = c_af * S_AF + c_stain * S_Stain
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.figure(figsize=(10, 6), dpi=150)
    
    plt.plot(wavelengths, X_cell, color='black', linestyle=':', linewidth=2, label='Raw Spectral Data (Brightest Cell)')
    plt.plot(wavelengths, pred, color='red', linestyle='--', linewidth=2, label='Predicted (IRLS Fit)')
    
    plt.fill_between(wavelengths, 0, np.maximum(c_af * S_AF, 0), color='gray', alpha=0.3, label=f'Unmixed AF (Coef={c_af:.0f})')
    plt.fill_between(wavelengths, 0, np.maximum(c_stain * S_Stain, 0), color='forestgreen', alpha=0.3, label=f'Unmixed {stain_name} (Coef={c_stain:.0f})')
    
    plt.title(f'Spectral Unmixing Verification (IRLS Poisson MLE)\nBrightest {stain_name} Cell', fontsize=14, fontweight='bold')
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Intensity', fontsize=12)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_png)
    print(f"Saved plot to {output_png}")

if __name__ == '__main__':
    neg_csv = r"analysis\results\2026-05-27\negative_B01\B01 Well - B01_20260529_134834.csv"
    calcein_csv = r"analysis\results\2026-05-27\Calcein_A01\A01 Well - A01_20260529_134433.csv"
    out_png = r"analysis\results\2026-05-27\unmixing_spectra_verification.png"
    plot_unmixing_verification(neg_csv, calcein_csv, stain_name="Calcein", output_png=out_png)
