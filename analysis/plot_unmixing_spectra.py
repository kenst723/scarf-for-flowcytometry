import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

def plot_unmixing_verification(neg_csv_path, stain_csv_path, stain_name="Calcein", output_png="unmixing_spectra.png"):
    print("Loading data...")
    df_neg = pd.read_csv(neg_csv_path)
    df_stain = pd.read_csv(stain_csv_path)
    
    # Extract wavelength features and numeric wavelengths
    wl_features = [
        c for c in df_neg.columns
        if c.startswith('Area_') and c.endswith('nm') and '638.6nm' not in c
    ]
    wavelengths = [float(c.replace('Area_', '').replace('nm', '')) for c in wl_features]
    
    X_neg = df_neg[wl_features].values
    X_stain = df_stain[wl_features].values
    
    # Calculate Reference Spectra
    print("Computing Reference Spectra...")
    S_AF = np.median(X_neg, axis=0)
    S_AF = S_AF / np.sum(S_AF)
    
    total_intensity = np.sum(X_stain, axis=1)
    bright_cells_total = X_stain[total_intensity >= np.percentile(total_intensity, 99)]
    S_bright_total = np.median(bright_cells_total, axis=0)
    ratios = S_bright_total / (S_AF + 1e-9)
    peak_idx = np.argmax(ratios)
    
    peak_values = X_stain[:, peak_idx]
    stained_cells = X_stain[peak_values >= np.percentile(peak_values, 98)]
    S_Stain_raw = np.median(stained_cells, axis=0)
    S_Stain = np.maximum(S_Stain_raw - (S_AF * np.min(S_Stain_raw / (S_AF + 1e-9))), 0)
    S_Stain = S_Stain / np.sum(S_Stain)
    
    # Find the brightest cell in the stained sample based on Unmixed_Calcein
    unmixed_stain = df_stain[f'Unmixed_{stain_name}'].values
    unmixed_af = df_stain['Unmixed_AF'].values
    
    brightest_idx = np.argmax(unmixed_stain)
    cell_raw = X_stain[brightest_idx]
    cell_af_comp = unmixed_af[brightest_idx] * S_AF
    cell_stain_comp = unmixed_stain[brightest_idx] * S_Stain
    cell_fit = cell_af_comp + cell_stain_comp
    
    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    
    # Plot 1: Reference Spectra
    axes[0].plot(wavelengths, S_AF, label='Autofluorescence (AF) Reference', color='gray', linewidth=2)
    axes[0].plot(wavelengths, S_Stain, label=f'{stain_name} Reference (AF Subtracted)', color='green', linewidth=2)
    axes[0].set_title("Computed Reference Spectra (Normalized)", fontsize=14)
    axes[0].set_xlabel("Wavelength (nm)", fontsize=12)
    axes[0].set_ylabel("Normalized Intensity", fontsize=12)
    axes[0].legend(fontsize=11)
    
    # Plot 2: Example Unmixing on Brightest Cell
    axes[1].plot(wavelengths, cell_raw, label='Raw Cell Spectrum', color='black', linestyle=':', linewidth=2, alpha=0.7)
    axes[1].plot(wavelengths, cell_af_comp, label=f'AF Component (Coef: {unmixed_af[brightest_idx]:.1f})', color='gray', linewidth=2, alpha=0.8)
    axes[1].plot(wavelengths, cell_stain_comp, label=f'{stain_name} Component (Coef: {unmixed_stain[brightest_idx]:.1f})', color='green', linewidth=2, alpha=0.8)
    axes[1].plot(wavelengths, cell_fit, label='Total Fit (AF + Stain)', color='red', linestyle='--', linewidth=2)
    
    axes[1].set_title(f"Unmixing Result for a Highly {stain_name}-Positive Cell", fontsize=14)
    axes[1].set_xlabel("Wavelength (nm)", fontsize=12)
    axes[1].set_ylabel("Fluorescence Intensity", fontsize=12)
    axes[1].legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    print(f"Saved plot to {output_png}")

if __name__ == '__main__':
    neg_csv = r"analysis\results\2026-05-27\negative_B01\B01 Well - B01_20260527_110441.csv"
    calcein_csv = r"analysis\results\2026-05-27\Calcein_A01\A01 Well - A01_20260527_110439.csv"
    out_png = r"analysis\results\2026-05-27\unmixing_spectra_verification.png"
    
    plot_unmixing_verification(neg_csv, calcein_csv, stain_name="Calcein", output_png=out_png)
