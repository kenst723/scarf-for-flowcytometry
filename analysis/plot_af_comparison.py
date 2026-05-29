import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

def plot_af_comparison(neg_csv_path, stain_csv_path, stain_name="Calcein", output_png="af_comparison.png"):
    print("Loading data...")
    df_neg = pd.read_csv(neg_csv_path)
    df_stain = pd.read_csv(stain_csv_path)
    
    # Extract wavelength features
    wl_features = [
        c for c in df_neg.columns
        if c.startswith('Area_') and c.endswith('nm') and '638.6nm' not in c
    ]
    wavelengths = [float(c.replace('Area_', '').replace('nm', '')) for c in wl_features]
    
    X_neg = df_neg[wl_features].values
    X_stain = df_stain[wl_features].values
    
    # Calculate Reference Spectra
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
    
    # Reconstruct AF for stained sample using Weighted Least Squares (WLSM)
    # WLSM properly accounts for Poisson noise by weighting dim channels more
    def perform_wlsm(X, S, epsilon=1.0):
        N, M = X.shape
        num_fluor = S.shape[1]
        C = np.zeros((N, num_fluor))
        for i in range(N):
            w = 1.0 / np.maximum(X[i], epsilon)
            St_W = S.T * w  # Broadcasting multiplication is faster than np.diag
            try:
                inv = np.linalg.inv(St_W @ S)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(St_W @ S)
            C[i] = inv @ St_W @ X[i]
        return C

    print("Performing WLSM Unmixing...")
    S = np.column_stack((S_AF, S_Stain))
    C_stain = perform_wlsm(X_stain, S)
    raw_calcein_coef = C_stain[:, 1]
    
    # Subtract Calcein component. DO NOT use np.maximum(..., 0) to allow natural noise distribution around 0.
    X_unmixed_af = X_stain - raw_calcein_coef[:, None] * S_Stain[None, :]
    
    # Calculate Medians
    median_neg = np.median(X_neg, axis=0)
    median_unmixed_af = np.median(X_unmixed_af, axis=0)
    
    # Sample 100 random cells for plotting lines
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_neg), size=min(100, len(X_neg)), replace=False)
    X_neg_sample = X_neg[sample_idx]
    X_unmixed_af_sample = X_unmixed_af[sample_idx]
    
    # Helper to apply ArcSinh transformation for plotting
    from config import COFACTOR
    def arcsinh_scale(x):
        return np.arcsinh(x / COFACTOR)

    # Plotting
    print("Generating plot...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=150)
    
    # Subplot 1: Raw Negative Spectra
    for row in X_neg_sample:
        axes[0].plot(wavelengths, arcsinh_scale(row), color='gray', alpha=0.1)
    axes[0].plot(wavelengths, arcsinh_scale(median_neg), color='black', linewidth=2, label='Median Negative')
    axes[0].set_title("1. Raw Negative Cells (Autofluorescence)")
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel(f"ArcSinh Intensity (Cofactor={COFACTOR})")
    axes[0].legend()
    
    # Subplot 2: Unmixed AF Spectra from Stained
    for row in X_unmixed_af_sample:
        axes[1].plot(wavelengths, arcsinh_scale(row), color='lightcoral', alpha=0.1)
    axes[1].plot(wavelengths, arcsinh_scale(median_unmixed_af), color='red', linewidth=2, label='Median Unmixed AF')
    axes[1].set_title(f"2. Unmixed AF (from {stain_name} Sample, WLSM)")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].legend()
    
    # Subplot 3: Median Overlay
    axes[2].plot(wavelengths, arcsinh_scale(median_neg), color='black', linewidth=2, label='Median Negative')
    axes[2].plot(wavelengths, arcsinh_scale(median_unmixed_af), color='red', linestyle='--', linewidth=2, label='Median Unmixed AF')
    axes[2].set_title("3. Comparison of Medians (ArcSinh Scale)")
    axes[2].set_xlabel("Wavelength (nm)")
    axes[2].legend()
    
    # Match Y-axis limits for fair comparison
    all_y = np.concatenate([arcsinh_scale(X_neg_sample.flatten()), arcsinh_scale(X_unmixed_af_sample.flatten())])
    min_y, max_y = np.percentile(all_y, [0.1, 99.9])
    axes[0].set_ylim(min_y, max_y)
    axes[1].set_ylim(min_y, max_y)
    
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    print(f"Saved plot to {output_png}")

if __name__ == '__main__':
    neg_csv = r"analysis\results\2026-05-27\negative_B01\B01 Well - B01_20260527_110441.csv"
    calcein_csv = r"analysis\results\2026-05-27\Calcein_A01\A01 Well - A01_20260527_110439.csv"
    out_png = r"analysis\results\2026-05-27\af_comparison_spectra.png"
    
    plot_af_comparison(neg_csv, calcein_csv, stain_name="Calcein", output_png=out_png)
