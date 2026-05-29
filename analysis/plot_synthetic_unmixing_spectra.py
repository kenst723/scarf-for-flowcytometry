import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

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

def main():
    neg_csv = r"analysis\results\2026-05-27\negative_B01\B01 Well - B01_20260527_110441.csv"
    calcein_csv = r"analysis\results\2026-05-27\Calcein_A01\A01 Well - A01_20260527_110439.csv"
    output_png = r"analysis\results\2026-05-27\synthetic_unmixing_spectra_verification.png"
    
    print("Loading data...")
    df_neg = pd.read_csv(neg_csv)
    df_stain = pd.read_csv(calcein_csv)
    
    wl_features = [c for c in df_neg.columns if c.startswith('Area_') and c.endswith('nm') and '638.6nm' not in c]
    wavelengths = [float(c.replace('Area_', '').replace('nm', '')) for c in wl_features]
    
    X_neg_full = df_neg[wl_features].values
    X_stain_full = df_stain[wl_features].values
    
    print("Creating Synthetic Mixture...")
    np.random.seed(42)
    idx_neg = np.random.choice(len(X_neg_full), 2900, replace=False)
    idx_stain = np.random.choice(len(X_stain_full), 2000, replace=False)
    
    X_neg_mix = X_neg_full[idx_neg]
    X_stain_mix = X_stain_full[idx_stain]
    X_mix = np.vstack([X_neg_mix, X_stain_mix])
    
    # Calculate References from the mixture
    S_AF = np.median(X_neg_mix, axis=0) # We use the known negative part to get perfect S_AF
    S_AF = S_AF / np.sum(S_AF)
    
    total_intensity = np.sum(X_mix, axis=1)
    bright_cells = X_mix[total_intensity >= np.percentile(total_intensity, 95)]
    S_bright = np.median(bright_cells, axis=0)
    peak_idx = np.argmax(S_bright / (S_AF + 1e-9))
    
    stained_cells = X_mix[X_mix[:, peak_idx] >= np.percentile(X_mix[:, peak_idx], 95)]
    S_Stain = np.median(stained_cells, axis=0)
    S_Stain = np.maximum(S_Stain - (S_AF * np.min(S_Stain / (S_AF + 1e-9))), 0)
    S_Stain = S_Stain / np.sum(S_Stain)
    
    print("Unmixing Synthetic Mixture using WLSM...")
    S = np.column_stack((S_AF, S_Stain))
    C_mix = perform_wlsm(X_mix, S)
    
    # Select cells to plot
    # 1. A True Negative Cell (index 0)
    idx_neg_cell = 0
    X_cell_neg = X_mix[idx_neg_cell]
    C_cell_neg = C_mix[idx_neg_cell]
    pred_neg = C_cell_neg[0] * S_AF + C_cell_neg[1] * S_Stain
    
    # 2. A Highly Positive Cell (index 2900)
    idx_pos_cell = 2900
    X_cell_pos = X_mix[idx_pos_cell]
    C_cell_pos = C_mix[idx_pos_cell]
    pred_pos = C_cell_pos[0] * S_AF + C_cell_pos[1] * S_Stain
    
    print("Plotting...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), dpi=150)
    
    # Plot 1: References
    axes[0].plot(wavelengths, S_AF, color='gray', linewidth=2, label='Autofluorescence Ref')
    axes[0].plot(wavelengths, S_Stain, color='forestgreen', linewidth=2, label='Calcein Ref')
    axes[0].set_title('1. Computed Reference Spectra\n(from Synthetic Mixture)')
    axes[0].set_xlabel('Wavelength (nm)')
    axes[0].set_ylabel('Normalized Intensity')
    axes[0].legend()
    
    # Plot 2: Negative Cell Prediction
    axes[1].plot(wavelengths, X_cell_neg, color='black', linestyle=':', linewidth=2, label='Raw Data (Negative Cell)')
    axes[1].plot(wavelengths, pred_neg, color='red', linestyle='--', linewidth=2, label='Predicted (WLSM Fit)')
    axes[1].fill_between(wavelengths, 0, C_cell_neg[0] * S_AF, color='gray', alpha=0.3, label=f'Unmixed AF (Coef={C_cell_neg[0]:.0f})')
    # Use max to prevent filling below 0 for visualization
    axes[1].fill_between(wavelengths, 0, np.maximum(C_cell_neg[1] * S_Stain, 0), color='forestgreen', alpha=0.3, label=f'Unmixed Calcein (Coef={C_cell_neg[1]:.0f})')
    axes[1].set_title('2. Unmixing Result for a True Negative Cell')
    axes[1].set_xlabel('Wavelength (nm)')
    axes[1].set_ylabel('Intensity')
    axes[1].legend()
    
    # Plot 3: Positive Cell Prediction
    axes[2].plot(wavelengths, X_cell_pos, color='black', linestyle=':', linewidth=2, label='Raw Data (Calcein Cell)')
    axes[2].plot(wavelengths, pred_pos, color='red', linestyle='--', linewidth=2, label='Predicted (WLSM Fit)')
    axes[2].fill_between(wavelengths, 0, np.maximum(C_cell_pos[0] * S_AF, 0), color='gray', alpha=0.3, label=f'Unmixed AF (Coef={C_cell_pos[0]:.0f})')
    axes[2].fill_between(wavelengths, 0, np.maximum(C_cell_pos[1] * S_Stain, 0), color='forestgreen', alpha=0.3, label=f'Unmixed Calcein (Coef={C_cell_pos[1]:.0f})')
    axes[2].set_title('3. Unmixing Result for a Highly Calcein-Positive Cell')
    axes[2].set_xlabel('Wavelength (nm)')
    axes[2].set_ylabel('Intensity')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_png)
    print(f"Saved to {output_png}")

if __name__ == "__main__":
    main()
