import os
import sys
import glob
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
    
    # Current method
    median_neg = np.median(X_neg, axis=0)
    peak_idx = 4
    peak_values = X_stain[:, peak_idx]
    bright_calcein = X_stain[peak_values >= np.percentile(peak_values, 98)]
    median_bright_calcein = np.median(bright_calcein, axis=0)
    
    S_Stain_old = median_bright_calcein - median_neg
    S_Stain_old = np.maximum(S_Stain_old, 0)
    S_Stain_old /= np.sum(S_Stain_old)
    
    # Proportional method
    # Use channels > 680nm for AF scaling
    tail_mask = wl_values > 680
    af_scale = np.sum(median_bright_calcein[tail_mask]) / np.sum(median_neg[tail_mask])
    print(f"AF Scale factor for bright cells: {af_scale:.2f}")
    
    S_Stain_new = median_bright_calcein - af_scale * median_neg
    S_Stain_new = np.maximum(S_Stain_new, 0)
    S_Stain_new /= np.sum(S_Stain_new)
    
    plt.figure(figsize=(10, 5))
    plt.plot(wl_values, S_Stain_old, label='Old (Contaminated with AF)')
    plt.plot(wl_values, S_Stain_new, label='New (Purified)')
    plt.yscale('log')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Normalized Intensity')
    plt.legend()
    plt.grid(True)
    plt.savefig('scratch/stain_reference_comparison.png')
    
if __name__ == "__main__":
    main()
