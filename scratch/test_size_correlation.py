import os
import sys
import glob
import numpy as np
import pandas as pd

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
    X_neg = df_neg[wl_features].values
    X_stain = df_stain[wl_features].values
    
    peak_idx = 4 # roughly 515nm
    
    # Median negative
    median_neg = np.median(X_neg, axis=0)
    print(f"Median Negative Total Intensity: {np.sum(median_neg):.1f}")
    
    # Brightest Calcein cells (top 2%)
    peak_values = X_stain[:, peak_idx]
    bright_calcein = X_stain[peak_values >= np.percentile(peak_values, 98)]
    median_bright_calcein = np.median(bright_calcein, axis=0)
    
    print(f"Median Bright Calcein Total Intensity: {np.sum(median_bright_calcein):.1f}")
    
    # Let's look at a channel where Calcein does NOT emit, e.g., 750nm (index around 30)
    tail_idx = 30
    print(f"\nIntensity at channel {tail_idx} (pure AF region):")
    print(f"  Median Negative cell: {median_neg[tail_idx]:.1f}")
    print(f"  Median Bright Calcein cell: {median_bright_calcein[tail_idx]:.1f}")
    
if __name__ == "__main__":
    main()
