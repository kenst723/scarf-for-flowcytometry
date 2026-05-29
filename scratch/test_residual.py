import os
import sys
import glob
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
    X_neg = df_neg[wl_features].values
    X_stain = df_stain[wl_features].values
    
    unmixer = PoissonUnmixer(max_iter=5)
    unmixer.fit(X_neg, X_stain)
    
    C = unmixer._unmix_irls(X_stain)
    c_calcein = C[:, 1]
    
    peak_idx = np.argmax(unmixer.S_Stain)
    
    residuals = X_stain[:, peak_idx] - c_calcein * unmixer.S_Stain[peak_idx]
    
    print(f"Residual at peak:")
    print(f"  Mean: {np.mean(residuals):.1f}")
    print(f"  Median: {np.median(residuals):.1f}")
    print(f"  Max: {np.max(residuals):.1f}")
    print(f"  Min: {np.min(residuals):.1f}")
    print(f"  Std: {np.std(residuals):.1f}")
    
    # How many residuals are greater than 10,000?
    high_residuals = np.sum(residuals > 10000)
    print(f"Cells with residual > 10,000: {high_residuals} / {len(residuals)}")
    
    # Are the high residuals correlated with high c_calcein?
    plt.scatter(c_calcein, residuals, alpha=0.5, s=2)
    plt.xlabel('c_calcein')
    plt.ylabel('Residual at peak')
    plt.savefig('scratch/residual_scatter.png')

if __name__ == "__main__":
    main()
