import os
import sys
import glob
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.unmix_spectral import PoissonUnmixer

def test_unmixing():
    neg_dir = r"analysis\results\Experiment 2026!05!27 9!30\negative_B01"
    calcein_dir = r"analysis\results\Experiment 2026!05!27 9!30\Calcein_A01"
    
    neg_csv = glob.glob(os.path.join(neg_dir, "*.csv"))[0]
    calcein_csv = glob.glob(os.path.join(calcein_dir, "*.csv"))[0]
    
    df_neg = pd.read_csv(neg_csv)
    df_stain = pd.read_csv(calcein_csv)
    
    wl_features = [c for c in df_neg.columns if c.startswith('Area_') and c.endswith('nm') and '638.6nm' not in c]
    X_neg = df_neg[wl_features].values
    X_stain = df_stain[wl_features].values
    
    unmixer = PoissonUnmixer(max_iter=10)
    unmixer.fit(X_neg, X_stain)
    
    bright_idx = np.argmax(np.sum(X_stain, axis=1))
    x_bright = X_stain[bright_idx]
    
    S = unmixer.S
    peak_idx = np.argmax(unmixer.S_Stain)
    
    def fit_with_weights(W):
        M_mat = S.T @ np.diag(W) @ S
        B_vec = S.T @ np.diag(W) @ x_bright
        return np.linalg.solve(M_mat, B_vec)

    print(f"X at peak: {x_bright[peak_idx]}")
    
    for thresh in [1, 10, 100, 1000, 5000, 10000]:
        c = fit_with_weights(1.0 / np.maximum(x_bright, thresh))
        res = x_bright[peak_idx] - c[1] * unmixer.S_Stain[peak_idx]
        print(f"Threshold {thresh:<5} -> Residual at peak: {res:8.1f} | c_calcein: {c[1]:10.1f}")
        
if __name__ == "__main__":
    test_unmixing()
