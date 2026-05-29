import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.unmix_spectral import PoissonUnmixer

def main():
    neg_csv = r"analysis\results\2026-05-27\negative_B01\B01 Well - B01_20260529_134834.csv"
    calcein_csv = r"analysis\results\2026-05-27\Calcein_A01\A01 Well - A01_20260529_134433.csv"
    output_png = r"analysis\results\2026-05-27\original_density_verification.png"
    
    print("Loading Original Data...")
    df_neg = pd.read_csv(neg_csv)
    df_stain = pd.read_csv(calcein_csv)
    
    wl_features = [c for c in df_neg.columns if c.startswith('Area_') and c.endswith('nm') and '638.6nm' not in c]
    X_neg = df_neg[wl_features].values
    X_stain = df_stain[wl_features].values
    
    print("Calculating References from Original Data...")
    unmixer = PoissonUnmixer().fit(X_neg, X_stain)
    
    print("Unmixing Original Calcein Data...")
    C_af, C_calcein = unmixer.transform(X_stain)
    
    C_af_arcsinh = np.arcsinh(C_af / 150.0)
    C_calcein_arcsinh = np.arcsinh(C_calcein / 150.0)
    
    print("Plotting Original Density...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.figure(figsize=(8, 6), dpi=150)
    
    sns.kdeplot(x=C_af_arcsinh, y=C_calcein_arcsinh, cmap="viridis", fill=True, bw_adjust=0.5, levels=30, thresh=0.01)
    plt.scatter(C_af_arcsinh, C_calcein_arcsinh, color='black', s=1, alpha=0.1)
    
    plt.title('Original Calcein Data Density Plot (IRLS Poisson MLE)', fontsize=14, fontweight='bold')
    plt.xlabel('Unmixed AF (ArcSinh)')
    plt.ylabel('Unmixed Calcein (ArcSinh)')
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_png)
    print(f"Saved to {output_png}")

if __name__ == "__main__":
    main()
