import os
import glob
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import StandardScaler

def main(experiment_dir):
    print(f"Loading data from: {experiment_dir}")
    calcein_csvs = glob.glob(os.path.join(experiment_dir, "Calcein_*", "*.csv"))
    
    if not calcein_csvs:
        print("Error: Could not find Calcein CSV files.")
        return
        
    df = pd.read_csv(calcein_csvs[0])
    print(f"Loaded {len(df)} cells from {os.path.basename(calcein_csvs[0])}")
    
    # 散乱光の代替となると思われるV1, V2を抽出 (FSC, SSCの可能性が高い)
    scatter_cols = ['Area_V1', 'Height_V1', 'Area_V2', 'Height_V2']
    
    for col in scatter_cols:
        if col not in df.columns:
            print(f"Error: Required column {col} not found in the CSV.")
            return
            
    X_scatter = df[scatter_cols].values
    
    # 標準化 (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_scatter)
    
    print("Running UMAP on Scatter parameters (Area_V1, Height_V1, Area_V2, Height_V2)...")
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)
    umap_coords = reducer.fit_transform(X_scaled)
    
    # 色付け用: Unmixed_Calcein (ArcSinh スケール)
    if 'Unmixed_Calcein' not in df.columns:
        print("Error: 'Unmixed_Calcein' not found.")
        return
        
    c_stain = df['Unmixed_Calcein'].values
    c_stain_arcsinh = np.arcsinh(c_stain / 150.0)
    
    print("Generating plot...")
    plt.figure(figsize=(10, 8), dpi=150)
    
    # 99パーセンタイルでカラーバーの上限をクリップ
    vmax = np.percentile(c_stain_arcsinh, 99)
    vmin = np.percentile(c_stain_arcsinh, 1)
    
    sc = plt.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                     c=c_stain_arcsinh, cmap='inferno', s=2, alpha=0.7,
                     vmin=vmin, vmax=vmax)
    
    plt.colorbar(sc, label='Unmixed Calcein (ArcSinh)')
    plt.title('UMAP based purely on Scatter (V1/V2 Area & Height)\nColored by Calcein Intensity', fontsize=14)
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    
    out_png = os.path.join(experiment_dir, "umap_scatter_calcein.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    
    print(f"Plot saved to: {out_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UMAP on Scatter Parameters")
    parser.add_argument("--dir", type=str, required=True, help="Path to experiment results dir")
    args = parser.parse_args()
    
    main(args.dir)
