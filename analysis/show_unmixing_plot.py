"""
アンミキシング（自家蛍光分離）散布図 表示スクリプト

Usage:
    python analysis/show_unmixing_plot.py --experiment "Experiment 2026!05!21 15!59" --rack "24 Tube Rack (5mL) - 1" --stain PI
"""

import os
import sys
import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import EXPERIMENTS, RESULTS_DIR

def main():
    parser = argparse.ArgumentParser(description='アンミキシング結果をウインドウで表示します')
    parser.add_argument('--experiment', type=str, default="Experiment 2026!05!21 15!59",
                        help='実験フォルダ名')
    parser.add_argument('--rack', type=str, default="24 Tube Rack (5mL) - 1",
                        help='ラック名')
    parser.add_argument('--stain', type=str, default="PI",
                        help='染色名')
    args = parser.parse_args()

    date_str = EXPERIMENTS.get(args.experiment, args.experiment)
    results_base_dir = os.path.join(RESULTS_DIR, date_str)
    
    # 染色サンプルの結果CSVを探す
    stain_csv_paths = glob.glob(os.path.join(results_base_dir, f"{args.stain}_*", "*.csv"))
    stain_csv_paths = [p for p in stain_csv_paths if "scarf_embeddings" not in p]

    if not stain_csv_paths:
        print(f"Error: No unmixed CSV files found for {args.stain} in {results_base_dir}")
        sys.exit(1)

    # 最も新しいCSVを使用
    csv_path = max(stain_csv_paths, key=os.path.getmtime)
    print(f"Loading {csv_path}...")

    df = pd.read_csv(csv_path)
    if 'Unmixed_AF' not in df.columns or f'Unmixed_{args.stain}' not in df.columns:
        print(f"Error: Unmixed columns not found in {csv_path}. Please run the pipeline first.")
        sys.exit(1)

    af_vals = df['Unmixed_AF'].values
    stain_vals = df[f'Unmixed_{args.stain}'].values

    # ArcSinh変換してプロット (cofactor=150)
    af_plot = np.arcsinh(af_vals / 150)
    stain_plot = np.arcsinh(stain_vals / 150)

    # 散布図の表示
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(af_plot, stain_plot, s=2, alpha=0.3, c=stain_plot, cmap='jet', edgecolors='none')
    plt.colorbar(sc, label=f'Unmixed {args.stain} (ArcSinh)')
    plt.xlabel("Unmixed Autofluorescence (ArcSinh)")
    plt.ylabel(f"Unmixed {args.stain} (ArcSinh)")
    plt.title(f"Spectral Unmixing Result\n{os.path.basename(csv_path)}")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    print("\nShowing plot window. Please close the window to end the script.")
    plt.show()

if __name__ == '__main__':
    main()
