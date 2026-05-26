"""
UMAP散布図 表示スクリプト

Usage:
    python analysis/show_umap_plot.py --experiment "Experiment 2026!05!21 15!59" --stain PI
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import EXPERIMENTS, RESULTS_DIR

def main():
    parser = argparse.ArgumentParser(description='UMAP結果をウインドウで表示します')
    parser.add_argument('--experiment', type=str, default="Experiment 2026!05!21 15!59",
                        help='実験フォルダ名')
    parser.add_argument('--stain', type=str, default="PI",
                        help='染色名')
    args = parser.parse_args()

    date_str = EXPERIMENTS.get(args.experiment, args.experiment)
    results_base_dir = os.path.join(RESULTS_DIR, date_str)
    
    # PNGファイルパス
    png_path = os.path.join(results_base_dir, f"autofluor_umap_{args.stain}.png")

    if not os.path.isfile(png_path):
        print(f"Error: UMAP PNG file not found at {png_path}. Please run UMAP generation first.")
        sys.exit(1)

    print(f"Loading and displaying {png_path}...")
    img = plt.imread(png_path)
    plt.figure(figsize=(12, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Autofluor UMAP - {args.stain}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    print("\nShowing plot window. Please close the window to end the script.")
    plt.show()

if __name__ == '__main__':
    main()
