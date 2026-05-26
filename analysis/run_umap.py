"""
UMAP次元圧縮解析 単体実行スクリプト

Usage:
    python analysis/run_umap.py --experiment "Experiment 2026!05!21 15!59" --rack "24 Tube Rack (5mL) - 1" --stain PI
"""

import os
import sys
import argparse

# プロジェクトルートを追加
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import get_experiment_data_dir, EXPERIMENTS, RESULTS_DIR
from src.run_umap_autofluor import run_umap_autofluor

def main():
    parser = argparse.ArgumentParser(description='UMAP次元圧縮のみを単体実行')
    parser.add_argument('--experiment', type=str, required=True,
                        help='実験フォルダ名 (例: "Experiment 2026!05!21 15!59")')
    parser.add_argument('--rack', type=str, required=True,
                        help='ラック名 (例: "24 Tube Rack (5mL) - 1")')
    parser.add_argument('--stain', type=str, required=True,
                        help='染色名 (例: PI, Calcein)')
    args = parser.parse_args()

    sraw_dir = os.path.join(
        get_experiment_data_dir(args.experiment),
        args.rack,
        args.stain,
    )
    neg_dir = os.path.join(
        get_experiment_data_dir(args.experiment),
        args.rack,
        "Negative",
    )

    # ディレクトリ存在確認
    if not os.path.isdir(sraw_dir):
        print(f"Error: Stained directory not found: {sraw_dir}")
        sys.exit(1)
    if not os.path.isdir(neg_dir):
        print(f"Error: Negative directory not found: {neg_dir}")
        sys.exit(1)

    date_str = EXPERIMENTS.get(args.experiment, args.experiment)
    results_base_dir = os.path.join(RESULTS_DIR, date_str)
    output_path = os.path.join(results_base_dir, f"autofluor_umap_{args.stain}.html")
    png_path = os.path.join(results_base_dir, f"autofluor_umap_{args.stain}.png")

    print(f"============================================================")
    print(f"Running standalone Autofluor UMAP for {args.stain}...")
    print(f"  Negative dir: {neg_dir}")
    print(f"  Stained dir:  {sraw_dir}")
    print(f"  Output HTML:  {output_path}")
    print(f"  Output PNG:   {png_path}")
    print(f"============================================================")

    try:
        run_umap_autofluor(
            neg_dir=neg_dir,
            stain_dir=sraw_dir,
            output_path=output_path,
            stain_name=args.stain,
            png_output_path=png_path
        )
        print("\nUMAP generation complete!")
    except Exception as e:
        print(f"\nError: UMAP generation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
