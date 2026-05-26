"""
解析パイプライン — 一括実行エントリーポイント

指定した実験データに対して以下のステップを順番に実行する:
1. .sraw → CSV 変換
2. スペクトル密度プロット生成
3. 蛍光強度ヒストグラム生成（対応する .fcs ファイルがある場合）
4. UMAP 次元圧縮解析（対応する .fcs ファイルがある場合）

Usage:
    python run_pipeline.py --experiment "Experiment 2026!05!21 15!59" --rack "24 Tube Rack (5mL) - 1" --stain PI
"""

import os
import sys
import argparse

# Add project root to sys.path to allow importing 'config' and 'src'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
from config import get_experiment_data_dir, get_results_dir, find_sraw_files, EXPERIMENTS
from src.convert import convert_sraw_to_csv
from src.plot_spectral import plot_spectral_density
from src.plot_histogram import plot_histogram
from src.run_umap_autofluor import run_umap_autofluor


def run_pipeline(experiment_folder, rack_name, stain_name):
    """
    指定実験のパイプラインを実行する。

    Parameters
    ----------
    experiment_folder : str
        実験フォルダ名 (例: "Experiment 2026!05!21 15!59")
    rack_name : str
        ラック名 (例: "24 Tube Rack (5mL) - 1")
    stain_name : str
        染色名 (例: "PI", "Calcein", "negative")
    """
    # .sraw ディレクトリの特定
    sraw_dir = os.path.join(
        get_experiment_data_dir(experiment_folder),
        rack_name,
        stain_name,
    )

    print(f"=" * 70)
    print(f"Pipeline: {experiment_folder}")
    print(f"  Rack:  {rack_name}")
    print(f"  Stain: {stain_name}")
    print(f"  Dir:   {sraw_dir}")
    print(f"=" * 70)

    # .sraw ファイルを検索
    sraw_files = find_sraw_files(sraw_dir)
    print(f"\nFound {len(sraw_files)} .sraw file(s)\n")

    for filepath in sraw_files:
        filename = os.path.basename(filepath)
        base_name = os.path.splitext(filename)[0]

        # ウェル名を抽出 (例: "A01 Well - A01" → "A01")
        well_id = base_name.split(' ')[0] if ' ' in base_name else base_name
        sample_label = f"{stain_name}_{well_id}"

        # 結果ディレクトリを取得
        result_dir = get_results_dir(experiment_folder, sample_label)

        # 対応する .fcs ファイルを探す
        fcs_path = os.path.join(sraw_dir, base_name + '.fcs')
        has_fcs = os.path.isfile(fcs_path)
        total_steps = 4 if has_fcs else 2

        # --- Step 1: .sraw → CSV ---
        print(f"[1/{total_steps}] Converting {filename} ...")
        csv_path, df = convert_sraw_to_csv(filepath, output_dir=result_dir)
        print(f"      -> {csv_path}  (shape: {df.shape})")

        # --- Step 2: Spectral density plot ---
        print(f"[2/{total_steps}] Generating spectral density plot ...")
        plot_path = os.path.join(result_dir, 'spectral_density.png')
        plot_spectral_density(csv_path, plot_path)

        # --- Step 3: Fluorescence intensity histogram ---
        if has_fcs:
            print(f"[3/{total_steps}] Generating fluorescence histogram ...")
            hist_path = os.path.join(result_dir, 'histogram.png')
            plot_histogram(fcs_path, hist_path, stain_name=stain_name)
        else:
            print(f"  (Histogram skipped — .fcs file not found: {fcs_path})")

        # UMAP is now handled collectively at the end of the pipeline
        print()

    # --- Group-level Autofluor UMAP ---
    if stain_name.lower() != 'negative':
        neg_dir = os.path.join(get_experiment_data_dir(experiment_folder), rack_name, "Negative")
        if os.path.isdir(neg_dir):
            print(f"\n{'=' * 70}")
            print(f"Generating Autofluor UMAP projection for '{stain_name}' group...")
            
            date_str = EXPERIMENTS.get(experiment_folder, experiment_folder)
            results_base = os.path.join(config.RESULTS_DIR, date_str)
            os.makedirs(results_base, exist_ok=True)
            output_path = os.path.join(results_base, f"autofluor_umap_{stain_name}.html")
            
            run_umap_autofluor(neg_dir, sraw_dir, output_path, stain_name=stain_name)
        else:
            print(f"\nSkipping Autofluor UMAP: Negative directory not found at {neg_dir}")

    print("Pipeline complete!")


def main():
    parser = argparse.ArgumentParser(description='解析パイプライン一括実行')
    parser.add_argument('--experiment', type=str, required=True,
                        help='実験フォルダ名 (例: "Experiment 2026!05!21 15!59")')
    parser.add_argument('--rack', type=str, required=True,
                        help='ラック名 (例: "24 Tube Rack (5mL) - 1")')
    parser.add_argument('--stain', type=str, required=True,
                        help='染色名 (例: PI, Calcein, negative)')
    args = parser.parse_args()

    run_pipeline(args.experiment, args.rack, args.stain)


if __name__ == '__main__':
    main()
