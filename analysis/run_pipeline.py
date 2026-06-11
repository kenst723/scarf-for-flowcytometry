import os
import sys
import glob
import argparse

# Add the project root to the path so we can import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EXPERIMENTS, get_experiment_data_dir, get_results_dir, find_sraw_files, RESULTS_DIR
from src.convert import convert_sraw_to_csv
from src.plot_spectral import plot_spectral_density
from src.plot_histogram import plot_histogram
from src.run_umap_autofluor import run_umap_autofluor
from src.unmix_spectral import run_unmixing_group, get_spectral_features
from src.plot_unmixing_comparison import plot_unmixing_comparison, find_csv_in_dir


"""
python analysis/run_pipeline.py   --experiment 'Experiment 2026!06!02 12!39'   --rack '24 Tube Rack (5mL) - 1'  --method ols
"""


def generate_raw_csv_umap(results_base_dir, stain_name, output_path, stain_dir=None):
    """
    指定された Stain のみの全CSV (Unmixed_*, Time Stamp 以外) を結合し、
    ArcSinh変換 → StandardScaler → 1枚の UMAP として投影する。
    """
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from scipy.signal import savgol_filter
    
    try:
        import cuml.manifold.umap as cuml_umap
        def _umap_fit_transform(X, n_neighbors=15, min_dist=0.1, seed=42):
            reducer = cuml_umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                                     min_dist=min_dist, random_state=seed, metric='euclidean')
            return reducer.fit_transform(X)
        print("  [Raw CSV UMAP] Using GPU cuML UMAP")
    except ImportError:
        import umap
        def _umap_fit_transform(X, n_neighbors=15, min_dist=0.1, seed=42):
            reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                                min_dist=min_dist, n_jobs=-1, low_memory=False)
            return reducer.fit_transform(X)
        print("  [Raw CSV UMAP] Using CPU UMAP")

    cofactor = 150.0
    window_length, polyorder = 7, 2

    # --- Stain CSVs ---
    all_stain_csvs = sorted([
        p for p in glob.glob(os.path.join(results_base_dir, f"{stain_name}_*", "*_wavelength.csv"))
        if "scarf_embeddings" not in p
    ])

    import re
    latest_stains = {}
    for p in all_stain_csvs:
        csv_basename = os.path.basename(p)
        m = re.match(r"(.*)_\d{8}_\d{6}_wavelength\.csv", csv_basename)
        orig_name = m.group(1) if m else os.path.splitext(csv_basename)[0]
        latest_stains[orig_name] = p
        
    stain_csvs = list(latest_stains.values())

    if not stain_csvs:
        print(f"  [Raw CSV UMAP] Warning: CSV files for {stain_name} not found. Skipping.")
        return

    def load_all_features(paths):
        frames = []
        for p in paths:
            df = pd.read_csv(p)
            # Exclude non-biological or post-processed columns
            exclude_cols = ['Time Stamp', 'event_id']
            exclude_cols += [c for c in df.columns if c.startswith('Unmixed_')]
            
            # Select all remaining numeric columns (FCS features + Spectral features)
            feature_cols = [c for c in df.columns if c not in exclude_cols]
            frames.append(df[feature_cols].values)
        return np.vstack(frames) if frames else None

    X_stain = load_all_features(stain_csvs)
    if X_stain is None:
        print("  [Raw CSV UMAP] Warning: No features found. Skipping.")
        return

    print(f"  [Raw CSV UMAP] {stain_name}: {len(X_stain)} cells")

    # --- Preprocessing ---
    # Apply ArcSinh to all features (Standard for flow cytometry data)
    def preprocess(X):
        # Prevent negative values from causing issues before arcsinh
        X_safe = np.maximum(X, 0)
        return np.arcsinh(X_safe / cofactor)

    X_stain_arc = preprocess(X_stain)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_stain_arc)

    # Stain intensityのラベル（FCSファイルから直接取得して色付け用に使用）
    stain_intensity = None
    stain_label = f'{stain_name} (ArcSinh)'
    import fcsparser
    import re
    
    for p in stain_csvs:
        # FCSファイルを探す
        csv_basename = os.path.basename(p)
        # タイムスタンプ部分(_YYYYMMDD_HHMMSS_wavelength)を取り除いて元のFCSベースネームを取得
        m = re.match(r"(.*)_\d{8}_\d{6}_wavelength\.csv", csv_basename)
        orig_name = m.group(1) if m else os.path.splitext(csv_basename)[0]
        
        # FCSのパス
        fcs_path = os.path.join(stain_dir, orig_name + ".fcs") if stain_dir else None
        
        if fcs_path and os.path.exists(fcs_path):
            try:
                meta, df_fcs = fcsparser.parse(fcs_path, reformat_meta=True)
                for col in df_fcs.columns:
                    if stain_name.lower() in col.lower() and 'area' in col.lower():
                        vals = np.arcsinh(df_fcs[col].values / cofactor)
                        stain_intensity = vals if stain_intensity is None else np.concatenate([stain_intensity, vals])
                        stain_label = f'{col} (ArcSinh)'
                        break
            except Exception as e:
                print(f"  [Raw CSV UMAP] Warning: Failed to parse FCS for coloring: {e}")
        else:
            # 念のため、CSV内にカラムが存在する場合のフォールバック
            df = pd.read_csv(p)
            for col in df.columns:
                if stain_name.lower() in col.lower() and 'area' in col.lower():
                    vals = np.arcsinh(df[col].values / cofactor)
                    stain_intensity = vals if stain_intensity is None else np.concatenate([stain_intensity, vals])
                    stain_label = f'{col} (ArcSinh)'
                    break

    # --- UMAP ---
    print(f"  [Raw CSV UMAP] Running UMAP on {len(X_scaled)} cells...")
    coords = _umap_fit_transform(X_scaled)

    vmin = vmax = None
    if stain_intensity is not None:
        vmin, vmax = np.percentile(stain_intensity, [1, 99])

    # --- Plot ---
    fig, ax_main = plt.subplots(1, 1, figsize=(8, 6), dpi=150)

    # Plot the UMAP colored by stain intensity if available, otherwise solid color
    if stain_intensity is not None:
        sc = ax_main.scatter(coords[:, 0], coords[:, 1],
                             c=stain_intensity, vmin=vmin, vmax=vmax, cmap='coolwarm', s=2, alpha=0.5)
        plt.colorbar(sc, ax=ax_main, label=stain_label)
    else:
        ax_main.scatter(coords[:, 0], coords[:, 1], c='#c45a5a', s=2, alpha=0.5)

    ax_main.set_title(f'Raw CSV UMAP (All Features) — {stain_name}', fontsize=14, fontweight='bold', pad=10)
    ax_main.set_xlabel('UMAP 1')
    ax_main.set_ylabel('UMAP 2')

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Raw CSV UMAP] Saved: {output_path}")


"""
python analysis/run_pipeline.py --experiment "Experiment 2026!06!02 12!39" --rack "24 Tube Rack (5mL) - 1" --method autoencoder

python analysis/run_pipeline.py   --experiment 'Experiment 2026!06!02 12!39'   --rack '24 Tube Rack (5mL) - 1'  --method poisson_glm

"""

def generate_markdown_report(results_base_dir, stain_name, sraw_files):
    """
    パイプラインで生成された各プロットをMarkdownファイルにまとめます。
    """
    report_path = os.path.join(results_base_dir, f"pipeline_report_{stain_name}.md")
    lines = [f"# Pipeline Report: {stain_name}", ""]
    
    # 1. Group UMAP
    umap_png = f"autofluor_umap_{stain_name}.png"
    umap_html = f"autofluor_umap_{stain_name}.html"
    
    if os.path.exists(os.path.join(results_base_dir, umap_png)):
        lines.append("## Group Autofluor UMAP")
        lines.append(f"[Interactive HTML Report]({umap_html})\n")
        lines.append(f"![UMAP Plot]({umap_png})\n")
        lines.append("---")
        
    # 2. Individual Samples
    lines.append("## Individual Sample Results")
    for filepath in sraw_files:
        filename = os.path.basename(filepath)
        base_name = os.path.splitext(filename)[0]
        well_id = base_name.split(' ')[0] if ' ' in base_name else base_name
        sample_label = f"{stain_name}_{well_id}"
        
        lines.append(f"### Sample: {sample_label}")
        
        # Spectral Density
        spectral_png = f"{sample_label}/spectral_density.png"
        if os.path.exists(os.path.join(results_base_dir, spectral_png)):
            lines.append("#### Spectral Density")
            lines.append(f"![Spectral Density]({spectral_png})\n")
            
        # Histogram
        hist_png = f"{sample_label}/histogram.png"
        if os.path.exists(os.path.join(results_base_dir, hist_png)):
            lines.append("#### Fluorescence Histogram")
            lines.append(f"![Histogram]({hist_png})\n")
            
        # Unmixing Scatter
        unmix_png = f"{sample_label}/unmixing_scatter.png"
        if os.path.exists(os.path.join(results_base_dir, unmix_png)):
            lines.append("#### Spectral Unmixing")
            lines.append(f"![Unmixing Scatter]({unmix_png})\n")
            
        # Unmixing Comparison
        comparison_png = f"{sample_label}/unmixing_comparison_{stain_name}.png"
        if os.path.exists(os.path.join(results_base_dir, comparison_png)):
            lines.append("#### Spectral Unmixing Comparison")
            lines.append(f"![Unmixing Comparison]({comparison_png})\n")
            
        lines.append("---\n")
        
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
        
    print(f"\nMarkdown Report generated: {report_path}")
    
    # PDF output using markdown_pdf
    try:
        from markdown_pdf import Section, MarkdownPdf
        pdf_path = os.path.join(results_base_dir, f"pipeline_report_{stain_name}.pdf")
        
        pdf = MarkdownPdf(toc_level=2)
        # Set root parameter to the results directory so images are resolved correctly
        pdf.add_section(Section('\n'.join(lines), root=results_base_dir))
        pdf.save(pdf_path)
        print(f"PDF Report generated: {pdf_path}")
    except Exception as e:
        print(f"Failed to generate PDF Report: {e}")


# Configure absolute paths based on this file's location
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ANALYSIS_DIR)
RESULTS_DIR = os.path.join(ANALYSIS_DIR, 'results')


def process_stain_files(experiment_folder, rack_name, stain_name):
    sraw_dir = os.path.join(get_experiment_data_dir(experiment_folder), rack_name, stain_name)
    if not os.path.isdir(sraw_dir):
        return []
        
    print(f"=" * 70)
    print(f"Pipeline: {experiment_folder}")
    print(f"  Rack:  {rack_name}")
    print(f"  Stain: {stain_name}")
    print(f"  Dir:   {sraw_dir}")
    print(f"=" * 70)

    sraw_files = find_sraw_files(sraw_dir)
    print(f"\nFound {len(sraw_files)} .sraw file(s)\n")

    for filepath in sraw_files:
        filename = os.path.basename(filepath)
        base_name = os.path.splitext(filename)[0]

        well_id = base_name.split(' ')[0] if ' ' in base_name else base_name
        sample_label = f"{stain_name}_{well_id}"

        result_dir = get_results_dir(experiment_folder, sample_label)

        fcs_path = os.path.join(sraw_dir, base_name + '.fcs')
        has_fcs = os.path.isfile(fcs_path)
        total_steps = 3 if has_fcs else 2

        print(f"[1/{total_steps}] Converting {filename} ...")
        csv_path_ch, csv_path_wl, df_ch, df_wl = convert_sraw_to_csv(filepath, output_dir=result_dir)
        print(f"      -> {csv_path_wl}  (shape: {df_wl.shape})")

        print(f"[2/{total_steps}] Generating spectral density plot ...")
        plot_path = os.path.join(result_dir, 'spectral_density.png')
        plot_spectral_density(csv_path_wl, plot_path)

        if has_fcs:
            print(f"[3/{total_steps}] Generating fluorescence histogram ...")
            hist_path = os.path.join(result_dir, 'histogram.png')
            plot_histogram(fcs_path, hist_path, stain_name=stain_name)
        else:
            print(f"  (Histogram skipped — .fcs file not found: {fcs_path})")

        print()
    return sraw_files


def run_pipeline(experiment_folder, rack_name, method='poisson', retrain=False, **tf_kwargs):
    rack_dir = os.path.join(get_experiment_data_dir(experiment_folder), rack_name)
    if not os.path.isdir(rack_dir):
        print(f"Error: Rack directory not found: {rack_dir}")
        return

    stain_dirs = [d for d in os.listdir(rack_dir) if os.path.isdir(os.path.join(rack_dir, d))]
    negative_stains = [d for d in stain_dirs if d.lower() == 'negative']
    other_stains = sorted([d for d in stain_dirs if d.lower() != 'negative'])

    date_str = EXPERIMENTS.get(experiment_folder, experiment_folder)
    results_base_dir = os.path.join(RESULTS_DIR, date_str)

    # 1. Process Negative first
    for neg_stain in negative_stains:
        sraw_files = process_stain_files(experiment_folder, rack_name, neg_stain)
        # if sraw_files:
        #     generate_markdown_report(results_base_dir, neg_stain, sraw_files)

    # 2. Process other stains
    for stain_name in other_stains:
        sraw_files = process_stain_files(experiment_folder, rack_name, stain_name)
        if not sraw_files:
            continue
            
        neg_dir = os.path.join(get_experiment_data_dir(experiment_folder), rack_name, "Negative")
        if os.path.isdir(neg_dir):
            print(f"\n[Group Pipeline] Running Autofluor UMAP projection and Spectral Unmixing for {stain_name}...")
            
            print(f"  -> Performing Spectral Unmixing... (Method: {method})")
            run_unmixing_group(results_base_dir=results_base_dir, stain_name=stain_name, method=method, retrain=retrain, **tf_kwargs)
            
            # If we retrained the shared model on the first stain, don't retrain it again for subsequent stains
            if method in ['autoencoder', 'transformer', 'scarf'] and retrain:
                print("  -> Shared model retrained. Disabling retrain for remaining stains.")
                retrain = False
            
            print("  -> Generating Unmixing Comparison Plots...")
            neg_csv = find_csv_in_dir(results_base_dir, "Negative")
            if neg_csv:
                stain_csv_pattern = os.path.join(results_base_dir, f"{stain_name}_*", "*_wavelength.csv")
                all_stain_csvs = sorted(list(set([p for p in glob.glob(stain_csv_pattern) if "scarf_embeddings" not in p])))
                
                import re
                latest_stains = {}
                for p in all_stain_csvs:
                    csv_basename = os.path.basename(p)
                    m = re.match(r"(.*)_\d{8}_\d{6}_wavelength\.csv", csv_basename)
                    orig_name = m.group(1) if m else os.path.splitext(csv_basename)[0]
                    latest_stains[orig_name] = p
                stain_csvs = list(latest_stains.values())
                for stain_csv in stain_csvs:
                    comp_out = os.path.join(os.path.dirname(stain_csv), f"unmixing_comparison_{stain_name}.png")
                    plot_unmixing_comparison(neg_csv, stain_csv, comp_out, stain_name=stain_name, method=method)
            
            print("  -> Generating Group UMAP...")
            try:
                stain_dir = os.path.join(rack_dir, stain_name)
                run_umap_autofluor(
                    neg_dir=neg_dir,
                    stain_dir=stain_dir,
                    output_path=os.path.join(results_base_dir, f"autofluor_umap_{stain_name}.html"),
                    png_output_path=os.path.join(results_base_dir, f"autofluor_umap_{stain_name}.png"),
                    stain_name=stain_name,
                    method=method
                )
            except Exception as e:
                print(f"  Warning: UMAP projection failed: {e}")

            print("  -> Generating Raw CSV UMAP (no unmixing)...")
            try:
                raw_umap_out = os.path.join(results_base_dir, f"raw_csv_umap_{stain_name}.png")
                generate_raw_csv_umap(results_base_dir, stain_name, raw_umap_out, stain_dir)
            except Exception as e:
                print(f"  Warning: Raw CSV UMAP failed: {e}")
        else:
            print(f"\nWarning: Could not find Negative directory at {neg_dir}. Skipping group UMAP and Unmixing.")

        # print("\n[Report] Generating Markdown overview...")
        # generate_markdown_report(results_base_dir, stain_name, sraw_files)

    print("\nPipeline complete!")


def main():
    parser = argparse.ArgumentParser(description='解析パイプライン一括実行')
    parser.add_argument('--experiment', type=str, required=True,
                        help='実験フォルダ名 (例: "Experiment 2026!05!21 15!59")')
    parser.add_argument('--rack', type=str, required=True,
                        help='ラック名 (例: "24 Tube Rack (5mL) - 1")')
    parser.add_argument('--method', type=str,
                        choices=['poisson', 'poisson_glm', 'ols'],
                        default='ols',
                        help='アンミキシング手法 (poisson, poisson_glm, ols)')
    parser.add_argument('--retrain', action='store_true',
                        help='キャッシュされた学習済みモデルを使わず再学習する')

    args = parser.parse_args()

    run_pipeline(args.experiment, args.rack, method=args.method, retrain=args.retrain)


if __name__ == '__main__':
    main()
