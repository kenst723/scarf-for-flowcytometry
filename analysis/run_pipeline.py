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
from src.unmix_spectral import run_unmixing_group
from src.plot_unmixing_comparison import plot_unmixing_comparison, find_csv_in_dir


"""
python analysis/run_pipeline.py --experiment "Experiment 2026!06!02 12!39" --rack "24 Tube Rack (5mL) - 1" --method autoencoder

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
        csv_path, df = convert_sraw_to_csv(filepath, output_dir=result_dir)
        print(f"      -> {csv_path}  (shape: {df.shape})")

        print(f"[2/{total_steps}] Generating spectral density plot ...")
        plot_path = os.path.join(result_dir, 'spectral_density.png')
        plot_spectral_density(csv_path, plot_path)

        if has_fcs:
            print(f"[3/{total_steps}] Generating fluorescence histogram ...")
            hist_path = os.path.join(result_dir, 'histogram.png')
            plot_histogram(fcs_path, hist_path, stain_name=stain_name)
        else:
            print(f"  (Histogram skipped — .fcs file not found: {fcs_path})")

        print()
    return sraw_files


def run_pipeline(experiment_folder, rack_name, method='poisson'):
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
        if sraw_files:
            generate_markdown_report(results_base_dir, neg_stain, sraw_files)

    # 2. Process other stains
    for stain_name in other_stains:
        sraw_files = process_stain_files(experiment_folder, rack_name, stain_name)
        if not sraw_files:
            continue
            
        neg_dir = os.path.join(get_experiment_data_dir(experiment_folder), rack_name, "Negative")
        if os.path.isdir(neg_dir):
            print(f"\n[Group Pipeline] Running Autofluor UMAP projection and Spectral Unmixing for {stain_name}...")
            
            print(f"  -> Performing Spectral Unmixing... (Method: {method})")
            run_unmixing_group(results_base_dir=results_base_dir, stain_name=stain_name, method=method)
            
            print("  -> Generating Unmixing Comparison Plots...")
            neg_csv = find_csv_in_dir(results_base_dir, "Negative")
            if neg_csv:
                stain_csv_pattern = os.path.join(results_base_dir, f"{stain_name}_*", "*.csv")
                stain_csvs = sorted(list(set([p for p in glob.glob(stain_csv_pattern) if "scarf_embeddings" not in p])))
                for stain_csv in stain_csvs:
                    comp_out = os.path.join(os.path.dirname(stain_csv), f"unmixing_comparison_{stain_name}.png")
                    plot_unmixing_comparison(neg_csv, stain_csv, comp_out, stain_name=stain_name, method=method)
            
            print("  -> Generating Group UMAP...")
            try:
                sraw_dir = os.path.join(rack_dir, stain_name)
                output_path = os.path.join(results_base_dir, f"autofluor_umap_{stain_name}.html")
                png_path = os.path.join(results_base_dir, f"autofluor_umap_{stain_name}.png")
                run_umap_autofluor(neg_dir, sraw_dir, output_path, stain_name=stain_name, png_output_path=png_path)
            except Exception as e:
                print(f"  Warning: UMAP projection failed: {e}")
        else:
            print(f"\nWarning: Could not find Negative directory at {neg_dir}. Skipping group UMAP and Unmixing.")

        print("\n[Report] Generating Markdown overview...")
        generate_markdown_report(results_base_dir, stain_name, sraw_files)

    print("\nPipeline complete!")


def main():
    parser = argparse.ArgumentParser(description='解析パイプライン一括実行')
    parser.add_argument('--experiment', type=str, required=True,
                        help='実験フォルダ名 (例: "Experiment 2026!05!21 15!59")')
    parser.add_argument('--rack', type=str, required=True,
                        help='ラック名 (例: "24 Tube Rack (5mL) - 1")')
    parser.add_argument('--method', type=str, choices=['poisson', 'scarf', 'autoencoder'], default='poisson',
                        help='アンミキシング手法 (poisson, scarf, autoencoder)')
    args = parser.parse_args()

    run_pipeline(args.experiment, args.rack, method=args.method)


if __name__ == '__main__':
    main()
