import os
import sys
import argparse
import glob

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import umap
import fcsparser
from sklearn.preprocessing import StandardScaler

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import COFACTOR
from src.convert import convert_sraw_to_csv


def load_spectral_data(sraw_path, fcs_path, cofactor=None):
    if cofactor is None:
        cofactor = COFACTOR

    temp_dir = os.path.join(PROJECT_ROOT, "analysis", "results", "_temp_autofluor")
    os.makedirs(temp_dir, exist_ok=True)
    csv_path, df_sraw = convert_sraw_to_csv(sraw_path, output_dir=temp_dir)

    _, df_fcs = fcsparser.parse(fcs_path, reformat_meta=True)
    assert len(df_sraw) == len(df_fcs), \
        f"Event counts do not match: sraw={len(df_sraw)}, fcs={len(df_fcs)}"

    wl_features = [
        c for c in df_sraw.columns
        if c.startswith('Area_') and c.endswith('nm') and '638.6nm' not in c
    ]
    X_spectral = df_sraw[wl_features].values

    try:
        os.remove(csv_path)
    except OSError:
        pass

    return X_spectral, wl_features, df_fcs


def find_sraw_fcs_pairs(directory):
    pairs = []
    sraw_files = sorted(glob.glob(os.path.join(directory, "*.sraw")))
    for sraw_path in sraw_files:
        base = os.path.splitext(sraw_path)[0]
        fcs_path = base + ".fcs"
        if os.path.isfile(fcs_path):
            pairs.append((sraw_path, fcs_path))
        else:
            print(f"  Warning: No matching .fcs for {os.path.basename(sraw_path)}, skipping.")
    return pairs


def run_umap_autofluor(neg_dir, stain_dir, output_path, stain_name="PI",
                       cofactor=None, seed=42, png_output_path=None):
    if cofactor is None:
        cofactor = COFACTOR

    # =========================================================================
    # 1. Loading Negative Samples (used for Reference Spectra extraction)
    # =========================================================================
    print("=" * 60)
    print("Step 1: Loading Negative (autofluorescence) samples...")
    print("=" * 60)

    neg_pairs = find_sraw_fcs_pairs(neg_dir)
    if not neg_pairs:
        print(f"Error: No .sraw/.fcs pairs found in {neg_dir}")
        sys.exit(1)

    neg_pairs = neg_pairs[:1]
    neg_spectral_list = []
    wl_features = None

    for sraw_path, fcs_path in neg_pairs:
        print(f"  Loading {os.path.basename(sraw_path)}...")
        X_sp, wl_feat, _ = load_spectral_data(sraw_path, fcs_path, cofactor)
        neg_spectral_list.append(X_sp)
        if wl_features is None:
            wl_features = wl_feat

    X_neg = np.vstack(neg_spectral_list)
    print(f"  Total Negative events: {len(X_neg)} ({len(neg_pairs)} files)")

    # =========================================================================
    # 2. Loading Stained Samples
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"Step 2: Loading {stain_name}-stained samples...")
    print("=" * 60)

    stain_pairs = find_sraw_fcs_pairs(stain_dir)
    if not stain_pairs:
        print(f"Error: No .sraw/.fcs pairs found in {stain_dir}")
        sys.exit(1)

    stain_spectral_list = []
    stain_fcs_list = []

    for sraw_path, fcs_path in stain_pairs:
        print(f"  Loading {os.path.basename(sraw_path)}...")
        X_sp, _, df_fcs = load_spectral_data(sraw_path, fcs_path, cofactor)
        stain_spectral_list.append(X_sp)
        stain_fcs_list.append(df_fcs)

    X_stain = np.vstack(stain_spectral_list)
    df_stain_fcs = pd.concat(stain_fcs_list, ignore_index=True)
    print(f"  Total {stain_name} events: {len(X_stain)} ({len(stain_pairs)} files)")

    # =========================================================================
    # 3. Extracting unmixed intensity and Reconstructing AF Spectra
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"Step 3: Extracting unmixed intensity & Reconstructing AF Spectra...")
    print("=" * 60)

    def find_processed_csv(sraw_path, target_stain):
        parts = os.path.normpath(sraw_path).split(os.sep)
        try:
            data_idx = parts.index("data")
            experiment_folder = parts[data_idx + 1]
        except (ValueError, IndexError):
            experiment_folder = "Experiment 2026!05!21 15!59"
            
        from config import EXPERIMENTS, RESULTS_DIR
        date_str = EXPERIMENTS.get(experiment_folder, experiment_folder)
        
        filename = os.path.basename(sraw_path)
        base_name = os.path.splitext(filename)[0]
        well_id = base_name.split(' ')[0] if ' ' in base_name else base_name
        
        sample_label = f"{target_stain}_{well_id}"
        pattern = os.path.join(RESULTS_DIR, date_str, sample_label, "*.csv")
        csv_files = glob.glob(pattern)
        csv_files = [p for p in csv_files if "scarf_embeddings" not in p]
        if csv_files:
            csv_files.sort(reverse=True)  # Get the most recent file by timestamp in filename
            return csv_files[0]
        return None

    stain_unmixed_af = []
    stain_unmixed_stain = []
    
    for sraw_path, _ in stain_pairs:
        csv_path = find_processed_csv(sraw_path, stain_name)
        if csv_path:
            df_csv = pd.read_csv(csv_path)
            if 'Unmixed_AF' in df_csv.columns and f'Unmixed_{stain_name}' in df_csv.columns:
                stain_unmixed_af.append(df_csv['Unmixed_AF'].values)
                stain_unmixed_stain.append(df_csv[f'Unmixed_{stain_name}'].values)
                
    has_unmixed = False
    if len(stain_unmixed_af) == len(stain_pairs):
        stain_unmixed_af = np.concatenate(stain_unmixed_af)
        stain_unmixed_stain = np.concatenate(stain_unmixed_stain)
        af_intensity = np.arcsinh(stain_unmixed_af / cofactor)
        stain_intensity = np.arcsinh(stain_unmixed_stain / cofactor)
        af_label = 'Unmixed AF (ArcSinh)'
        stain_label = f'Unmixed {stain_name} (ArcSinh)'
        has_unmixed = True
        print(f"  Successfully loaded unmixed intensities from processed CSVs.")
        
        # Calculate reference spectra directly to reconstruct AF
        print("  Reconstructing pure Autofluorescence spectra for Stained samples using PoissonUnmixer...")
        from src.unmix_spectral import PoissonUnmixer
        unmixer = PoissonUnmixer()
        unmixer.fit(X_neg, X_stain)
        
        # Subtract stain component from raw spectra using the refined S_Stain
        X_to_umap = X_stain - stain_unmixed_stain[:, None] * unmixer.S_Stain[None, :]
        X_to_umap = np.maximum(X_to_umap, 0)
    else:
        print("  Warning: Unmixed CSV data not found for all samples. Falling back to raw spectra.")
        X_to_umap = X_stain
        
        stain_col = None
        for col in df_stain_fcs.columns:
            if stain_name.lower() in col.lower() and 'area' in col.lower():
                stain_col = col
                break

        if stain_col is not None:
            stain_intensity = np.arcsinh(df_stain_fcs[stain_col].values / cofactor)
            stain_label = f'{stain_col} (ArcSinh)'
        else:
            stain_intensity = np.arcsinh(X_stain.mean(axis=1) / cofactor)
            stain_label = 'Mean Spectral Intensity (ArcSinh)'
            
    vmin, vmax = np.percentile(stain_intensity, [1, 99])

    # =========================================================================
    # 4. Preprocessing (ArcSinh + StandardScaler)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Step 4: Preprocessing (ArcSinh + StandardScaler)...")
    print("=" * 60)

    X_to_umap_arcsinh = np.arcsinh(X_to_umap / cofactor)
    scaler = StandardScaler()
    X_to_umap_scaled = scaler.fit_transform(X_to_umap_arcsinh)

    # =========================================================================
    # 5. 2D UMAP
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Step 5: Running 2D UMAP on reconstructed AF data...")
    print("=" * 60)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.3,
        metric='euclidean',
        random_state=seed
    )

    umap_coords = reducer.fit_transform(X_to_umap_scaled)

    # =========================================================================
    # 6. Plotly Visualization
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Step 6: Generating interactive 2D plot...")
    print("=" * 60)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'Stained ({stain_name}) - Autofluorescence UMAP',
            f'Stained ({stain_name}) - Colored by {stain_label}'
        )
    )
    
    # Left panel: Uncolored (gray) AF UMAP of the stained sample
    fig.add_trace(
        go.Scatter(
            x=umap_coords[:, 0], y=umap_coords[:, 1],
            mode='markers', name='Autofluorescence UMAP',
            marker=dict(
                size=3, color='#d3d3d3',
                opacity=0.6
            ),
            showlegend=False
        ), row=1, col=1
    )
    
    # Right panel: Same UMAP colored by PI
    fig.add_trace(
        go.Scatter(
            x=umap_coords[:, 0], y=umap_coords[:, 1],
            mode='markers', name=f'{stain_name} Intensity',
            marker=dict(
                size=3, color=stain_intensity,
                cmin=vmin, cmax=vmax,
                colorscale='bluered', opacity=0.7,
                colorbar=dict(title=stain_label, x=1.0)
            ),
            showlegend=False
        ), row=1, col=2
    )

    fig.update_layout(
        title=f'Reconstructed Autofluorescence UMAP + {stain_name} Projection',
        width=1400, height=700,
        margin=dict(l=40, r=40, b=40, t=60),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    fig.update_xaxes(title_text='UMAP 1')
    fig.update_yaxes(title_text='UMAP 2')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.write_html(output_path)
    print(f"\n  Interactive 2D plot saved to: {output_path}")

    # Generate Matplotlib PNG if png_output_path is specified
    if png_output_path:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig_mpl, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=200)
        
        # Left: Uncolored AF UMAP
        axes[0].scatter(
            umap_coords[:, 0], umap_coords[:, 1], 
            c='#d3d3d3', s=2, alpha=0.5
        )
        axes[0].set_title(f'Stained ({stain_name}) - Autofluorescence UMAP\n(Uncolored)', fontsize=12, fontweight='bold', pad=10)
        axes[0].set_xlabel('UMAP 1', fontsize=10)
        axes[0].set_ylabel('UMAP 2', fontsize=10)
        
        # Right: Colored by Stain
        sc2 = axes[1].scatter(
            umap_coords[:, 0], umap_coords[:, 1], 
            c=stain_intensity, vmin=vmin, vmax=vmax, cmap='coolwarm', s=2, alpha=0.5
        )
        axes[1].set_title(f'Stained ({stain_name}) - Target Dye Intensity\n({stain_label})', fontsize=12, fontweight='bold', pad=10)
        axes[1].set_xlabel('UMAP 1', fontsize=10)
        axes[1].set_ylabel('UMAP 2', fontsize=10)
        fig_mpl.colorbar(sc2, ax=axes[1], label=stain_label)
        
        fig_mpl.suptitle(f'Reconstructed Autofluorescence UMAP + {stain_name} Projection', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        os.makedirs(os.path.dirname(png_output_path) or '.', exist_ok=True)
        plt.savefig(png_output_path, bbox_inches='tight')
        plt.close()
        print(f"  Static 2D plot saved to: {png_output_path}")

    temp_dir = os.path.join(PROJECT_ROOT, "analysis", "results", "_temp_autofluor")
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description='自家蛍光 UMAP + マーカー投影 (2D Plotly)'
    )
    parser.add_argument('--neg-dir', type=str, required=True,
                        help='ネガティブコントロールのデータディレクトリ')
    parser.add_argument('--stain-dir', type=str, required=True,
                        help='マーカー染色サンプルのデータディレクトリ')
    parser.add_argument('--stain', type=str, default='PI',
                        help='染色マーカー名 (デフォルト: PI)')
    parser.add_argument('--output', type=str, default=None,
                        help='出力 HTML ファイルのパス')
    parser.add_argument('--png-output', type=str, default=None,
                        help='出力 PNG ファイルのパス')
    parser.add_argument('--cofactor', type=float, default=None,
                        help='ArcSinh cofactor')
    parser.add_argument('--seed', type=int, default=42,
                        help='UMAP の乱数シード')

    args = parser.parse_args()

    if args.output is None:
        results_dir = os.path.join(PROJECT_ROOT, "analysis", "results")
        os.makedirs(results_dir, exist_ok=True)
        args.output = os.path.join(results_dir, f"autofluor_umap_{args.stain}.html")

    run_umap_autofluor(
        neg_dir=args.neg_dir,
        stain_dir=args.stain_dir,
        output_path=args.output,
        stain_name=args.stain,
        cofactor=args.cofactor,
        seed=args.seed,
        png_output_path=args.png_output
    )


if __name__ == '__main__':
    main()
