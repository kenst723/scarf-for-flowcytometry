"""
SCARF Embeddings 3D UMAP Visualization

This script runs 3D UMAP on SCARF embeddings and saves the plot to the result folder.

Usage:
    python -m learning.umap_embeddings_3d --embeddings <embeddings_csv> --fcs <fcs_file>
    or run without arguments to auto-discover and process all generated embeddings.
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import fcsparser
import umap

# Add project root to system path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import COFACTOR
from learning.scarf import discover_samples


def run_umap_projection_3d(embeddings_path, fcs_path, output_dir, stain_name=None, cofactor=None, seed=42, sample_label=None):
    """Run 3D UMAP on SCARF embeddings and save the plots."""
    if cofactor is None:
        cofactor = COFACTOR

    # Create output directory (defaults to learning/result)
    if output_dir is None:
        scarf_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(scarf_dir, "result")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading SCARF embeddings from {os.path.basename(embeddings_path)}...")
    df_embeddings = pd.read_csv(embeddings_path)
    X_embeddings = df_embeddings.values

    print(f"Loading scatter data from {os.path.basename(fcs_path)}...")
    _, df_fcs = fcsparser.parse(fcs_path, reformat_meta=True)

    # Verify event count alignment
    assert len(X_embeddings) == len(df_fcs), "Event counts do not match between embeddings and FCS!"

    # Setup stain intensity coloring helper
    stain_col = None
    if stain_name and stain_name.lower() != 'negative':
        for col in df_fcs.columns:
            if stain_name.lower() in col.lower() and 'area' in col.lower():
                stain_col = col
                break

    if stain_col is not None:
        stain_values = np.arcsinh(df_fcs[stain_col].values / cofactor)
        stain_label_str = f'{stain_col} (ArcSinh)'
        stain_title_suffix = f'Colored by {stain_col}'
        print(f"Coloring plots by FCS channel: {stain_col}")
    else:
        # Find any other fluorescence channel (excluding FSC/SSC)
        fluor_cols = [col for col in df_fcs.columns if 'area' in col.lower() and 'fsc' not in col.lower() and 'ssc' not in col.lower()]
        if fluor_cols:
            stain_col = fluor_cols[0]
            stain_values = np.arcsinh(df_fcs[stain_col].values / cofactor)
            stain_label_str = f'{stain_col} (ArcSinh)'
            stain_title_suffix = f'Colored by {stain_col}'
            print(f"Stain not found, automatically selected fluorescence channel: {stain_col}")
        else:
            stain_values = df_fcs['SSC - Area'].values
            stain_label_str = 'SSC - Area'
            stain_title_suffix = 'Colored by SSC - Area'
            print("No fluorescence channels found, falling back to SSC - Area")

    # UMAP Dimensionality Reduction (3D)
    print("Running 3D UMAP on SCARF embeddings (this may take 10-30 seconds)...")
    reducer = umap.UMAP(
        n_components=3,  # Set to 3D
        n_neighbors=10,
        min_dist=0.3,
        metric='euclidean',
        random_state=seed
    )
    umap_embedding = reducer.fit_transform(X_embeddings)

    print("Generating interactive 3D UMAP plot (HTML)...")
    
    # Create DataFrame for plotly
    plot_df = pd.DataFrame({
        'UMAP1': umap_embedding[:, 0],
        'UMAP2': umap_embedding[:, 1],
        'UMAP3': umap_embedding[:, 2],
        'FSC_Area': df_fcs['FSC - Area'].values,
        'Stain_Intensity': stain_values
    })

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=('Colored by FSC (Cell Size)', f'Colored by {stain_label_str}')
    )

    # Add FSC scatter
    fig.add_trace(
        go.Scatter3d(
            x=plot_df['UMAP1'], y=plot_df['UMAP2'], z=plot_df['UMAP3'],
            mode='markers',
            marker=dict(
                size=2,
                color=plot_df['FSC_Area'],
                colorscale='Jet',
                opacity=0.7,
                colorbar=dict(title="FSC Area", x=0.45)
            )
        ),
        row=1, col=1
    )

    # Add Stain scatter
    fig.add_trace(
        go.Scatter3d(
            x=plot_df['UMAP1'], y=plot_df['UMAP2'], z=plot_df['UMAP3'],
            mode='markers',
            marker=dict(
                size=2,
                color=plot_df['Stain_Intensity'],
                colorscale='Jet',
                opacity=0.7,
                colorbar=dict(title=stain_label_str, x=1.0)
            )
        ),
        row=1, col=2
    )

    title_label = sample_label if sample_label else os.path.splitext(os.path.basename(embeddings_path))[0]
    fig.update_layout(
        title=f'SCARF + 3D UMAP Representation — {title_label}',
        width=1600,
        height=800,
        margin=dict(l=0, r=0, b=0, t=50)
    )

    output_filename = f"{title_label}_umap_3d.html" if sample_label else "scarf_umap_3d.html"
    umap_plot_path = os.path.join(output_dir, output_filename)
    fig.write_html(umap_plot_path)
    print(f"Interactive 3D UMAP plot saved to: {umap_plot_path}")


def main():
    parser = argparse.ArgumentParser(description='SCARF Embeddings 3D UMAP Visualization')
    parser.add_argument('--embeddings', type=str, default=None, help='Path to the SCARF embeddings CSV file (optional)')
    parser.add_argument('--fcs', type=str, default=None, help='Path to the corresponding .fcs file (optional)')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save the UMAP plot')
    parser.add_argument('--stain', type=str, default=None, help='Stain name for coloring (e.g. PI, Calcein)')
    parser.add_argument('--cofactor', type=float, default=None, help='ArcSinh transformation cofactor')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for UMAP')

    args = parser.parse_args()

    # If only one of --embeddings or --fcs is specified, raise an error
    if (args.embeddings is None) != (args.fcs is None):
        parser.error("Either specify both --embeddings and --fcs, or specify neither to auto-discover.")

    if args.embeddings is not None and args.fcs is not None:
        # Determine stain name if not provided
        stain_name = args.stain
        if stain_name is None:
            # Try to guess stain from path or filename
            for part in os.path.basename(args.embeddings).split('_'):
                if part.lower() in ['pi', 'calcein', 'negative']:
                    stain_name = part
                    break

        run_umap_projection_3d(
            embeddings_path=args.embeddings,
            fcs_path=args.fcs,
            output_dir=args.output_dir,
            stain_name=stain_name,
            cofactor=args.cofactor,
            seed=args.seed
        )
    else:
        print("Scanning results directory for matching SCARF embeddings...")
        samples = discover_samples()
        scarf_dir = os.path.dirname(os.path.abspath(__file__))
        result_dir = os.path.join(scarf_dir, "result")
        
        processed_count = 0
        
        for sample in samples:
            sample_label = sample['sample_label']
            stain_name = sample_label.split('_')[0] if '_' in sample_label else None
            
            # Check if the embeddings CSV exists in learning/result
            embeddings_filename = f"{sample_label}_scarf_embeddings.csv"
            embeddings_path = os.path.join(result_dir, embeddings_filename)
            
            if os.path.isfile(embeddings_path):
                print("\n" + "=" * 60)
                print(f"Running 3D UMAP for sample: {sample_label}")
                print("=" * 60)
                run_umap_projection_3d(
                    embeddings_path=embeddings_path,
                    fcs_path=sample['fcs'],
                    output_dir=args.output_dir,
                    stain_name=stain_name,
                    cofactor=args.cofactor,
                    seed=args.seed,
                    sample_label=sample_label
                )
                processed_count += 1
                
        if processed_count == 0:
            print("No matching *_scarf_embeddings.csv files found in the result directory.")
            print(f"Please run scarf.py first to generate embeddings in: {result_dir}")
            sys.exit(1)


if __name__ == '__main__':
    main()
