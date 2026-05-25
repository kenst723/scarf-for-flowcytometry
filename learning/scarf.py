"""
SCARF (Self-Supervised Contrastive Learning for Tabular Data) Model Training

This script trains a SCARF model on spectral + scatter flow cytometry data
and extracts and saves the learned embeddings into a CSV file.

Usage:
    python -m learning.scarf --csv <sraw_csv> --fcs <fcs_file> --output-dir <output_dir>
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import fcsparser
import torch
from sklearn.preprocessing import StandardScaler

# Add project root and pytorch-scarf directory to system path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "pytorch-scarf"))

from config import COFACTOR
from scarf.loss import NTXent
from scarf.model import SCARF
from scarf.dataset import SCARFDataset

# ---------------------------------------------------------------------------
# デフォルトパス設定 (特定のファイルを個別に処理したい場合はここにパスを記述します)
# ---------------------------------------------------------------------------
DEFAULT_CSV_PATH = None  # 例: "results/2026-05-21/PI_A01/A01 Well - A01_20260523_164523.csv"
DEFAULT_FCS_PATH = None  # 例: "data/Experiment 2026!05!21 15!59/24 Tube Rack (5mL) - 1/PI/A01 Well - A01.fcs"


def fix_seed(seed):
    """Fix random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def run_scarf_pipeline(csv_path, fcs_path, output_dir, cofactor=None,
                       epochs=200, batch_size=128, lr=0.001, temperature=1.0,
                       dim_hidden_encoder=16, num_hidden_encoder=4,
                       dim_hidden_head=16, num_hidden_head=2,
                       corruption_rate=0.6, dropout=0.0, num_frequencies=4, sigma=1.0, seed=42, sample_label=None):
    """
    Train SCARF on spectral + scatter data and save embeddings to a CSV file.
    """
    fix_seed(seed)
    
    if cofactor is None:
        cofactor = COFACTOR

    # Create output directory (defaults to learning/result)
    if output_dir is None:
        scarf_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(scarf_dir, "result")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading spectral data from {os.path.basename(csv_path)}...")
    df_sraw = pd.read_csv(csv_path)

    print(f"Loading scatter data from {os.path.basename(fcs_path)}...")
    _, df_fcs = fcsparser.parse(fcs_path, reformat_meta=True)

    # Verify event count alignment
    assert len(df_sraw) == len(df_fcs), "Event counts do not match between CSV and FCS!"

    # --- 1. Feature Selection ---
    # 33 Wavelength channels (excluding 638.6nm laser noise)
    wl_features = [c for c in df_sraw.columns if c.startswith('Area_') and c.endswith('nm') and '638.6nm' not in c]
    X_spectral = df_sraw[wl_features].values

    # FSC and SSC (Area)
    scatter_features = ['FSC - Area', 'SSC - Area']
    X_scatter = df_fcs[scatter_features].values

    print(f"Selected {len(wl_features)} spectral channels + {len(scatter_features)} scatter channels.")

    # Combine spectral and scatter features
    X_combined = np.hstack((X_scatter, X_spectral))

    # --- 2. Preprocessing ---
    # print("Applying ArcSinh transformation...")
    # X_arcsinh = np.arcsinh(X_combined / cofactor)

    print("Applying StandardScaler (Z-score normalization)...")
    scaler = StandardScaler()
    # ArcSinhをスキップし、生データをそのまま標準化
    X_scaled = scaler.fit_transform(X_combined)

    # --- 3. Dataset & DataLoader ---
    dummy_targets = np.zeros(len(X_scaled))
    dataset = SCARFDataset(X_scaled, dummy_targets)
    
    # Use drop_last=True during training to avoid issues with small batch sizes in contrastive loss
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
    )

    # --- 4. Initialize SCARF Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SCARF(
        input_dim=dataset.shape[1],
        features_low=dataset.features_low,
        features_high=dataset.features_high,
        dim_hidden_encoder=dim_hidden_encoder,
        num_hidden_encoder=num_hidden_encoder,
        dim_hidden_head=dim_hidden_head,
        num_hidden_head=num_hidden_head,
        corruption_rate=corruption_rate,
        dropout=dropout,
        num_frequencies=num_frequencies,
        sigma=sigma
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = NTXent(temperature=temperature)

    # --- 5. Model Training ---
    print(f"Training SCARF model for {epochs} epochs (batch size: {batch_size})...")
    
    try:
        from tqdm import tqdm
        epochs_iter = tqdm(range(1, epochs + 1), desc="Training SCARF")
    except ImportError:
        epochs_iter = range(1, epochs + 1)

    for epoch in epochs_iter:
        model.train()
        epoch_loss = 0.0
        for x in dataloader:
            x = x.to(device)
            
            # Forward pass
            emb_anchor, emb_positive = model(x)
            loss = criterion(emb_anchor, emb_positive)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            
        epoch_loss_avg = epoch_loss / len(dataloader)
        
        # Display progress
        if isinstance(epochs_iter, range):
            if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
                print(f"  Epoch {epoch:03d}/{epochs:03d} - Loss: {epoch_loss_avg:.4f}")
        else:
            epochs_iter.set_postfix({"loss": f"{epoch_loss_avg:.4f}"})

    # --- 6. Extract Embeddings ---
    print("Extracting SCARF embeddings...")
    eval_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for x in eval_dataloader:
            x = x.to(device)
            emb = model.get_embeddings(x)
            all_embeddings.append(emb.cpu().numpy())
            
    embeddings = np.concatenate(all_embeddings, axis=0)

    # Save embeddings to CSV
    emb_columns = [f"SCARF_Dim_{i+1}" for i in range(embeddings.shape[1])]
    df_embeddings = pd.DataFrame(embeddings, columns=emb_columns)
    
    if sample_label:
        filename = f"{sample_label}_scarf_embeddings.csv"
    else:
        csv_base = os.path.basename(csv_path)
        well_id = csv_base.split('_')[0] if '_' in csv_base else os.path.splitext(csv_base)[0]
        filename = f"{well_id}_scarf_embeddings.csv"
        
    embeddings_csv_path = os.path.join(output_dir, filename)
    df_embeddings.to_csv(embeddings_csv_path, index=False)
    print(f"SCARF embeddings saved to: {embeddings_csv_path} (shape: {embeddings.shape})")
    print("SCARF Pipeline run complete!")


def discover_samples():
    """
    Automatically discover all sample files (CSV and corresponding FCS)
    by scanning the data/ and results/ directories according to config.
    """
    import glob
    from config import DATA_DIR, RESULTS_DIR, EXPERIMENTS
    
    samples = []
    if not os.path.isdir(DATA_DIR):
        print(f"Data directory not found: {DATA_DIR}")
        return samples

    # Iterate through all configured experiments
    for exp_folder, date_str in EXPERIMENTS.items():
        exp_data_dir = os.path.join(DATA_DIR, exp_folder)
        if not os.path.isdir(exp_data_dir):
            continue
            
        # Scan racks (e.g. "24 Tube Rack (5mL) - 1")
        for rack in os.listdir(exp_data_dir):
            rack_dir = os.path.join(exp_data_dir, rack)
            if not os.path.isdir(rack_dir):
                continue
                
            # Scan stains (e.g. "PI", "Negative")
            for stain in os.listdir(rack_dir):
                stain_dir = os.path.join(rack_dir, stain)
                if not os.path.isdir(stain_dir):
                    continue
                    
                # Find all .fcs files in the stain directory
                for f in os.listdir(stain_dir):
                    if f.endswith('.fcs'):
                        fcs_path = os.path.join(stain_dir, f)
                        base_name = os.path.splitext(f)[0]
                        well_id = base_name.split(' ')[0] if ' ' in base_name else base_name
                        sample_label = f"{stain}_{well_id}"
                        
                        # Look for converted CSV in results directory
                        result_dir = os.path.join(RESULTS_DIR, date_str, sample_label)
                        if os.path.isdir(result_dir):
                            # Find all CSV files starting with base_name
                            csv_pattern = os.path.join(result_dir, f"{base_name}_*.csv")
                            csv_files = sorted(glob.glob(csv_pattern))
                            
                            # Exclude any embeddings or other files
                            csv_files = [cf for cf in csv_files if "scarf_embeddings" not in cf]
                            
                            if csv_files:
                                # Use the most recent CSV (latest sorted timestamp)
                                matched_csv = csv_files[-1]
                                samples.append({
                                    'csv': matched_csv,
                                    'fcs': fcs_path,
                                    'output_dir': result_dir,
                                    'sample_label': sample_label
                                })
                                
    return samples


def main():
    parser = argparse.ArgumentParser(description='SCARF self-supervised representation learning')
    parser.add_argument('--csv', type=str, default=None, help='Path to the converted sraw CSV file (optional)')
    parser.add_argument('--fcs', type=str, default=None, help='Path to the corresponding .fcs file (optional)')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save outputs')
    parser.add_argument('--cofactor', type=float, default=None, help='ArcSinh transformation cofactor')
    parser.add_argument('--epochs', type=int, default=200, help='Number of SCARF training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--temperature', type=float, default=1.0, help='NT-Xent similarity temperature')
    parser.add_argument('--dim-hidden-encoder', type=int, default=16, help='Hidden size (and embedding dim) of encoder')
    parser.add_argument('--num-hidden-encoder', type=int, default=4, help='Number of hidden layers in encoder')
    parser.add_argument('--dim-hidden-head', type=int, default=16, help='Hidden size of projection head')
    parser.add_argument('--num-hidden-head', type=int, default=2, help='Number of hidden layers in projection head')
    parser.add_argument('--corruption-rate', type=float, default=0.6, help='Feature corruption rate for SCARF')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-frequencies', type=int, default=4, help='Number of frequencies for Periodic Encoding')
    parser.add_argument('--sigma', type=float, default=1.0, help='Standard deviation for frequency initialization')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()

    csv_path = args.csv
    fcs_path = args.fcs

    # Use defaults if defined in the script
    if csv_path is None and fcs_path is None:
        if DEFAULT_CSV_PATH is not None and DEFAULT_FCS_PATH is not None:
            csv_path = DEFAULT_CSV_PATH
            fcs_path = DEFAULT_FCS_PATH
            
            # Resolve relative paths relative to PROJECT_ROOT
            if not os.path.isabs(csv_path) and not os.path.exists(csv_path):
                csv_path = os.path.join(PROJECT_ROOT, csv_path)
            if not os.path.isabs(fcs_path) and not os.path.exists(fcs_path):
                fcs_path = os.path.join(PROJECT_ROOT, fcs_path)
                
            print(f"Using default paths defined in scarf.py:")
            print(f"  CSV: {csv_path}")
            print(f"  FCS: {fcs_path}")

    # If only one of csv_path or fcs_path is specified, raise an error
    if (csv_path is None) != (fcs_path is None):
        parser.error("Either specify both CSV and FCS paths (via CLI or DEFAULT_*_PATH), or specify neither to auto-discover paths.")

    if csv_path is not None and fcs_path is not None:
        # Try to extract sample_label from path structure
        csv_base = os.path.basename(csv_path)
        well_id = csv_base.split('_')[0] if '_' in csv_base else os.path.splitext(csv_base)[0]
        parent_dir = os.path.basename(os.path.dirname(csv_path))
        sample_label = parent_dir if '_' in parent_dir else well_id

        # Run on the specified files
        run_scarf_pipeline(
            csv_path=csv_path,
            fcs_path=fcs_path,
            output_dir=args.output_dir,
            cofactor=args.cofactor,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            temperature=args.temperature,
            dim_hidden_encoder=args.dim_hidden_encoder,
            num_hidden_encoder=args.num_hidden_encoder,
            dim_hidden_head=args.dim_hidden_head,
            num_hidden_head=args.num_hidden_head,
            corruption_rate=args.corruption_rate,
            dropout=args.dropout,
            num_frequencies=args.num_frequencies,
            sigma=args.sigma,
            seed=args.seed,
            sample_label=sample_label
        )
    else:
        # Auto-discover files based on config
        print("No CSV/FCS paths provided. Scanning directories for matching files based on config.py...")
        samples = discover_samples()
        if not samples:
            print("No matching CSV and FCS samples found. Please run the pipeline first to generate CSVs.")
            sys.exit(1)
            
        print(f"Discovered {len(samples)} sample(s) to process:")
        for idx, sample in enumerate(samples, 1):
            print(f"  {idx}. Sample: {sample['sample_label']}")
            print(f"     CSV: {os.path.basename(sample['csv'])}")
            print(f"     FCS: {os.path.basename(sample['fcs'])}")
            
        for idx, sample in enumerate(samples, 1):
            print("\n" + "=" * 60)
            print(f"Processing sample {idx}/{len(samples)}: {sample['sample_label']}")
            print("=" * 60)
            run_scarf_pipeline(
                csv_path=sample['csv'],
                fcs_path=sample['fcs'],
                output_dir=args.output_dir,
                cofactor=args.cofactor,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                temperature=args.temperature,
                dim_hidden_encoder=args.dim_hidden_encoder,
                num_hidden_encoder=args.num_hidden_encoder,
                dim_hidden_head=args.dim_hidden_head,
                num_hidden_head=args.num_hidden_head,
                corruption_rate=args.corruption_rate,
                dropout=args.dropout,
                num_frequencies=args.num_frequencies,
                sigma=args.sigma,
                seed=args.seed,
                sample_label=sample['sample_label']
            )


if __name__ == '__main__':
    main()

