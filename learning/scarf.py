"""
SCARF (Self-Supervised Contrastive Learning for Tabular Data) Model Training

This script trains a GLOBAL SCARF model on spectral + scatter flow cytometry data
across all discovered samples (Negative + stained), and extracts and saves the 
learned embeddings into a CSV file for each sample.

Usage:
    python -m learning.scarf
"""

import os
import sys
import argparse
import glob

import numpy as np
import pandas as pd
import fcsparser
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Add project root and pytorch-scarf directory to system path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "pytorch-scarf"))

from config import COFACTOR
from scarf.loss import NTXent
from scarf.model import SCARF
from scarf.dataset import SCARFDataset


def fix_seed(seed):
    """Fix random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def discover_samples():
    """
    Automatically discover all sample files (CSV and corresponding FCS)
    by scanning the data/ and analysis/results/ directories according to config.
    """
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
                                
                                # Define new output dir in learning/result
                                learning_dir = os.path.dirname(os.path.abspath(__file__))
                                learning_result_dir = os.path.join(learning_dir, "result", date_str, sample_label)
                                
                                samples.append({
                                    'csv': matched_csv,
                                    'fcs': fcs_path,
                                    'output_dir': learning_result_dir,
                                    'sample_label': sample_label
                                })
                                
    return samples


def run_global_scarf_pipeline(samples, epochs=200, batch_size=128, lr=0.001, temperature=1.0,
                              dim_hidden_encoder=16, num_hidden_encoder=4,
                              dim_hidden_head=16, num_hidden_head=2,
                              corruption_rate=0.6, dropout=0.0, num_frequencies=4, sigma=1.0, 
                              embedding_dim=16, seed=42):
    """
    Train SCARF on all samples combined, then extract and save embeddings per sample.
    """
    fix_seed(seed)
    
    print(f"Loading {len(samples)} samples to build a global dataset...")
    X_combined_list = []
    sample_indices = []
    current_idx = 0
    
    for sample in samples:
        print(f"  Loading {sample['sample_label']}...")
        df_sraw = pd.read_csv(sample['csv'])
        _, df_fcs = fcsparser.parse(sample['fcs'], reformat_meta=True)
        assert len(df_sraw) == len(df_fcs), f"Event counts do not match for {sample['sample_label']}"
        
        # Extract features
        wl_features = [c for c in df_sraw.columns if c.startswith('Area_') and c.endswith('nm') and '638.6nm' not in c]
        X_spectral = df_sraw[wl_features].values
        scatter_features = ['FSC - Area', 'SSC - Area']
        X_scatter = df_fcs[scatter_features].values
        
        X_combined = np.hstack((X_scatter, X_spectral))
        X_combined_list.append(X_combined)
        
        n_events = len(X_combined)
        sample_indices.append({
            'label': sample['sample_label'],
            'start': current_idx,
            'end': current_idx + n_events,
            'output_dir': sample['output_dir']
        })
        current_idx += n_events

    X_all = np.vstack(X_combined_list)
    print(f"Total events in global dataset: {len(X_all)}")
    
    print("Applying StandardScaler (Z-score normalization) globally...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    
    dummy_targets = np.zeros(len(X_scaled))
    dataset = SCARFDataset(X_scaled, dummy_targets)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

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
        sigma=sigma,
        embedding_dim=embedding_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = NTXent(temperature=temperature)

    # --- Training ---
    print(f"Training SCARF model for {epochs} epochs (batch size: {batch_size})...")
    epochs_iter = tqdm(range(1, epochs + 1), desc="Training SCARF")

    for epoch in epochs_iter:
        model.train()
        epoch_loss = 0.0
        for x in dataloader:
            x = x.to(device)
            emb_anchor, emb_positive = model(x)
            loss = criterion(emb_anchor, emb_positive)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            
        epoch_loss_avg = epoch_loss / len(dataloader)
        epochs_iter.set_postfix({"loss": f"{epoch_loss_avg:.4f}"})

    # --- Extract Embeddings ---
    print("Extracting global SCARF embeddings...")
    eval_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for x in tqdm(eval_dataloader, desc="Extracting"):
            x = x.to(device)
            emb = model.get_embeddings(x)
            all_embeddings.append(emb.cpu().numpy())
            
    embeddings = np.concatenate(all_embeddings, axis=0)

    # --- Save per sample ---
    print("Saving embeddings per sample...")
    for idx_info in sample_indices:
        start = idx_info['start']
        end = idx_info['end']
        sample_emb = embeddings[start:end]
        
        emb_columns = [f"SCARF_Dim_{i+1}" for i in range(sample_emb.shape[1])]
        df_embeddings = pd.DataFrame(sample_emb, columns=emb_columns)
        
        sample_output_dir = idx_info['output_dir']
        os.makedirs(sample_output_dir, exist_ok=True)
        filename = f"{idx_info['label']}_scarf_embeddings.csv"
        out_path = os.path.join(sample_output_dir, filename)
        
        df_embeddings.to_csv(out_path, index=False)
        print(f"  Saved {filename} ({len(sample_emb)} events) to {sample_output_dir}")

    print("Global SCARF Pipeline run complete!")


def main():
    parser = argparse.ArgumentParser(description='SCARF global training and embedding extraction')
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
    parser.add_argument('--embedding-dim', type=int, default=16, help='Embedding dimension d for each numerical feature in PLR')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()

    print("Scanning directories for matching files based on config.py...")
    samples = discover_samples()
    if not samples:
        print("No matching CSV and FCS samples found. Please run the pipeline first to generate CSVs.")
        sys.exit(1)
        
    print(f"Discovered {len(samples)} sample(s) to process.")
    
    run_global_scarf_pipeline(
        samples=samples,
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
        embedding_dim=args.embedding_dim,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
