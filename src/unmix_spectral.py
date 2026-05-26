import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_spectral_features(df):
    """Get spectral channels excluding 638.6nm"""
    return [c for c in df.columns if c.startswith('Area_') and c.endswith('nm') and '638.6nm' not in c]

def compute_reference_spectra(neg_csv_paths, stain_csv_paths):
    """
    Compute reference spectra for Negative and Stained samples.
    """
    print("Computing Reference Spectra...")
    
    # --- 1. Autofluorescence (AF) Reference ---
    af_spectra_list = []
    for path in neg_csv_paths:
        df = pd.read_csv(path)
        wl_features = get_spectral_features(df)
        af_spectra_list.append(df[wl_features].values)
    
    X_neg_all = np.vstack(af_spectra_list)
    S_AF = np.median(X_neg_all, axis=0)
    S_AF = S_AF / np.sum(S_AF)
    
    # --- 2. Stain (e.g. PI) Reference ---
    stain_spectra_list = []
    for path in stain_csv_paths:
        df = pd.read_csv(path)
        wl_features = get_spectral_features(df)
        stain_spectra_list.append(df[wl_features].values)
    
    X_stain_all = np.vstack(stain_spectra_list)
    
    # Detect the peak emission channel automatically
    total_intensity = np.sum(X_stain_all, axis=1)
    bright_cells_total = X_stain_all[total_intensity >= np.percentile(total_intensity, 99)]
    S_bright_total = np.median(bright_cells_total, axis=0)
    ratios = S_bright_total / (S_AF + 1e-9)
    peak_idx = np.argmax(ratios)
    
    # Use top 2% of cells in the peak channel as positive for stain
    peak_values = X_stain_all[:, peak_idx]
    stained_cells = X_stain_all[peak_values >= np.percentile(peak_values, 98)]
    
    S_Stain = np.median(stained_cells, axis=0)
    # Subtract AF component
    S_Stain = np.maximum(S_Stain - (S_AF * np.min(S_Stain / (S_AF + 1e-9))), 0)
    S_Stain = S_Stain / np.sum(S_Stain)
    
    return S_AF, S_Stain

def save_unmixing_plot(af_vals, stain_vals, stain_name, plot_path, title):
    """Save 2D scatter plot of unmixing results with coolwarm colormap"""
    af_plot = np.arcsinh(af_vals / 150.0)
    stain_plot = np.arcsinh(stain_vals / 150.0)
    
    plt.figure(figsize=(6.5, 5.5), dpi=150)
    sc = plt.scatter(af_plot, stain_plot, s=2, alpha=0.3, c=stain_plot, cmap='coolwarm', edgecolors='none')
    plt.colorbar(sc, label=f'Unmixed {stain_name} (ArcSinh)')
    plt.xlabel("Unmixed Autofluorescence (ArcSinh)")
    plt.ylabel(f"Unmixed {stain_name} (ArcSinh)")
    plt.title(f"Spectral Unmixing Result\n{title}")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

def run_unmixing_group(results_base_dir, stain_name="PI"):
    """
    Run unmixing for one experimental group.
    """
    # Find Negative sample CSVs (case insensitive)
    neg_csv_paths = glob.glob(os.path.join(results_base_dir, "Negative_*", "*.csv")) + \
                    glob.glob(os.path.join(results_base_dir, "negative_*", "*.csv"))
    # Find Stain sample CSVs (case insensitive)
    stain_csv_paths = glob.glob(os.path.join(results_base_dir, f"{stain_name}_*", "*.csv")) + \
                      glob.glob(os.path.join(results_base_dir, f"{stain_name.lower()}_*", "*.csv")) + \
                      glob.glob(os.path.join(results_base_dir, f"{stain_name.upper()}_*", "*.csv"))
    
    # Exclude SCARF embeddings and duplicates
    neg_csv_paths = sorted(list(set([p for p in neg_csv_paths if "scarf_embeddings" not in p])))
    stain_csv_paths = sorted(list(set([p for p in stain_csv_paths if "scarf_embeddings" not in p])))
    
    if not neg_csv_paths or not stain_csv_paths:
        print("    Warning: Missing CSV files for unmixing.")
        return
        
    # Calculate reference spectra
    S_AF, S_Stain = compute_reference_spectra(neg_csv_paths, stain_csv_paths)
    
    # Create S matrix and calculate pseudoinverse
    S = np.column_stack((S_AF, S_Stain))
    pinv_S = np.linalg.pinv(S)
    
    # Store unmixed data
    unmixed_neg = {}
    unmixed_stain = {}
    
    for path in neg_csv_paths:
        df = pd.read_csv(path)
        wl_features = get_spectral_features(df)
        unmixed_neg[path] = df[wl_features].values @ pinv_S.T
        
    for path in stain_csv_paths:
        df = pd.read_csv(path)
        wl_features = get_spectral_features(df)
        unmixed_stain[path] = df[wl_features].values @ pinv_S.T
        
    # Calculate AF leakage slope using Negative controls
    neg_C_all = np.vstack(list(unmixed_neg.values()))
    slope = neg_C_all[:, 1].mean() / neg_C_all[:, 0].mean()
    print(f"    Calculated autoflour leakage slope: {slope:.6f}")
    
    # Apply leakage correction, background subtraction, and save
    # 1. Negative Control
    for path in neg_csv_paths:
        C = unmixed_neg[path]
        C_no_slope = C[:, 1] - slope * C[:, 0]
        bg = np.median(C_no_slope)
        
        df = pd.read_csv(path)
        df['Unmixed_AF'] = np.maximum(C[:, 0], 0)
        df[f'Unmixed_{stain_name}'] = np.maximum(C_no_slope - bg, 0)
        df.to_csv(path, index=False)
        
        plot_path = os.path.join(os.path.dirname(path), "unmixing_scatter.png")
        save_unmixing_plot(df['Unmixed_AF'].values, df[f'Unmixed_{stain_name}'].values, stain_name, plot_path, os.path.basename(path))
        print(f"    Unmixing complete: {os.path.basename(path)}")
        
    # 2. Stained Samples
    for path in stain_csv_paths:
        C = unmixed_stain[path]
        C_no_slope = C[:, 1] - slope * C[:, 0]
        bg = np.median(C_no_slope)
        
        df = pd.read_csv(path)
        df['Unmixed_AF'] = np.maximum(C[:, 0], 0)
        df[f'Unmixed_{stain_name}'] = np.maximum(C_no_slope - bg, 0)
        df.to_csv(path, index=False)
        
        plot_path = os.path.join(os.path.dirname(path), "unmixing_scatter.png")
        save_unmixing_plot(df['Unmixed_AF'].values, df[f'Unmixed_{stain_name}'].values, stain_name, plot_path, os.path.basename(path))
        print(f"    Unmixing complete: {os.path.basename(path)}")
