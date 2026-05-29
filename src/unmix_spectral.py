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

class PoissonUnmixer:
    def __init__(self, max_iter=5, tol=1e-4):
        self.max_iter = max_iter
        self.tol = tol
        self.S_AF = None
        self.S_Stain = None
        self.S = None
        self.slope = 0.0
        self.bg = 0.0

    def fit(self, X_neg, X_stain):
        # 1. Autofluorescence (AF) Reference
        self.S_AF = np.median(X_neg, axis=0)
        self.S_AF = self.S_AF / (np.sum(self.S_AF) + 1e-9)
        
        # 2. Stain Reference
        total_intensity = np.sum(X_stain, axis=1)
        bright_cells_total = X_stain[total_intensity >= np.percentile(total_intensity, 99)]
        if len(bright_cells_total) == 0:
            bright_cells_total = X_stain
            
        S_bright_total = np.median(bright_cells_total, axis=0)
        ratios = S_bright_total / (self.S_AF + 1e-9)
        peak_idx = np.argmax(ratios)
        
        peak_values = X_stain[:, peak_idx]
        stained_cells = X_stain[peak_values >= np.percentile(peak_values, 98)]
        if len(stained_cells) == 0:
            stained_cells = X_stain
            
        # ====== S_Stain estimation: iterative refinement ======
        # Step A: Initial rough S_Stain (simple subtraction, top 2% bright cells)
        S_AF_unnorm = np.median(X_neg, axis=0)
        median_stained = np.median(stained_cells, axis=0)
        S_Stain_init = median_stained - S_AF_unnorm  # simple 1x subtraction
        S_Stain_init = np.maximum(S_Stain_init, 0)
        S_Stain_init = S_Stain_init / (np.sum(S_Stain_init) + 1e-9)
        self.S_Stain = S_Stain_init
        self.S = np.column_stack((self.S_AF, self.S_Stain))
        
        # Step B: Run sequential unmixing with rough S_Stain on ALL stained cells
        C_init = self._unmix_sequential(X_stain)
        c_af_all = C_init[:, 0]
        c_stain_all = C_init[:, 1]
        
        # Step C: Refine S_Stain using fitted coefficients
        # For cells with significant Calcein (c_stain > 0), compute:
        #   S_Stain_i = (X_i - c_af_i * S_AF) / c_stain_i
        # Then take the median across cells
        valid = c_stain_all > np.percentile(c_stain_all, 50)  # use top 50% by c_stain
        X_valid = X_stain[valid]
        c_af_valid = c_af_all[valid]
        c_stain_valid = c_stain_all[valid]
        
        # Compute per-cell stain spectrum estimate
        residuals = X_valid - c_af_valid[:, None] * self.S_AF[None, :]
        per_cell_stain = residuals / (c_stain_valid[:, None] + 1e-9)
        
        S_Stain = np.median(per_cell_stain, axis=0)
        S_Stain = np.maximum(S_Stain, 0)
        self.S_Stain = S_Stain / (np.sum(S_Stain) + 1e-9)
        
        self.S = np.column_stack((self.S_AF, self.S_Stain))
        
        # 3. Calculate leakage slope and background using negative control
        C_neg = self._unmix_sequential(X_neg)
        
        # slopeがマイナスになるのを防ぐセーフティ
        raw_slope = C_neg[:, 1].mean() / (C_neg[:, 0].mean() + 1e-9)
        self.slope = max(raw_slope, 0.0)
        
        C_no_slope = C_neg[:, 1] - self.slope * C_neg[:, 0]
        self.bg = np.median(C_no_slope)
        
        return self

    def _unmix_sequential(self, X):
        """Sequential estimation: tail→c_af, then peak→c_stain.
        
        Joint fitting (OLS/IRLS) over all channels is biased because
        S_AF and S_Stain overlap at 500-540nm. The Calcein peak dominates,
        causing c_af to be severely underestimated and c_stain inflated.
        
        Instead:
          1. Estimate c_af from tail channels (>650nm) where S_Stain ≈ 0
          2. Subtract c_af * S_AF from the full spectrum
          3. Estimate c_stain from the residual at peak channels
        """
        N = X.shape[0]
        S_AF = self.S_AF
        S_Stain = self.S_Stain
        
        # Channel masks (based on index into wl_features)
        # Tail: last ~8 channels (roughly >650nm, where Calcein emission is zero)
        # Peak: channels where S_Stain has significant signal (top channels by S_Stain value)
        stain_threshold = 0.3 * np.max(S_Stain)  # channels with >30% of peak S_Stain
        peak_mask = S_Stain >= stain_threshold
        
        # Tail: channels where S_Stain is negligible AND not in peak region
        # Use the last 8 channels as the tail
        tail_mask = np.zeros(len(S_AF), dtype=bool)
        tail_mask[-8:] = True
        
        # Step 1: c_af from tail channels only
        S_AF_tail = S_AF[tail_mask]
        X_tail = X[:, tail_mask]
        denom_af = np.dot(S_AF_tail, S_AF_tail) + 1e-9
        c_af = X_tail @ S_AF_tail / denom_af  # (N,)
        
        # Step 2: subtract AF, then estimate c_stain from peak channels
        residual = X - c_af[:, None] * S_AF[None, :]  # (N, M)
        S_Stain_peak = S_Stain[peak_mask]
        R_peak = residual[:, peak_mask]
        denom_stain = np.dot(S_Stain_peak, S_Stain_peak) + 1e-9
        c_stain = R_peak @ S_Stain_peak / denom_stain  # (N,)
        
        C = np.column_stack((c_af, c_stain))
        return C

    def transform(self, X):
        C = self._unmix_sequential(X)
        C_af = C[:, 0]
        C_stain = C[:, 1]
        C_stain_corrected = C_stain - self.slope * C_af - self.bg
        return C_af, C_stain_corrected

    def get_raw_coefficients(self, X):
        return self._unmix_sequential(X)

    def remove_stain_component(self, X):
        C = self._unmix_sequential(X)
        return X - C[:, 1][:, None] * self.S_Stain[None, :]

def run_unmixing_group(results_base_dir, stain_name="PI"):
    neg_csv_paths = glob.glob(os.path.join(results_base_dir, "Negative_*", "*.csv")) + \
                    glob.glob(os.path.join(results_base_dir, "negative_*", "*.csv"))
    stain_csv_paths = glob.glob(os.path.join(results_base_dir, f"{stain_name}_*", "*.csv")) + \
                      glob.glob(os.path.join(results_base_dir, f"{stain_name.lower()}_*", "*.csv")) + \
                      glob.glob(os.path.join(results_base_dir, f"{stain_name.upper()}_*", "*.csv"))
    
    neg_csv_paths = sorted(list(set([p for p in neg_csv_paths if "scarf_embeddings" not in p])))
    stain_csv_paths = sorted(list(set([p for p in stain_csv_paths if "scarf_embeddings" not in p])))
    
    if not neg_csv_paths or not stain_csv_paths:
        print("    Warning: Missing CSV files for unmixing.")
        return
        
    print("Loading unmixing data for fit...")
    X_neg_all = []
    for path in neg_csv_paths:
        df = pd.read_csv(path)
        wl_features = get_spectral_features(df)
        X_neg_all.append(df[wl_features].values)
    X_neg_all = np.vstack(X_neg_all)
    
    X_stain_all = []
    for path in stain_csv_paths:
        df = pd.read_csv(path)
        wl_features = get_spectral_features(df)
        X_stain_all.append(df[wl_features].values)
    X_stain_all = np.vstack(X_stain_all)
    
    unmixer = PoissonUnmixer()
    unmixer.fit(X_neg_all, X_stain_all)
    print(f"    Calculated autoflour leakage slope: {unmixer.slope:.6f}")
    
    for path in neg_csv_paths + stain_csv_paths:
        df = pd.read_csv(path)
        wl_features = get_spectral_features(df)
        af, stain_corr = unmixer.transform(df[wl_features].values)
        
        df['Unmixed_AF'] = np.maximum(af, 0)
        df[f'Unmixed_{stain_name}'] = np.maximum(stain_corr, 0)
        
        # De-fragmentation warning workaround
        df = df.copy()
        df.to_csv(path, index=False)
        
        plot_path = os.path.join(os.path.dirname(path), "unmixing_scatter.png")
        save_unmixing_plot(df['Unmixed_AF'].values, df[f'Unmixed_{stain_name}'].values, stain_name, plot_path, os.path.basename(path))
        print(f"    Unmixing complete: {os.path.basename(path)}")
