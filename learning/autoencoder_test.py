import os
import sys
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import copy

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.unmix_spectral import get_spectral_features
from src.unmix_autoencoder import AF_AutoEncoder

# -----------------------------------------
# Main Script
# -----------------------------------------
def main():
    exp_dir = r"C:\PythonProject\analysis\results\Experiment 2026!06!02 12!39"
    
    neg_files = glob.glob(os.path.join(exp_dir, "Negative_*", "*.csv"))
    cal_files = glob.glob(os.path.join(exp_dir, "Calcein_*", "*.csv"))
    
    if not neg_files or not cal_files:
        print("Missing CSV files!")
        return

    print("Loading data...")
    df_neg = pd.read_csv(neg_files[0])
    df_cal = pd.read_csv(cal_files[0])
    
    wl_features = get_spectral_features(df_neg)
    wl_numeric = np.array([float(f.replace('Area_', '').replace('nm', '')) for f in wl_features])
    
    X_neg_raw = df_neg[wl_features].values
    X_cal_raw = df_cal[wl_features].values
    
    cofactor = 150.0
    X_neg_arcsinh = np.arcsinh(X_neg_raw / cofactor)
    X_cal_arcsinh = np.arcsinh(X_cal_raw / cofactor)
    
    val_min = np.min(X_neg_arcsinh)
    global_max = max(np.max(X_neg_arcsinh), np.max(X_cal_arcsinh))
    
    X_neg_scaled = (X_neg_arcsinh - val_min) / (global_max - val_min)
    X_cal_scaled = (X_cal_arcsinh - val_min) / (global_max - val_min)
    
    X_neg_scaled = np.clip(X_neg_scaled, 0, 1)
    X_cal_scaled = np.clip(X_cal_scaled, 0, 1)
    
    tensor_neg = torch.FloatTensor(X_neg_scaled)
    tensor_cal = torch.FloatTensor(X_cal_scaled)
    
    dataset = TensorDataset(tensor_neg, tensor_neg)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model = AF_AutoEncoder(input_dim=len(wl_features))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training AutoEncoder on Negative (Unstained) data...")
    epochs = 100
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, _ in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
    
    print("Training Complete.")
    
    model.eval()
    with torch.no_grad():
        pred_neg_scaled = model(tensor_neg).numpy()
        pred_cal_scaled = model(tensor_cal).numpy()
        
    def inverse_transform(X_scaled):
        X_arcsinh = X_scaled * (global_max - val_min) + val_min
        return np.sinh(X_arcsinh) * cofactor
        
    pred_neg_raw = inverse_transform(pred_neg_scaled)
    pred_cal_raw = inverse_transform(pred_cal_scaled)
    
    pure_calcein = X_cal_raw - pred_cal_raw
    # Clip negative residuals
    pure_calcein = np.maximum(pure_calcein, 1e-1)
    
    # Plotting
    print("Generating result plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    
    # 1. Negative Reconstruction
    ax = axes[0, 0]
    for i in range(50):
        ax.plot(wl_numeric, X_neg_raw[i], color='gray', alpha=0.1)
        ax.plot(wl_numeric, pred_neg_raw[i], color='blue', alpha=0.1)
    ax.plot([], [], color='gray', label='Original (Negative)')
    ax.plot([], [], color='blue', label='Reconstructed (AE)')
    ax.set_title("Negative Control: AutoEncoder Reconstruction")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity")
    ax.legend()
    
    # 2. Calcein Input vs Predicted AF
    ax = axes[0, 1]
    for i in range(50):
        ax.plot(wl_numeric, X_cal_raw[i], color='orange', alpha=0.1)
        ax.plot(wl_numeric, pred_cal_raw[i], color='green', alpha=0.1)
    ax.plot([], [], color='orange', label='Original (Stained Input)')
    ax.plot([], [], color='green', label='Predicted AF (AE Output)')
    ax.set_title("Stained Sample: Predicting Autofluorescence")
    ax.set_xlabel("Wavelength (nm)")
    ax.legend()
    
    # 3. Pure Calcein (Line Plot)
    ax = axes[1, 0]
    for i in range(50):
        ax.plot(wl_numeric, pure_calcein[i], color='red', alpha=0.1)
    
    ref_calcein = np.median(pure_calcein, axis=0)
    ref_calcein = ref_calcein / np.max(ref_calcein) * np.max(pure_calcein[:50])
    ax.plot(wl_numeric, ref_calcein, color='black', linewidth=2, label='Median Pure Stain')
    ax.set_title("Extracted Pure Calcein (Input - Predicted AF)")
    ax.set_xlabel("Wavelength (nm)")
    ax.legend()
    
    # 4. Pure Calcein (Spectral Density Plot)
    ax = axes[1, 1]
    num_channels = len(wl_numeric)
    
    intensity_min = max(pure_calcein.min(), 1e-1)
    intensity_max = pure_calcein.max() * 1.5
    num_intensity_bins = 256
    intensity_bins = np.logspace(np.log10(intensity_min), np.log10(intensity_max), num_intensity_bins + 1)
    
    density = np.zeros((num_intensity_bins, num_channels))
    for i in range(num_channels):
        hist, _ = np.histogram(pure_calcein[:, i], bins=intensity_bins)
        density[:, i] = hist
        
    density_masked = np.ma.masked_where(density == 0, density)
    cmap = copy.copy(plt.get_cmap('jet'))
    cmap.set_bad(color='white')
    
    # Plot using pcolormesh
    im = ax.pcolormesh(np.arange(num_channels + 1), intensity_bins, density_masked,
                       cmap=cmap, norm=LogNorm(vmin=1, vmax=density_masked.max()))
    
    ax.set_yscale('log')
    ax.set_ylim(bottom=max(10, intensity_min))
    
    # Set xticks to match wavelength
    tick_indices = np.arange(0, num_channels, max(1, num_channels // 6))
    ax.set_xticks(tick_indices + 0.5)
    ax.set_xticklabels([f"{wl_numeric[i]:.0f}" for i in tick_indices])
    
    ax.set_title("Spectral Density Plot of Pure Calcein")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity (Log Scale)")
    fig.colorbar(im, ax=ax, label='Event Count (Log Scale)')
    
    plt.tight_layout()
    
    out_png = os.path.join(exp_dir, "ae_test_density_results.png")
    plt.savefig(out_png)
    print(f"Saved results to {out_png}")

    # New Plot: Density plots of True Negative vs Predicted AF
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    
    def plot_density(ax, data, title):
        num_channels = data.shape[1]
        intensity_min = max(data.min(), 1e-1)
        intensity_max = data.max() * 1.5
        intensity_bins = np.logspace(np.log10(intensity_min), np.log10(intensity_max), 257)
        density = np.zeros((256, num_channels))
        for i in range(num_channels):
            hist, _ = np.histogram(data[:, i], bins=intensity_bins)
            density[:, i] = hist
        
        density_masked = np.ma.masked_where(density == 0, density)
        cmap = copy.copy(plt.get_cmap('jet'))
        cmap.set_bad(color='white')
        im = ax.pcolormesh(np.arange(num_channels + 1), intensity_bins, density_masked,
                           cmap=cmap, norm=LogNorm(vmin=1, vmax=density_masked.max()))
        ax.set_yscale('log')
        ax.set_ylim(bottom=max(10, intensity_min))
        tick_indices = np.arange(0, num_channels, max(1, num_channels // 6))
        ax.set_xticks(tick_indices + 0.5)
        ax.set_xticklabels([f"{wl_numeric[i]:.0f}" for i in tick_indices])
        ax.set_title(title)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity (Log Scale)")
        plt.colorbar(im, ax=ax, label='Event Count (Log Scale)')

    plot_density(axes2[0], X_neg_raw, "True Autofluorescence (Negative Control)")
    plot_density(axes2[1], pred_cal_raw, "Predicted Autofluorescence (from Stained Cells)")
    
    out_png2 = os.path.join(exp_dir, "ae_predicted_af_density.png")
    plt.tight_layout()
    plt.savefig(out_png2)
    print(f"Saved predicted AF density to {out_png2}")

if __name__ == "__main__":
    main()
