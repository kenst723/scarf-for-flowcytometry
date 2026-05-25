"""
Sony Spectral Flow Cytometer .sraw file parser and analyzer.

File Structure (reverse-engineered):
    Global Header (12 bytes):
        - uint32: version/magic (0x00010000)
        - uint32: num_events
        - uint32: num_measurement_types (3: Area, Height, Width)
    
    Type Name Table (num_types x 256 bytes):
        - Null-padded ASCII strings: "Intensity - Area", "Intensity - Height", "Intensity - Width"
    
    Channel Count Table (num_types x 4 bytes):
        - uint32 per type: number of spectral channels (typically 34)
    
    Channel Name Table (num_types x num_channels x 256 bytes):
        - Null-padded ASCII strings: "V1", "V2", "1", "2", ..., "32"
    
    Wavelength Table (num_types x num_channels x 4 bytes):
        - float32 per channel: wavelength in nm (repeated for each type)
    
    Event Data (num_types x (num_events + 2) x num_channels x 4 bytes):
        - For each measurement type:
            - 2 reference/background rows (34 float32 values each)
            - num_events rows of spectral data (34 float32 values each)
"""

import matplotlib
matplotlib.use('Agg')
import struct
import os
import numpy as np
import pandas as pd


def parse_sraw(filepath):
    """
    Parse a Sony .sraw spectral flow cytometry file.
    
    Returns:
        dict with keys:
            - 'wavelengths': np.array of wavelengths (nm)
            - 'num_events': int
            - 'num_channels': int
            - 'type_names': list of str
            - 'channel_names': list of str
            - 'reference': dict of {type_name: np.array (2, num_channels)}
            - 'data': dict of {type_name: np.array (num_events, num_channels)}
    """
    with open(filepath, 'rb') as f:
        raw = f.read()
    
    file_size = len(raw)
    
    # Global header
    version = struct.unpack_from('<I', raw, 0)[0]
    num_events = struct.unpack_from('<I', raw, 4)[0]
    num_types = struct.unpack_from('<I', raw, 8)[0]
    
    # Type names (256-byte null-padded strings)
    type_names = []
    for i in range(num_types):
        offset = 12 + i * 256
        name = raw[offset:offset+256].split(b'\x00')[0].decode('ascii', errors='replace')
        type_names.append(name)
    
    # Channel counts per type
    ch_count_offset = 12 + num_types * 256
    ch_counts = []
    for i in range(num_types):
        val = struct.unpack_from('<I', raw, ch_count_offset + i * 4)[0]
        ch_counts.append(val)
    
    num_channels = ch_counts[0]
    
    # Channel names (num_types sets of num_channels x 256-byte strings)
    ch_names_offset = ch_count_offset + num_types * 4
    channel_names = []
    for i in range(num_channels):
        offset = ch_names_offset + i * 256
        name = raw[offset:offset+256].split(b'\x00')[0].decode('ascii', errors='replace')
        channel_names.append(name)
    
    # Wavelength tables (num_types x num_channels x float32, all identical)
    wl_offset = ch_names_offset + num_types * num_channels * 256
    wavelengths = np.frombuffer(raw[wl_offset:wl_offset + num_channels * 4], dtype='<f4').copy()
    
    # Event data
    data_offset = wl_offset + num_types * num_channels * 4
    all_floats = np.frombuffer(raw[data_offset:], dtype='<f4')
    
    rows_per_type = num_events + 2  # 2 reference rows + num_events data rows
    stride = rows_per_type * num_channels
    
    reference = {}
    data = {}
    for t in range(num_types):
        base = t * stride
        type_data = all_floats[base:base + stride].reshape(rows_per_type, num_channels)
        reference[type_names[t]] = type_data[:2].copy()
        data[type_names[t]] = type_data[2:].copy()
    
    return {
        'wavelengths': wavelengths,
        'num_events': num_events,
        'num_channels': num_channels,
        'type_names': type_names,
        'channel_names': channel_names,
        'reference': reference,
        'data': data,
    }


def print_summary(result):
    """Print a summary of parsed .sraw data."""
    print(f"Events: {result['num_events']}")
    print(f"Spectral Channels: {result['num_channels']}")
    print(f"Measurement Types: {result['type_names']}")
    print(f"Channel Names: {result['channel_names']}")
    print(f"Wavelength Range: {result['wavelengths'][0]:.1f} - {result['wavelengths'][-1]:.1f} nm")
    print(f"Wavelengths: {[f'{w:.1f}' for w in result['wavelengths']]}")
    
    for name in result['type_names']:
        d = result['data'][name]
        ref = result['reference'][name]
        print(f"\n{name}:")
        print(f"  Data shape: {d.shape}")
        print(f"  Range: [{d.min():.2f}, {d.max():.2f}]")
        print(f"  Mean: {d.mean():.2f}, Std: {d.std():.2f}")
        print(f"  Reference row 0: [{', '.join(f'{v:.1f}' for v in ref[0, :5])}] ...")
        print(f"  Reference row 1: [{', '.join(f'{v:.1f}' for v in ref[1, :5])}] ...")


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    sraw_dir = r'C:\PythonProject\data\Experiment 2026!05!19 9!16\24 Tube Rack (5mL) - 1\negative'
    
    files = [
        ('A02 Well - A02.sraw', 'A02'),
        ('A03 Well - A03.sraw', 'A03'),
    ]
    
    for filename, label in files:
        filepath = os.path.join(sraw_dir, filename)
        print(f"\n{'='*60}")
        print(f"Parsing: {filename}")
        print(f"{'='*60}")
        result = parse_sraw(filepath)
        print_summary(result)
    
    # --- Visualization ---
    # Use first file (A02) for detailed plots
    filepath = os.path.join(sraw_dir, files[0][0])
    result = parse_sraw(filepath)
    
    wavelengths = result['wavelengths']
    area_data = result['data']['Intensity - Area']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Sony Spectral Flow Cytometry - .sraw Analysis (A02, Negative Control)', 
                 fontsize=14, fontweight='bold')
    
    # 1. Overlay of individual spectra (first 200 events, semi-transparent)
    ax = axes[0, 0]
    for i in range(min(200, area_data.shape[0])):
        ax.plot(wavelengths, area_data[i], alpha=0.05, color='steelblue', linewidth=0.5)
    # Mean spectrum
    mean_spectrum = area_data.mean(axis=0)
    ax.plot(wavelengths, mean_spectrum, color='darkred', linewidth=2, label='Mean')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity (Area)')
    ax.set_title('Individual Spectra (first 200 events)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Mean spectrum with std shading for all 3 types
    ax = axes[0, 1]
    colors = ['#2196F3', '#FF5722', '#4CAF50']
    for t_idx, t_name in enumerate(result['type_names']):
        d = result['data'][t_name]
        mean = d.mean(axis=0)
        std = d.std(axis=0)
        ax.plot(wavelengths, mean, color=colors[t_idx], linewidth=2, label=t_name)
        ax.fill_between(wavelengths, mean - std, mean + std, color=colors[t_idx], alpha=0.15)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity')
    ax.set_title('Mean ± Std for Each Measurement Type')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 3. Heatmap of spectral data (Area)
    ax = axes[1, 0]
    # Sample every 10th event for visibility
    sample_step = max(1, area_data.shape[0] // 500)
    sampled = area_data[::sample_step]
    im = ax.imshow(sampled, aspect='auto', cmap='inferno',
                   extent=[wavelengths[0], wavelengths[-1], sampled.shape[0]*sample_step, 0])
    fig.colorbar(im, ax=ax, label='Intensity (Area)')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Event Index')
    ax.set_title('Spectral Heatmap (Area)')
    
    # 4. Reference spectrum comparison
    ax = axes[1, 1]
    ref = result['reference']['Intensity - Area']
    ax.plot(wavelengths, ref[0], 'b-', linewidth=2, marker='o', markersize=3, label='Reference Row 0')
    ax.plot(wavelengths, ref[1], 'r--', linewidth=2, marker='s', markersize=3, label='Reference Row 1')
    ax.plot(wavelengths, mean_spectrum, 'g-', linewidth=1.5, alpha=0.7, label='Mean Event Spectrum')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity (Area)')
    ax.set_title('Reference vs Mean Event Spectrum')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(r'C:\PythonProject\sraw_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: C:\\PythonProject\\sraw_analysis.png")
    plt.show()
