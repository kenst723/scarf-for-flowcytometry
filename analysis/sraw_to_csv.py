import struct
import os
import numpy as np
import pandas as pd

def parse_sraw_interleaved(filepath):
    with open(filepath, 'rb') as f:
        raw = f.read()
    
    num_events = struct.unpack_from('<I', raw, 4)[0]
    num_types = struct.unpack_from('<I', raw, 8)[0]  # 3 (Area, Height, Width)
    
    # Extract channel names (assume they start around offset 792 for the first type)
    channel_names = []
    # 780: 3 uint32 channel counts, normally 34
    num_channels = struct.unpack_from('<I', raw, 780)[0]
    
    ch_names_start = 792
    for i in range(num_channels):
        offset = ch_names_start + i * 256
        name = raw[offset:offset+256].split(b'\x00')[0].decode('ascii', errors='replace')
        channel_names.append(name)
        
    # Extract wavelengths
    wl_offset = 26904 # Known from previous analysis
    wavelengths = np.frombuffer(raw[wl_offset:wl_offset + num_channels * 4], dtype='<f4')
    
    # Event data starts at 27312
    event_data_start = 27312
    all_floats = np.frombuffer(raw[event_data_start:], dtype='<f4')
    
    # Reshape as interleaved: (TotalEvents, 3 types, 34 channels)
    total_events = len(all_floats) // (num_types * num_channels)
    data_reshaped = all_floats.reshape((total_events, num_types, num_channels))
    
    # The first 2 events might be reference/baseline, and the remaining 10000 are actual events
    actual_data = data_reshaped[-num_events:]
    
    return {
        'num_events': num_events,
        'num_channels': num_channels,
        'channel_names': channel_names,
        'wavelengths': wavelengths,
        'data': actual_data  # Shape: (10000, 3, 34)
    }

SRAW_DIR = r"/data/Experiment 2026!05!21 15!59/24 Tube Rack (5mL) - 1/PI"
sraw_files = [f for f in os.listdir(SRAW_DIR) if f.endswith('.sraw')]

for filename in sraw_files:
    filepath = os.path.join(SRAW_DIR, filename)
    result = parse_sraw_interleaved(filepath)
    
    num_events = result['num_events']
    num_channels = result['num_channels']
    channel_names = result['channel_names']
    wavelengths = result['wavelengths']
    data = result['data'] # (10000, 3, 34)
    
    # ---------------------------------------------------------
    # Calculate Wavelength Bandwidths (for normalization)
    # ---------------------------------------------------------
    # Sony uses a fixed array of multipliers derived from 8.0 / dLambda
    # These values perfectly match the user's wave-channel.csv data
    multipliers = np.array([
        2.352933, 2.352944, 2.352939, 2.162156, 2.105267, 1.904761, 1.860462, 1.702126, 
        1.632656, 1.509433, 1.428573, 1.333333, 1.269838, 1.212123, 1.142858, 1.066665, 
        1.025644, 0.97561, 0.909088, 0.869567, 0.816325, 0.769229, 0.727273, 0.68968, 
        0.655738, 0.625005, 0.601506, 0.575537, 0.555556, 0.529801, 0.503132, 0.484842, 
        0.45715, 0.439574
    ])

    # Normalize data for Wavelength series
    # data is shape (10000, 3, 34). We multiply along the last axis by multipliers.
    wavelength_data = data * multipliers
    
    # ---------------------------------------------------------
    # Create columns as requested
    # ---------------------------------------------------------
    columns = []
    
    # 1. Area(Channel)
    for ch in channel_names: columns.append(f'Area_{ch}')
    # 2. Height(Channel)
    for ch in channel_names: columns.append(f'Height_{ch}')
    # 3. Width(Channel)
    for ch in channel_names: columns.append(f'Width_{ch}')
    
    # 4. Area(Wavelength)
    for wl in wavelengths: columns.append(f'Area_{wl:.1f}nm')
    # 5. Height(Wavelength)
    for wl in wavelengths: columns.append(f'Height_{wl:.1f}nm')
    # 6. Width(Wavelength)
    for wl in wavelengths: columns.append(f'Width_{wl:.1f}nm')
    
    # Extract Raw (Channel) data
    area_ch = data[:, 0, :]
    height_ch = data[:, 1, :]
    width_ch = data[:, 2, :]
    
    # Extract Normalized (Wavelength) data
    area_wl = wavelength_data[:, 0, :]
    height_wl = wavelength_data[:, 1, :]
    width_wl = wavelength_data[:, 2, :]
    
    # Concatenate all 6 series horizontally
    flat_data = np.hstack([area_ch, height_ch, width_ch, area_wl, height_wl, width_wl])
    
    df = pd.DataFrame(flat_data, columns=columns)
    df.insert(0, 'event_id', range(num_events))
    
    # Output to 'output' directory with datetime in filename
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = os.path.splitext(filename)[0]
    csv_name = f'{base_name}_{timestamp}.csv'
    csv_path = os.path.join(output_dir, csv_name)
    df.to_csv(csv_path, index=False)

    print(f'Processed {filename} -> {csv_name}')
    print(f'  Output: {csv_path}')
    print(f'  Shape: {df.shape}')
