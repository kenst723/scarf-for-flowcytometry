import os
import fcsparser
import pandas as pd
from datetime import datetime

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import RESULTS_DIR
from src.convert import convert_sraw_to_csv

def integrate_well(sraw_path, fcs_path):
    print(f"  Processing {os.path.basename(sraw_path)}...")
    csv_path, df_sraw = convert_sraw_to_csv(sraw_path)
    try:
        os.remove(csv_path)
    except:
        pass
    
    print(f"  Processing {os.path.basename(fcs_path)}...")
    _, df_fcs = fcsparser.parse(fcs_path, reformat_meta=True)
    
    if 'event_id' in df_sraw.columns:
        df_sraw = df_sraw.drop(columns=['event_id'])
        
    assert len(df_sraw) == len(df_fcs), f"Size mismatch: SRAW {len(df_sraw)} != FCS {len(df_fcs)}"
    
    df_combined = pd.concat([df_sraw, df_fcs], axis=1)
    return df_combined

def process_stain(experiment_dir, stain, well1, well2, output_dir, event_limit=None):
    print(f"Integrating {stain}: {well1} and {well2}...")
    stain_dir = os.path.join(experiment_dir, stain)
    
    sraw1 = os.path.join(stain_dir, f"{well1} Well - {well1}.sraw")
    fcs1 = os.path.join(stain_dir, f"{well1} Well - {well1}.fcs")
    df1 = integrate_well(sraw1, fcs1)
    
    sraw2 = os.path.join(stain_dir, f"{well2} Well - {well2}.sraw")
    fcs2 = os.path.join(stain_dir, f"{well2} Well - {well2}.fcs")
    df2 = integrate_well(sraw2, fcs2)
    
    print("  Concatenating horizontally and vertically...")
    df_final = pd.concat([df1, df2], axis=0, ignore_index=True)
    
    if event_limit and len(df_final) > event_limit:
        print(f"  Cutting events from {len(df_final)} to {event_limit}...")
        df_final = df_final.sample(n=event_limit, random_state=42).reset_index(drop=True)
        # Or df_final = df_final.iloc[:event_limit]
        
    df_final.insert(0, 'event_id', range(len(df_final)))
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_name = f"{well1} Well - {well1}_{timestamp}.csv"
    
    out_folder = os.path.join(output_dir, f"{stain}_{well1}")
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, csv_name)
    
    df_final.to_csv(out_path, index=False)
    print(f"Saved {stain} integrated data to {out_path} (shape: {df_final.shape})\n")

if __name__ == '__main__':
    base_dir = r"data\Experiment 2026!05!27 9!30\24 Tube Rack (5mL) - 1"
    out_dir = os.path.join(RESULTS_DIR, "2026-05-27")
    
    process_stain(base_dir, "Calcein", "A01", "A02", out_dir, event_limit=4900)
    process_stain(base_dir, "negative", "B01", "B02", out_dir)
