"""Check if .sraw constant-value rows correspond to specific FCS events."""
import numpy as np
import fcsparser
from utils.parse_sraw import parse_sraw

sraw_path = r'C:\PythonProject\data\Experiment 2026!05!19 9!16\24 Tube Rack (5mL) - 1\negative\A02 Well - A02.sraw'
fcs_path  = r'C:\PythonProject\data\Experiment 2026!05!19 9!16\24 Tube Rack (5mL) - 1\negative\A02 Well - A02.fcs'

result = parse_sraw(sraw_path)
area = result['data']['Intensity - Area']

meta, fcs = fcsparser.parse(fcs_path)

def is_const(row):
    return len(np.unique(np.round(row, 0))) <= 3

# Check FCS data at constant-row positions
print("=== FCS columns ===")
print(list(fcs.columns))

print("\n=== FCS data at constant vs spectral sraw rows ===")
print(f"{'idx':>4s}  {'sraw':>6s}  {'FSC-A':>10s}  {'SSC-A':>10s}  {'CalAM-A':>10s}  {'PI-A':>10s}")
for i in range(21):
    tag = "CONST" if is_const(area[i]) else "spec"
    fsc = fcs.iloc[i]['FSC - Area']
    ssc = fcs.iloc[i]['SSC - Area']
    cal = fcs.iloc[i]['Calcein AM - Area']
    pi_v = fcs.iloc[i]['PI - Area']
    print(f"{i:4d}  {tag:>6s}  {fsc:10.1f}  {ssc:10.1f}  {cal:10.1f}  {pi_v:10.1f}")

# Statistics: compare FSC-Area for constant vs spectral events
const_idx = [i for i in range(10000) if is_const(area[i])]
spec_idx  = [i for i in range(10000) if not is_const(area[i])]

fsc_const = fcs.iloc[const_idx]['FSC - Area'].values
fsc_spec  = fcs.iloc[spec_idx]['FSC - Area'].values

print(f"\n=== FSC-Area statistics ===")
print(f"  Constant rows ({len(const_idx)}): mean={fsc_const.mean():.1f}, std={fsc_const.std():.1f}, min={fsc_const.min():.1f}, max={fsc_const.max():.1f}")
print(f"  Spectral rows ({len(spec_idx)}): mean={fsc_spec.mean():.1f}, std={fsc_spec.std():.1f}, min={fsc_spec.min():.1f}, max={fsc_spec.max():.1f}")

ssc_const = fcs.iloc[const_idx]['SSC - Area'].values
ssc_spec  = fcs.iloc[spec_idx]['SSC - Area'].values
print(f"\n=== SSC-Area statistics ===")
print(f"  Constant rows ({len(const_idx)}): mean={ssc_const.mean():.1f}, std={ssc_const.std():.1f}")
print(f"  Spectral rows ({len(spec_idx)}): mean={ssc_spec.mean():.1f}, std={ssc_spec.std():.1f}")

# Check if constant rows have notably different FCS values
print(f"\n=== Are constant rows distinguishable in FCS data? ===")
print(f"  FSC-Area: const_mean={fsc_const.mean():.0f} vs spec_mean={fsc_spec.mean():.0f}")
print(f"  SSC-Area: const_mean={ssc_const.mean():.0f} vs spec_mean={ssc_spec.mean():.0f}")
