"""
Verify: Do Calcein cells REALLY have more AF, or is unmixing still wrong?

Key test: Compare RAW intensity at far-tail channels (700-780nm) where
Calcein absolutely cannot emit. If Calcein cells are truly 3x brighter
at these wavelengths, it's a genuine physical difference. If not, the
unmixing algorithm is still wrong.
"""
import os, sys, glob
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

def main():
    neg_dir = r"analysis\results\Experiment 2026!05!27 9!30\negative_B01"
    calcein_dir = r"analysis\results\Experiment 2026!05!27 9!30\Calcein_A01"

    neg_csv = glob.glob(os.path.join(neg_dir, "*.csv"))[0]
    calcein_csv = glob.glob(os.path.join(calcein_dir, "*.csv"))[0]

    df_neg = pd.read_csv(neg_csv)
    df_stain = pd.read_csv(calcein_csv)

    wl_features = [c for c in df_neg.columns if c.startswith('Area_') and c.endswith('nm') and '638.6nm' not in c]
    wl_values = [float(c.replace('Area_', '').replace('nm', '')) for c in wl_features]

    X_neg = df_neg[wl_features].values
    X_stain = df_stain[wl_features].values

    print("=" * 60)
    print("TEST 1: RAW intensity at far-tail channels (no unmixing)")
    print("  If Calcein cells are truly brighter here, it's physical.")
    print("  If not, the unmixing is still wrong.")
    print("=" * 60)
    
    # Far-tail channels where Calcein CANNOT emit
    far_tail_channels = [i for i, wl in enumerate(wl_values) if wl >= 700]
    print(f"\nFar-tail channels (>700nm):")
    for i in far_tail_channels:
        neg_med = np.median(X_neg[:, i])
        stain_med = np.median(X_stain[:, i])
        ratio = stain_med / (neg_med + 1e-9)
        print(f"  {wl_values[i]:>8.1f}nm: Neg={neg_med:>8.1f}  Calcein={stain_med:>8.1f}  Ratio={ratio:.2f}")
    
    # Also check at Calcein peak
    peak_channels = [i for i, wl in enumerate(wl_values) if 500 <= wl <= 520]
    print(f"\nCalcein peak channels (500-520nm) for reference:")
    for i in peak_channels:
        neg_med = np.median(X_neg[:, i])
        stain_med = np.median(X_stain[:, i])
        ratio = stain_med / (neg_med + 1e-9)
        print(f"  {wl_values[i]:>8.1f}nm: Neg={neg_med:>8.1f}  Calcein={stain_med:>8.1f}  Ratio={ratio:.2f}")

    print("\n" + "=" * 60)
    print("TEST 2: Cell size proxies (FSC/SSC if available)")
    print("=" * 60)
    
    # Check if FSC/SSC columns exist
    fsc_cols = [c for c in df_neg.columns if 'FSC' in c.upper() and 'Area' in c]
    ssc_cols = [c for c in df_neg.columns if 'SSC' in c.upper() and 'Area' in c]
    
    if fsc_cols:
        for col in fsc_cols[:2]:
            neg_med = df_neg[col].median()
            stain_med = df_stain[col].median() if col in df_stain.columns else float('nan')
            print(f"  {col}: Neg={neg_med:.1f}  Calcein={stain_med:.1f}  Ratio={stain_med/neg_med:.2f}")
    else:
        print("  No FSC columns found")
        
    if ssc_cols:
        for col in ssc_cols[:2]:
            neg_med = df_neg[col].median()
            stain_med = df_stain[col].median() if col in df_stain.columns else float('nan')
            print(f"  {col}: Neg={neg_med:.1f}  Calcein={stain_med:.1f}  Ratio={stain_med/neg_med:.2f}")
    else:
        print("  No SSC columns found")

    print("\n" + "=" * 60)
    print("TEST 3: Total intensity distribution comparison")
    print("=" * 60)
    
    # Compare total intensity across ALL channels (sum)
    total_neg = np.sum(X_neg, axis=1)
    total_stain = np.sum(X_stain, axis=1)
    
    # Compare total intensity at ONLY far-tail channels (pure AF)
    tail_neg = np.sum(X_neg[:, far_tail_channels], axis=1)
    tail_stain = np.sum(X_stain[:, far_tail_channels], axis=1)
    
    print(f"\nTotal intensity (all channels):")
    print(f"  Neg:     median={np.median(total_neg):>10.0f}  mean={np.mean(total_neg):>10.0f}")
    print(f"  Calcein: median={np.median(total_stain):>10.0f}  mean={np.mean(total_stain):>10.0f}")
    print(f"  Ratio (median): {np.median(total_stain)/np.median(total_neg):.2f}")
    
    print(f"\nFar-tail intensity only (>700nm, pure AF proxy):")
    print(f"  Neg:     median={np.median(tail_neg):>10.0f}  mean={np.mean(tail_neg):>10.0f}")
    print(f"  Calcein: median={np.median(tail_stain):>10.0f}  mean={np.mean(tail_stain):>10.0f}")
    print(f"  Ratio (median): {np.median(tail_stain)/np.median(tail_neg):.2f}")
    
    print(f"\n  Percentiles of far-tail intensity:")
    for p in [10, 25, 50, 75, 90]:
        neg_p = np.percentile(tail_neg, p)
        stain_p = np.percentile(tail_stain, p)
        print(f"    P{p:>2}: Neg={neg_p:>8.0f}  Calcein={stain_p:>8.0f}  Ratio={stain_p/neg_p:.2f}")

    print("\n" + "=" * 60)
    print("TEST 4: Sample sizes")
    print("=" * 60)
    print(f"  Negative cells: {len(X_neg)}")
    print(f"  Calcein cells:  {len(X_stain)}")

if __name__ == "__main__":
    main()
