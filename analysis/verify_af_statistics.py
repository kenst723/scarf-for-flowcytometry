import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wasserstein_distance
import argparse

def main(experiment_dir):
    print(f"Analyzing experiment: {experiment_dir}")
    
    neg_csvs = glob.glob(os.path.join(experiment_dir, "Negative_*", "*.csv"))
    calcein_csvs = glob.glob(os.path.join(experiment_dir, "Calcein_*", "*.csv"))
    
    if not neg_csvs or not calcein_csvs:
        print("Error: Could not find Negative or Calcein CSV files in the specified directory.")
        return

    # Load Negative AF
    neg_af_list = []
    for f in neg_csvs:
        if "scarf_embeddings" in f: continue
        df = pd.read_csv(f)
        if 'Unmixed_AF' in df.columns:
            neg_af_list.append(df['Unmixed_AF'].values)
    
    # Load Calcein AF
    calcein_af_list = []
    for f in calcein_csvs:
        if "scarf_embeddings" in f: continue
        df = pd.read_csv(f)
        if 'Unmixed_AF' in df.columns:
            calcein_af_list.append(df['Unmixed_AF'].values)
            
    if not neg_af_list or not calcein_af_list:
        print("Error: Could not extract 'Unmixed_AF' column from the CSVs.")
        return

    neg_af = np.concatenate(neg_af_list)
    calcein_af = np.concatenate(calcein_af_list)
    
    print(f"Loaded {len(neg_af)} Negative cells and {len(calcein_af)} Calcein stained cells.")
    
    # 1. Descriptive Statistics
    neg_mean, neg_std = np.mean(neg_af), np.std(neg_af)
    cal_mean, cal_std = np.mean(calcein_af), np.std(calcein_af)
    neg_median = np.median(neg_af)
    cal_median = np.median(calcein_af)
    
    print("\n--- Descriptive Statistics ---")
    print(f"Negative AF : Mean = {neg_mean:.1f}, Median = {neg_median:.1f}, Std = {neg_std:.1f}")
    print(f"Stained AF  : Mean = {cal_mean:.1f}, Median = {cal_median:.1f}, Std = {cal_std:.1f}")
    
    # 2. Statistical Tests
    print("\n--- Statistical Tests ---")
    
    # Mann-Whitney U test (non-parametric comparison of medians/distributions)
    u_stat, p_val = stats.mannwhitneyu(neg_af, calcein_af, alternative='two-sided')
    print(f"Mann-Whitney U Test: statistic = {u_stat:.2f}, p-value = {p_val:.2e}")
    if p_val < 0.05:
        print("  -> Result: The two distributions are statistically significantly different (p < 0.05).")
    else:
        print("  -> Result: No significant difference found between the two distributions (p >= 0.05).")
    
    # Cohen's d (Effect Size)
    # d = (mean1 - mean2) / pooled_std
    pooled_std = np.sqrt((neg_std**2 + cal_std**2) / 2)
    cohens_d = (cal_mean - neg_mean) / pooled_std
    print(f"Cohen's d (Effect Size): {cohens_d:.3f}")
    if abs(cohens_d) < 0.2:
        print("  -> Interpretation: Small or negligible effect size (The distributions are highly similar).")
    elif abs(cohens_d) < 0.5:
        print("  -> Interpretation: Medium effect size.")
    else:
        print("  -> Interpretation: Large effect size (The distributions are substantially different).")
        
    # Wasserstein Distance (Earth Mover's Distance)
    wd = wasserstein_distance(neg_af, calcein_af)
    # Normalize Wasserstein by pooled std for scale-independent interpretation
    norm_wd = wd / pooled_std
    print(f"Wasserstein Distance (EMD): {wd:.1f} (Normalized by Std: {norm_wd:.3f})")

    # 3. Visualization
    print("\nGenerating distribution plots...")
    
    # Use ArcSinh scale for better visualization of flow cytometry data
    neg_af_arcsinh = np.arcsinh(neg_af / 150.0)
    cal_af_arcsinh = np.arcsinh(calcein_af / 150.0)
    
    plt.figure(figsize=(10, 6), dpi=150)
    
    sns.kdeplot(neg_af_arcsinh, fill=True, label='Negative (Unstained) AF', color='blue', alpha=0.4)
    sns.kdeplot(cal_af_arcsinh, fill=True, label='Stained (Unmixed) AF', color='red', alpha=0.4)
    
    plt.title('Comparison of Autofluorescence Distributions\n(Unstained vs Unmixed Stained)', fontsize=14, fontweight='bold')
    plt.xlabel('Autofluorescence Intensity (ArcSinh)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Add text box with statistics
    stat_text = (
        f"N (Cells) = {len(neg_af)} each\n"
        f"Cohen's d = {cohens_d:.3f}\n"
        f"Mann-Whitney p-val < 0.001" if p_val < 0.001 else f"Mann-Whitney p-val = {p_val:.3f}"
    )
    plt.text(0.95, 0.5, stat_text, 
             transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='center', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    out_png = os.path.join(experiment_dir, "af_distribution_comparison.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    
    print(f"Plot saved to: {out_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify AF distribution statistics")
    parser.add_argument("--dir", type=str, required=True, help="Path to experiment results dir (e.g. results/Experiment ...)")
    args = parser.parse_args()
    
    main(args.dir)
