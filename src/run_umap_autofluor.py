import os
import sys
import argparse
import glob

# Add project root to path BEFORE importing src modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import umap
import fcsparser
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from src.unmix_factory import get_unmixer

# GPUチェック
try:
    import cuml
    from cuml.manifold import UMAP as cuUMAP
    HAS_CUML = True
except ImportError:
    HAS_CUML = False

from config import COFACTOR
from src.convert import convert_sraw_to_csv


def load_spectral_data(sraw_path, fcs_path, cofactor=None):
    if cofactor is None:
        cofactor = COFACTOR

    temp_dir = os.path.join(PROJECT_ROOT, "analysis", "results", "_temp_autofluor")
    os.makedirs(temp_dir, exist_ok=True)
    _, csv_path_wl, _, df_sraw = convert_sraw_to_csv(sraw_path, output_dir=temp_dir)

    # FCSファイルからデータを取得
    _, df_fcs = fcsparser.parse(fcs_path, reformat_meta=True)
    # srawのイベント数とFCSのイベント数が一致するか確認
    assert len(df_sraw) == len(df_fcs), \
        f"Event counts do not match: sraw={len(df_sraw)}, fcs={len(df_fcs)}"

    # 波長ごとのフルオレッセンススペクトルを取得
    wl_features = [
        c for c in df_sraw.columns
        if c.startswith('Area_') and c.endswith('nm')
    ]
    X_spectral = df_sraw[wl_features].values

    # 一時ファイルの削除
    try:
        os.remove(csv_path_wl)
        csv_path_ch = csv_path_wl.replace('_wavelength.csv', '_channel.csv')
        if os.path.exists(csv_path_ch):
            os.remove(csv_path_ch)
    except OSError:
        pass

    return X_spectral, wl_features, df_fcs


# fcsファイルとsrawファイルをペアにし，パスをlistで返す
def find_sraw_fcs_pairs(directory):
    pairs = []
    sraw_files = sorted(glob.glob(os.path.join(directory, "*.sraw")))
    for sraw_path in sraw_files:
        base = os.path.splitext(sraw_path)[0]
        fcs_path = base + ".fcs"
        if os.path.isfile(fcs_path):
            pairs.append((sraw_path, fcs_path))
        else:
            print(f"  Warning: No matching .fcs for {os.path.basename(sraw_path)}, skipping.")
    return pairs


def run_umap_autofluor(neg_dir, stain_dir, output_path, stain_name="PI",
                       cofactor=None, seed=42, png_output_path=None, method='poisson'):
    """
    Args
    ----------
    neg_dir : str
        Neg. files directory.
    stain_dir : str
        Stained files directory.
    output_path : str
        Output CSV path.
    stain_name : str
        Stain name for legend.
    cofactor : float or None
        ArcSinh cofactor (default from config).
    seed : int
        Random seed.
    png_output_path : str or None
        If provided, generates 3-panel PNG visualization.
    method : str
        Unmixing method: 'poisson' | 'nnls' | 'linear' | 'poisson_glm' | 'None'.
        'None' performs no unmixing.
    """
    if cofactor is None:
        cofactor = COFACTOR

    # =========================================================================
    # 1. Loading Negative Samples (used for Reference Spectra extraction)
    # =========================================================================
    print("=" * 60)
    print("Step 1: Loading Negative (autofluorescence) samples...")
    print("=" * 60)

    # ネガティブサンプルを読み込む
    neg_pairs = find_sraw_fcs_pairs(neg_dir)
    if not neg_pairs:
        print(f"Error: No .sraw/.fcs pairs found in {neg_dir}")
        sys.exit(1)

    # ネガティブサンプルを1つだけ使用する
    neg_pairs = neg_pairs[:1]
    
    # ネガティブサンプルのスペクトルデータを読み込む際に必要な入れ物
    neg_spectral_list = []
    neg_fcs_list = []
    wl_features = None

    # ネガティブサンプルのスペクトルデータを読み込む
    for sraw_path, fcs_path in neg_pairs:
        print(f"  Loading {os.path.basename(sraw_path)}...")
        X_sp, wl_feat, df_fcs = load_spectral_data(sraw_path, fcs_path, cofactor)
        # さっき用意した入れ物に追加していく
        neg_spectral_list.append(X_sp)
        neg_fcs_list.append(df_fcs)
        if wl_features is None:
            wl_features = wl_feat

    # ネガティブサンプルのスペクトルデータを結合する
    X_neg = np.vstack(neg_spectral_list)
    df_neg_fcs = pd.concat(neg_fcs_list, ignore_index=True)
    from src.unmix_spectral import get_scatter_features
    scat_features = get_scatter_features(df_neg_fcs)
    scatter_neg = df_neg_fcs[scat_features].values if len(scat_features) > 0 else None
    print(f"  Total Negative events: {len(X_neg)} ({len(neg_pairs)} files)")

    # =========================================================================
    # 2. Loading Stained Samples
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"Step 2: Loading {stain_name}-stained samples...")
    print("=" * 60)

    # ここはさっきのunstainと同じ処理
    stain_pairs = find_sraw_fcs_pairs(stain_dir)
    if not stain_pairs:
        print(f"Error: No .sraw/.fcs pairs found in {stain_dir}")
        sys.exit(1)

    stain_spectral_list = []
    stain_fcs_list = []

    for sraw_path, fcs_path in stain_pairs:
        print(f"  Loading {os.path.basename(sraw_path)}...")
        X_sp, _, df_fcs = load_spectral_data(sraw_path, fcs_path, cofactor)
        stain_spectral_list.append(X_sp)
        stain_fcs_list.append(df_fcs)

    X_stain = np.vstack(stain_spectral_list)
    df_stain_fcs = pd.concat(stain_fcs_list, ignore_index=True)
    scatter_stain = df_stain_fcs[scat_features].values if len(scat_features) > 0 else None
    print(f"  Total {stain_name} events: {len(X_stain)} ({len(stain_pairs)} files)")

    # =========================================================================
    # 3. Extracting unmixed intensity and Reconstructing AF Spectra
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"Step 3: Extracting unmixed intensity & Reconstructing AF Spectra...")
    print("=" * 60)

    # 前処理済みCSVファイルを見つける
    def find_processed_csv(sraw_path, target_stain):
        parts = os.path.normpath(sraw_path).split(os.sep)
        try:
            data_idx = parts.index("data")
            experiment_folder = parts[data_idx + 1]
        except (ValueError, IndexError):
            experiment_folder = "Experiment 2026!05!21 15!59"
            
        from config import EXPERIMENTS, RESULTS_DIR
        date_str = EXPERIMENTS.get(experiment_folder, experiment_folder)
        
        filename = os.path.basename(sraw_path)
        base_name = os.path.splitext(filename)[0]
        well_id = base_name.split(' ')[0] if ' ' in base_name else base_name
        
        sample_label = f"{target_stain}_{well_id}"
        pattern = os.path.join(RESULTS_DIR, date_str, sample_label, "*_wavelength.csv")
        csv_files = glob.glob(pattern)
        csv_files = [p for p in csv_files if "scarf_embeddings" not in p]
        if csv_files:
            csv_files.sort(reverse=True)  # Get the most recent file by timestamp in filename
            return csv_files[0]
        return None

    stain_unmixed_af = []
    stain_unmixed_stain = []
    
    for sraw_path, _ in stain_pairs:
        csv_path = find_processed_csv(sraw_path, stain_name)
        if csv_path:
            df_csv = pd.read_csv(csv_path)
            if 'Unmixed_AF' in df_csv.columns and f'Unmixed_{stain_name}' in df_csv.columns:
                stain_unmixed_af.append(df_csv['Unmixed_AF'].values)
                stain_unmixed_stain.append(df_csv[f'Unmixed_{stain_name}'].values)
                
    has_unmixed = False
    if len(stain_unmixed_af) == len(stain_pairs):
        stain_unmixed_af = np.concatenate(stain_unmixed_af)
        stain_unmixed_stain = np.concatenate(stain_unmixed_stain)
        af_intensity = np.arcsinh(stain_unmixed_af / cofactor)
        stain_intensity = np.arcsinh(stain_unmixed_stain / cofactor)
        af_label = 'Unmixed AF (ArcSinh)'
        stain_label = f'Unmixed {stain_name} (ArcSinh)'
        has_unmixed = True
        print(f"  Successfully loaded unmixed intensities from processed CSVs.")
        
        # Calculate reference spectra directly to reconstruct AF
        print(f"  Reconstructing pure Autofluorescence spectra for Stained samples using {method}...")
        
        # date_str を CSV パスから直接抽出（run_unmixing_group と同じロジック）
        # results/{date_str}/{sample_label}/file.csv の構造を利用
        def _find_processed_csv_for_datestr(sraw_path, stain_name):
            """processed CSV のパスから date_str を抽出する"""
            from config import RESULTS_DIR
            base = os.path.splitext(os.path.basename(sraw_path))[0]
            pattern = os.path.join(RESULTS_DIR, "**", f"{stain_name}_*", f"*{base}*_wavelength.csv")
            csv_files = glob.glob(pattern, recursive=True)
            csv_files = [p for p in csv_files if "scarf_embeddings" not in p]
            if csv_files:
                parts = os.path.normpath(csv_files[0]).split(os.sep)
                return parts[-3]  # date_str
            return None
        
        date_str = _find_processed_csv_for_datestr(stain_pairs[0][0], stain_name)
        if date_str is None:
            # フォールバック: データパスから実験フォルダ名を抽出
            parts_neg = os.path.normpath(neg_pairs[0][0]).split(os.sep)
            try:
                data_idx = parts_neg.index("data")
                date_str = parts_neg[data_idx + 1]
            except (ValueError, IndexError):
                date_str = None
                print("  Warning: Could not determine date_str for model loading.")
        
        unmixer = get_unmixer(method, X_neg, X_stain, date_str=date_str, scatter_neg=scatter_neg)
        
        if method in ['poisson', 'poisson_glm']:
            # For dynamic AF, we must use remove_stain_component because S_AF isn't static
            X_to_umap = unmixer.remove_stain_component(X_stain, scatter_val=scatter_stain)
        else:
            X_to_umap = unmixer.remove_stain_component(X_stain, scatter_val=scatter_stain)
            
        X_to_umap = np.maximum(X_to_umap, 0)
    else:
        print("  Warning: Unmixed CSV data not found for all samples. Falling back to raw spectra.")
        X_to_umap = X_stain
        
        stain_col = None
        for col in df_stain_fcs.columns:
            if stain_name.lower() in col.lower() and 'area' in col.lower():
                stain_col = col
                break

        if stain_col is not None:
            stain_intensity = np.arcsinh(df_stain_fcs[stain_col].values / cofactor)
            stain_label = f'{stain_col} (ArcSinh)'
        else:
            stain_intensity = np.arcsinh(X_stain.mean(axis=1) / cofactor)
            stain_label = 'Mean Spectral Intensity (ArcSinh)'
            
    vmin, vmax = np.percentile(stain_intensity, [1, 99])

    # =========================================================================
    # 4. Preprocessing (ArcSinh + StandardScaler)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Step 4: Preprocessing (ArcSinh + StandardScaler)...")
    print("=" * 60)

    # ---------------------------------------------------------
    # Apply Savitzky-Golay filter to remove shot noise (scatter)
    # ---------------------------------------------------------
    window_length = 7  # Must be odd
    polyorder = 2
    
    X_neg_smooth = savgol_filter(X_neg, window_length, polyorder, axis=1)
    X_to_umap_smooth = savgol_filter(X_to_umap, window_length, polyorder, axis=1)
    
    # Prevent negative values caused by the polynomial fit
    X_neg_smooth = np.maximum(X_neg_smooth, 0)
    X_to_umap_smooth = np.maximum(X_to_umap_smooth, 0)

    X_neg_arcsinh = np.arcsinh(X_neg_smooth / cofactor)
    X_to_umap_arcsinh = np.arcsinh(X_to_umap_smooth / cofactor)
    
    # Negativeデータ（生の自家蛍光）を基準としてScalerをフィットする
    scaler = StandardScaler()
    # unstain(negative)で座標系を作成
    X_neg_scaled = scaler.fit_transform(X_neg_arcsinh)
    # 作成した座標系に対してstainを割り当てていく
    X_to_umap_scaled = scaler.transform(X_to_umap_arcsinh)
    
    X_combined_scaled = np.vstack([X_neg_scaled, X_to_umap_scaled])

    # =========================================================================
    # 5. 2D UMAP
    # =========================================================================
    print(f"\n{'=' * 60}")
    if HAS_CUML:
        print("Step 5: Running 2D UMAP on AF data (GPU cuML, fit on Negative only)...")
    else:
        print("Step 5: Running 2D UMAP on AF data (fit on Negative only)...")
    print("=" * 60)

    if HAS_CUML:
        reducer = cuUMAP(
            n_components=2,
            n_neighbors=10,
            min_dist=0.3,
            random_state=seed,
            metric='euclidean'
        )
    else:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=10,
            min_dist=0.3,
            metric='euclidean',
            n_jobs=-1,          # random_state を外して並列処理を有効化
            low_memory=False,
        )

    # UMAPもNegativeデータのみを基準の多様体（manifold）として学習する
    # これにより、AEでスムージングされたデータも元のNegativeの空間に投影される
    # お客様のご要望通り、Negativeの全データ（20000等）を用いて限界まで高精度にフィットさせます
    # （ただし超巨大データでのメモリ枯渇を防ぐため、安全装置として上限50000としています）
    n_fit = min(50000, len(X_neg_scaled))
    rng = np.random.default_rng(seed)
    fit_idx = rng.choice(len(X_neg_scaled), n_fit, replace=False)
    X_fit = X_neg_scaled[fit_idx]

    print(f"  Fitting UMAP on {n_fit} Negative points...")
    reducer.fit(X_fit)
    print(f"  Transforming all {len(X_combined_scaled)} points...")
    umap_coords_combined = reducer.transform(X_combined_scaled)
    umap_neg = umap_coords_combined[:len(X_neg)]
    umap_stain = umap_coords_combined[len(X_neg):]

    # =========================================================================
    # 5.5. Validating Unmixing Quality via 2D UMAP MMD
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 5.5: Validating Unmixing Quality via 2D UMAP MMD...")
    print("=" * 60)
    from src.evaluate_mmd import evaluate_unmixing_mmd, plot_mmd_results
    
    print(f"  Evaluating MMD between Unstain AF and Predicted {stain_name} AF (in High-Dimensional Spectral Space)...")
    try:
        mmd_results = evaluate_unmixing_mmd(
            X_unstain=X_neg_scaled, 
            X_af=X_to_umap_scaled, 
            n_samples=2000, 
            n_permutations=100,
            random_state=seed
        )
        print(f"  [HD MMD Result] Target MMD: {mmd_results['target_mmd']:.5f}")
        print(f"  [HD MMD Result] p-value:    {mmd_results['p_value']:.4f}")
        if "silhouette_score" in mmd_results:
            print(f"  [Silhouette Score]:         {mmd_results['silhouette_score']:.5f} (Closer to 0 is better, perfect mix)")
        
        # Save the plot
        out_dir = os.path.dirname(output_path)
        mmd_png_path = os.path.join(out_dir, f"mmd_hd_evaluation_{stain_name}.png")
        plot_mmd_results(mmd_results, output_path=mmd_png_path)
        print(f"  High-Dim MMD evaluation plot saved to: {mmd_png_path}")
    except Exception as e:
        print(f"  [UMAP MMD Evaluation Failed]: {e}")
    print()

    # =========================================================================
    # 6. Plotly Visualization
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Step 6: Generating interactive 2D plot...")
    print("=" * 60)

    fig = make_subplots(
        rows=1, cols=3,
        shared_xaxes="all", shared_yaxes="all",
        subplot_titles=(
            'Negative Control AF',
            f'Stained ({stain_name}) - Unmixed AF',
            f'Stained ({stain_name}) - Colored by {stain_label}'
        )
    )
    
    # Left panel: Negative Control
    fig.add_trace(
        go.Scatter(
            x=umap_neg[:, 0], y=umap_neg[:, 1],
            mode='markers', name='Negative AF',
            marker=dict(
                size=3, color='#808080',
                opacity=0.6
            ),
            showlegend=False
        ), row=1, col=1
    )
    
    # Middle panel: Uncolored AF UMAP of the stained sample
    fig.add_trace(
        go.Scatter(
            x=umap_stain[:, 0], y=umap_stain[:, 1],
            mode='markers', name='Unmixed AF',
            marker=dict(
                size=3, color='#4682b4',
                opacity=0.6
            ),
            showlegend=False
        ), row=1, col=2
    )
    
    # Right panel: Same UMAP colored by stain
    fig.add_trace(
        go.Scatter(
            x=umap_stain[:, 0], y=umap_stain[:, 1],
            mode='markers', name=f'{stain_name} Intensity',
            marker=dict(
                size=3, color=stain_intensity,
                cmin=vmin, cmax=vmax,
                colorscale='bluered', opacity=0.7,
                colorbar=dict(title=stain_label, x=1.0)
            ),
            showlegend=False
        ), row=1, col=3
    )

    fig.update_layout(
        title=f'Autofluorescence UMAP Projection',
        width=1800, height=600,
        margin=dict(l=40, r=40, b=40, t=60)
    )
    for i in range(1, 4):
        fig.update_xaxes(title_text='UMAP 1', row=1, col=i)
        fig.update_yaxes(title_text='UMAP 2', row=1, col=i)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    # fig.write_html(output_path)
    # print(f"\n  Interactive 2D plot saved to: {output_path}")

    # Generate Matplotlib PNG if png_output_path is specified
    if png_output_path:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig_mpl, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=200, sharex=True, sharey=True, layout="constrained")
        
        # Left: Negative Control
        axes[0].scatter(
            umap_neg[:, 0], umap_neg[:, 1], 
            c='#808080', s=2, alpha=0.5
        )
        axes[0].set_title('Negative Control AF', fontsize=12, fontweight='bold', pad=10)
        axes[0].set_xlabel('UMAP 1', fontsize=10)
        axes[0].set_ylabel('UMAP 2', fontsize=10)
        
        # Middle: Unmixed AF
        axes[1].scatter(
            umap_stain[:, 0], umap_stain[:, 1], 
            c='#4682b4', s=2, alpha=0.5
        )
        axes[1].set_title(f'Stained ({stain_name}) - Unmixed AF', fontsize=12, fontweight='bold', pad=10)
        axes[1].set_xlabel('UMAP 1', fontsize=10)
        axes[1].set_ylabel('UMAP 2', fontsize=10)
        
        # Right: Colored by Stain
        sc2 = axes[2].scatter(
            umap_stain[:, 0], umap_stain[:, 1], 
            c=stain_intensity, vmin=vmin, vmax=vmax, cmap='coolwarm', s=2, alpha=0.5
        )
        axes[2].set_title(f'Stained ({stain_name}) - Colored by {stain_name}', fontsize=12, fontweight='bold', pad=10)
        axes[2].set_xlabel('UMAP 1', fontsize=10)
        axes[2].set_ylabel('UMAP 2', fontsize=10)
        plt.colorbar(sc2, ax=axes.ravel().tolist(), label=stain_label)
        
        fig_mpl.savefig(png_output_path, bbox_inches='tight')
        plt.close(fig_mpl)
        print(f"  Static 2D plot saved to: {png_output_path}")

        # --- Generate additional plot: Raw stained data projected onto Negative UMAP manifold ---
        print("\n  Projecting RAW stained data onto Negative UMAP manifold...")
        # Preprocess raw X_stain with the SAME scaler fitted on Negative data
        X_stain_smooth = savgol_filter(X_stain, window_length, polyorder, axis=1)
        X_stain_smooth = np.maximum(X_stain_smooth, 0)
        X_stain_arcsinh = np.arcsinh(X_stain_smooth / cofactor)
        # Use the SAME scaler from Step 4 (fitted on Negative) to transform raw stain
        X_stain_scaled_raw = scaler.transform(X_stain_arcsinh)
        
        # Project onto the existing Negative UMAP manifold (no new fit!)
        print(f"  Transforming {len(X_stain_scaled_raw)} raw stain points onto Negative manifold...")
        umap_stain_raw = reducer.transform(X_stain_scaled_raw)
        
        # 3-panel figure: Negative | Raw Stain projected | colored by stain intensity
        fig_raw, axes_raw = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150, sharex=True, sharey=True, layout="constrained")
        
        axes_raw[0].scatter(umap_neg[:, 0], umap_neg[:, 1], c='#808080', s=2, alpha=0.3)
        axes_raw[0].set_title('Negative Control AF', fontsize=12, fontweight='bold', pad=10)
        axes_raw[0].set_xlabel('UMAP 1', fontsize=10)
        axes_raw[0].set_ylabel('UMAP 2', fontsize=10)
        
        axes_raw[1].scatter(umap_stain_raw[:, 0], umap_stain_raw[:, 1], c='#c45a5a', s=2, alpha=0.5)
        axes_raw[1].set_title(f'Raw Stained ({stain_name}) on Negative Manifold', fontsize=12, fontweight='bold', pad=10)
        axes_raw[1].set_xlabel('UMAP 1', fontsize=10)
        axes_raw[1].set_ylabel('UMAP 2', fontsize=10)
        
        sc_raw3 = axes_raw[2].scatter(
            umap_stain_raw[:, 0], umap_stain_raw[:, 1],
            c=stain_intensity, vmin=vmin, vmax=vmax, cmap='coolwarm', s=2, alpha=0.5
        )
        axes_raw[2].set_title(f'Raw Stained ({stain_name}) - Colored by {stain_name}', fontsize=12, fontweight='bold', pad=10)
        axes_raw[2].set_xlabel('UMAP 1', fontsize=10)
        axes_raw[2].set_ylabel('UMAP 2', fontsize=10)
        plt.colorbar(sc_raw3, ax=axes_raw.ravel().tolist(), label=stain_label)
        
        raw_png_path = png_output_path.replace("autofluor_umap_", "raw_on_neg_umap_")
        fig_raw.savefig(raw_png_path, bbox_inches='tight')
        plt.close(fig_raw)
        print(f"  Static RAW-on-Negative UMAP plot saved to: {raw_png_path}")

    temp_dir = os.path.join(PROJECT_ROOT, "analysis", "results", "_temp_autofluor")
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description='自家蛍光 UMAP + マーカー投影 (2D Plotly)'
    )
    parser.add_argument('--neg-dir', type=str, required=True,
                        help='ネガティブコントロールのデータディレクトリ')
    parser.add_argument('--stain-dir', type=str, required=True,
                        help='マーカー染色サンプルのデータディレクトリ')
    parser.add_argument('--stain', type=str, default='PI',
                        help='染色マーカー名 (デフォルト: PI)')
    parser.add_argument('--output', type=str, default=None,
                        help='出力 HTML ファイルのパス')
    parser.add_argument('--png-output', type=str, default=None,
                        help='出力 PNG ファイルのパス')
    parser.add_argument('--cofactor', type=float, default=None,
                        help='ArcSinh cofactor')
    parser.add_argument('--seed', type=int, default=42,
                        help='UMAP の乱数シード')
    parser.add_argument('--method', type=str, default='poisson_glm',
                        help='アンミキシング手法 (デフォルト: poisson_glm)')

    args = parser.parse_args()

    if args.output is None:
        results_dir = os.path.join(PROJECT_ROOT, "analysis", "results")
        os.makedirs(results_dir, exist_ok=True)
        args.output = os.path.join(results_dir, f"autofluor_umap_{args.stain}.html")

    run_umap_autofluor(
        neg_dir=args.neg_dir,
        stain_dir=args.stain_dir,
        output_path=args.output,
        stain_name=args.stain,
        cofactor=args.cofactor,
        seed=args.seed,
        png_output_path=args.png_output,
        method=args.method
    )


if __name__ == '__main__':
    main()
