import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_spectral_features(df):
    """Get spectral channels"""
    return [c for c in df.columns if c.startswith('Area_') and c.endswith('nm')]

def save_unmixing_plot(raw_af_vals, raw_stain_vals, af_vals, stain_vals, stain_name, plot_path, title):
    """Save 2D scatter plot comparing raw vs unmixed results"""
    raw_af_plot = np.arcsinh(raw_af_vals / 150.0)
    raw_stain_plot = np.arcsinh(raw_stain_vals / 150.0)
    af_plot = np.arcsinh(af_vals / 150.0)
    stain_plot = np.arcsinh(stain_vals / 150.0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=150)
    
    # Left: Raw data
    sc0 = axes[0].scatter(raw_af_plot, raw_stain_plot, s=2, alpha=0.3, c=raw_stain_plot, cmap='coolwarm', edgecolors='none')
    axes[0].set_xlabel("Raw Peak Autofluorescence Channel (ArcSinh)")
    axes[0].set_ylabel(f"Raw Peak {stain_name} Channel (ArcSinh)")
    axes[0].set_title("Before Unmixing (Raw Data)")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    fig.colorbar(sc0, ax=axes[0], label=f'Raw {stain_name} (ArcSinh)')

    # Right: Unmixed data
    sc1 = axes[1].scatter(af_plot, stain_plot, s=2, alpha=0.3, c=stain_plot, cmap='coolwarm', edgecolors='none')
    axes[1].set_xlabel("Unmixed Autofluorescence (ArcSinh)")
    axes[1].set_ylabel(f"Unmixed {stain_name} (ArcSinh)")
    axes[1].set_title("After Spectral Unmixing")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    fig.colorbar(sc1, ax=axes[1], label=f'Unmixed {stain_name} (ArcSinh)')
    
    fig.suptitle(f"Spectral Unmixing Result\n{title}", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()

class PoissonUnmixer:
    """
    スペクトルアンミキシング: 自家蛍光(AF)と蛍光色素を分離する.

    Parameters
    ----------
    max_iter : int
        参照スペクトル精製およびIRLS内部の最大反復回数.
    tol : float
        参照スペクトル精製の収束判定閾値 (S_Stain の最大変化量).
    method : str
        アンミキシング手法. 'poisson' (IRLS, デフォルト) or 'ols' (旧方式).
    irls_iter : int
        ポアソンIRLS内部の反復回数 (method='poisson' のみ).
    """
    def __init__(self, max_iter=5, tol=1e-4, method='poisson', irls_iter=3):
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.irls_iter = irls_iter
        
        # 学習される参照スペクトル (テンプレート)
        self.S_AF = None      # 自家蛍光(Autofluorescence)の基準スペクトル (合計1に正規化された波形)
        self.S_Stain = None   # 色素(Stain)の基準スペクトル (合計1に正規化された波形)
        self.S = None         # S_AFとS_Stainを結合した参照行列 (現在未使用だが可視化等で有用)
        
        # 漏れ込み補正用の係数
        self.slope = 0.0      # ネガティブ細胞における、自家蛍光強度に対する色素強度の漏れ込みの傾き
        self.bg = 0.0         # 漏れ込み補正後の色素チャンネルのバックグラウンド(オフセット)値
        
        # 動的に決定されるチャンネルマスク(True/Falseの配列)
        self._tail_mask = None  # 自家蛍光成分(c_af)を推定するための、色素の発光がゼロとみなせる波長領域
        self._peak_mask = None  # 色素成分(c_stain)を推定するための、色素の発光が強い波長領域

    def _compute_channel_masks(self):
        """S_Stain の値に基づいてテール領域とピーク領域のマスクを動的に決定する.

        テール領域: S_Stain が最大値の 5% 未満のチャンネル (色素の発光が無視できる領域)
        ピーク領域: S_Stain が最大値の 30% 以上のチャンネル (色素の発光が強い領域)

        テール領域が 4 チャンネル未満の場合は末尾 8 チャンネルにフォールバックする.
        """
        S_Stain = self.S_Stain
        S_AF = self.S_AF

        # ピーク領域: S_Stain が最大値の 30% 以上
        stain_threshold = 0.3 * np.max(S_Stain)
        self._peak_mask = S_Stain >= stain_threshold

        # テール領域: S_Stain が最大値の 5% 未満 (色素の寄与が無視できる)
        tail_candidates = S_Stain < 0.05 * np.max(S_Stain)
        if np.sum(tail_candidates) >= 4:
            self._tail_mask = tail_candidates
        else:
            # フォールバック: 末尾 8 チャンネル
            self._tail_mask = np.zeros(len(S_AF), dtype=bool)
            self._tail_mask[-8:] = True

    def fit(self, X_neg, X_stain):
        """ネガティブコントロールと染色サンプルから参照スペクトルを学習する.

        Parameters
        ----------
        X_neg : ndarray, shape (n_neg, n_channels)
            ネガティブコントロール(未染色)のスペクトルデータ.
        X_stain : ndarray, shape (n_stain, n_channels)
            染色サンプルのスペクトルデータ.
        """
        # 1. Autofluorescence (AF) Reference
        self.S_AF = np.median(X_neg, axis=0)
        self.S_AF = self.S_AF / (np.sum(self.S_AF) + 1e-9)

        # 2. Stain Reference — 初期推定
        total_intensity = np.sum(X_stain, axis=1)
        bright_cells_total = X_stain[total_intensity >= np.percentile(total_intensity, 99)]
        if len(bright_cells_total) == 0:
            bright_cells_total = X_stain

        S_bright_total = np.median(bright_cells_total, axis=0)
        ratios = S_bright_total / (self.S_AF + 1e-9)
        peak_idx = np.argmax(ratios)

        # ピーク波長の蛍光強度分布から、細胞の密集度が最も高い「最頻値（Mode）」を探す
        peak_values = X_stain[:, peak_idx]
        
        # ネガティブ細胞群のピークを誤って拾わないよう、上位50%の明るさを持つ細胞群に絞る
        bright_half = peak_values[peak_values > np.median(peak_values)]
        if len(bright_half) == 0:
            bright_half = peak_values
            
        # ヒストグラムを作成して最も密度が高い（細胞数が多い）強度帯を特定
        hist, bin_edges = np.histogram(bright_half, bins=50)
        max_bin_idx = np.argmax(hist)
        mode_val = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2.0
        
        # 最頻値周辺（±5%の幅）に密集している典型的な細胞群のみを抽出
        margin = (np.max(bright_half) - np.min(bright_half)) * 0.05
        stained_cells = X_stain[(peak_values >= mode_val - margin) & (peak_values <= mode_val + margin)]
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
        self._compute_channel_masks()

        # Step B & C: Iterative refinement of S_Stain using max_iter/tol
        for iteration in range(self.max_iter):
            # Unmix all stained cells with current S_Stain
            C = self._unmix(X_stain)
            c_af_all = C[:, 0]
            c_stain_all = C[:, 1]

            # Select cells in the densest typical region (40th to 60th percentile) to avoid saturated outliers
            p40 = np.percentile(c_stain_all, 40)
            p60 = np.percentile(c_stain_all, 60)
            valid = (c_stain_all >= p40) & (c_stain_all <= p60)
            X_valid = X_stain[valid]
            c_af_valid = c_af_all[valid]
            c_stain_valid = c_stain_all[valid]

            # Compute per-cell stain spectrum estimate:
            #   S_Stain_i = (X_i - c_af_i * S_AF) / c_stain_i
            residuals = X_valid - c_af_valid[:, None] * self.S_AF[None, :]
            per_cell_stain = residuals / (c_stain_valid[:, None] + 1e-9)

            S_Stain_new = np.median(per_cell_stain, axis=0)
            S_Stain_new = np.maximum(S_Stain_new, 0)
            S_Stain_new = S_Stain_new / (np.sum(S_Stain_new) + 1e-9)

            # Convergence check
            delta = np.max(np.abs(S_Stain_new - self.S_Stain))
            self.S_Stain = S_Stain_new
            self.S = np.column_stack((self.S_AF, self.S_Stain))
            self._compute_channel_masks()

            if delta < self.tol:
                break

        # 3. Calculate leakage slope and background using negative control
        C_neg = self._unmix(X_neg)

        # slopeがマイナスになるのを防ぐセーフティ
        raw_slope = C_neg[:, 1].mean() / (C_neg[:, 0].mean() + 1e-9)
        self.slope = max(raw_slope, 0.0)

        C_no_slope = C_neg[:, 1] - self.slope * C_neg[:, 0]
        self.bg = np.median(C_no_slope)

        return self

    def _unmix(self, X):
        """method パラメータに応じてアンミキシング手法を切り替える."""
        if self.method == 'poisson':
            return self._unmix_poisson_irls(X)
        else:
            return self._unmix_ols_sequential(X)

    def _unmix_ols_sequential(self, X):
        """旧方式: OLS による逐次推定 (tail→c_af, peak→c_stain).

        Joint fitting (OLS/IRLS) over all channels is biased because
        S_AF and S_Stain overlap at 500-540nm. The Calcein peak dominates,
        causing c_af to be severely underestimated and c_stain inflated.

        Instead:
          1. Estimate c_af from tail channels where S_Stain ≈ 0
          2. Subtract c_af * S_AF from the full spectrum
          3. Estimate c_stain from the residual at peak channels
        """
        S_AF = self.S_AF
        S_Stain = self.S_Stain
        tail_mask = self._tail_mask
        peak_mask = self._peak_mask

        # Step 1: c_af from tail channels only (OLS)
        S_AF_tail = S_AF[tail_mask]
        X_tail = X[:, tail_mask]
        denom_af = np.dot(S_AF_tail, S_AF_tail) + 1e-9
        c_af = X_tail @ S_AF_tail / denom_af  # (N,)

        # Step 2: subtract AF, then estimate c_stain from peak channels (OLS)
        residual = X - c_af[:, None] * S_AF[None, :]  # (N, M)
        S_Stain_peak = S_Stain[peak_mask]
        R_peak = residual[:, peak_mask]
        denom_stain = np.dot(S_Stain_peak, S_Stain_peak) + 1e-9
        c_stain = R_peak @ S_Stain_peak / denom_stain  # (N,)

        C = np.column_stack((c_af, c_stain))
        return C

    def _unmix_poisson_irls(self, X):
        """ポアソン IRLS による逐次推定.

        逐次推定の構造 (tail→c_af, peak→c_stain) を維持しつつ、
        各ステップでポアソンノイズモデルに基づく重み w_i = 1/(predicted_i + eps)
        を用いた反復再重み付き最小二乗法 (IRLS) を適用する.

        ポアソン分布では分散 = 期待値なので、分散の逆数で重み付けすることで
        高信号チャンネルの過大な影響を抑え、暗い細胞のノイズ耐性を向上させる.
        """
        S_AF = self.S_AF
        S_Stain = self.S_Stain
        tail_mask = self._tail_mask
        peak_mask = self._peak_mask
        eps = 1.0  # ゼロ割り防止 (ポアソンの最小分散)

        # ---- Step 1: c_af from tail channels (Poisson IRLS) ----
        S_AF_tail = S_AF[tail_mask]
        X_tail = X[:, tail_mask]  # (N, T)

        # 初期推定 (OLS)
        denom_af_init = np.dot(S_AF_tail, S_AF_tail) + 1e-9
        c_af = X_tail @ S_AF_tail / denom_af_init  # (N,)

        for _ in range(self.irls_iter):
            # 予測値: predicted_tail = c_af * S_AF_tail
            predicted_tail = np.maximum(c_af[:, None] * S_AF_tail[None, :], eps)  # (N, T)
            # ポアソン重み: w = 1 / predicted (分散 = 期待値)
            W = 1.0 / predicted_tail  # (N, T)

            # 重み付き最小二乗: c_af = sum(w * X * S) / sum(w * S^2)
            numerator = np.sum(W * X_tail * S_AF_tail[None, :], axis=1)    # (N,)
            denominator = np.sum(W * S_AF_tail[None, :] ** 2, axis=1) + 1e-9  # (N,)
            c_af = numerator / denominator

        # ---- Step 2: c_stain from peak channels (Poisson IRLS) ----
        # まず残差を計算
        residual = X - c_af[:, None] * S_AF[None, :]  # (N, M)
        S_Stain_peak = S_Stain[peak_mask]
        R_peak = residual[:, peak_mask]  # (N, P)

        # 初期推定 (OLS)
        denom_stain_init = np.dot(S_Stain_peak, S_Stain_peak) + 1e-9
        c_stain = R_peak @ S_Stain_peak / denom_stain_init  # (N,)

        for _ in range(self.irls_iter):
            # 予測値: predicted_peak = c_stain * S_Stain_peak
            # (残差に対するフィットなので、predicted はステイン成分のみ)
            predicted_peak = np.maximum(c_stain[:, None] * S_Stain_peak[None, :], eps)  # (N, P)
            W = 1.0 / predicted_peak  # (N, P)

            numerator = np.sum(W * R_peak * S_Stain_peak[None, :], axis=1)    # (N,)
            denominator = np.sum(W * S_Stain_peak[None, :] ** 2, axis=1) + 1e-9  # (N,)
            c_stain = numerator / denominator

        C = np.column_stack((c_af, c_stain))
        return C

    def transform(self, X):
        """アンミキシングを実行し、漏れ込み補正済みの係数を返す.

        Returns
        -------
        C_af : ndarray, shape (n,)
            自家蛍光の強度係数.
        C_stain_corrected : ndarray, shape (n,)
            漏れ込み補正済みの色素強度係数.
        """
        C = self._unmix(X)
        C_af = C[:, 0]
        C_stain = C[:, 1]
        C_stain_corrected = C_stain - self.slope * C_af - self.bg
        return C_af, C_stain_corrected

    def get_raw_coefficients(self, X):
        """補正前の生の係数 [c_af, c_stain] を返す."""
        return self._unmix(X)

    def remove_stain_component(self, X):
        """スペクトルから色素成分を除去し、純粋な自家蛍光スペクトルを返す."""
        C = self._unmix(X)
        return X - C[:, 1][:, None] * self.S_Stain[None, :]

def run_unmixing_group(results_base_dir, stain_name="PI", method='poisson'):
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
    
    if method == 'autoencoder':
        print("    [AutoEncoder] Initializing AutoEncoderUnmixer...")
        from src.unmix_autoencoder import AutoEncoderUnmixer
        model_path = os.path.join(results_base_dir, "ae_model.pth")
        unmixer = AutoEncoderUnmixer(model_save_path=model_path)
        unmixer.fit(X_neg_all, X_stain_all)
    elif method == 'scarf':
        print("    [SCARF] Initializing ScarfKnnUnmixer...")
        from src.unmix_scarf import ScarfKnnUnmixer
        unmixer = ScarfKnnUnmixer(k_neighbors=10)
        unmixer.fit(X_neg_all, X_stain_all)
        
        neg_emb_list = []
        for path in neg_csv_paths:
            parts = path.split(os.sep)
            date_str = parts[-3]
            sample_label = parts[-2]
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            emb_path = os.path.join(project_root, "learning", "results", date_str, sample_label, f"{sample_label}_scarf_embeddings.csv")
            if os.path.exists(emb_path):
                neg_emb_list.append(pd.read_csv(emb_path).values)
            else:
                print(f"    [SCARF Error] Missing embedding at {emb_path}")
                return
        emb_neg_all = np.vstack(neg_emb_list)
        unmixer.fit_knn(emb_neg_all, X_neg_all)
    else:
        unmixer = PoissonUnmixer()
        unmixer.fit(X_neg_all, X_stain_all)
        
    print(f"    Calculated autoflour leakage slope: {unmixer.slope:.6f}")
    
    peak_af_idx = np.argmax(unmixer.S_AF)
    peak_stain_idx = np.argmax(unmixer.S_Stain)
    
    for path in neg_csv_paths + stain_csv_paths:
        df = pd.read_csv(path)
        wl_features = get_spectral_features(df)
        X_val = df[wl_features].values
        
        raw_af_vals = X_val[:, peak_af_idx]
        raw_stain_vals = X_val[:, peak_stain_idx]
        
        if method == 'scarf':
            parts = path.split(os.sep)
            date_str = parts[-3]
            sample_label = parts[-2]
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            emb_path = os.path.join(project_root, "learning", "results", date_str, sample_label, f"{sample_label}_scarf_embeddings.csv")
            if os.path.exists(emb_path):
                emb_val = pd.read_csv(emb_path).values
                af, stain_corr = unmixer.transform_with_scarf(X_val, emb_val)
            else:
                print(f"    [SCARF Warning] Missing embedding for {sample_label}. Falling back to global S_AF.")
                af, stain_corr = unmixer.transform(X_val)
        else:
            af, stain_corr = unmixer.transform(X_val)

        
        df['Unmixed_AF'] = np.maximum(af, 0)
        df[f'Unmixed_{stain_name}'] = np.maximum(stain_corr, 0)
        
        # De-fragmentation warning workaround
        df = df.copy()
        df.to_csv(path, index=False)
        
        plot_path = os.path.join(os.path.dirname(path), "unmixing_scatter.png")
        save_unmixing_plot(raw_af_vals, raw_stain_vals, df['Unmixed_AF'].values, df[f'Unmixed_{stain_name}'].values, stain_name, plot_path, os.path.basename(path))
        print(f"    Unmixing complete: {os.path.basename(path)}")
