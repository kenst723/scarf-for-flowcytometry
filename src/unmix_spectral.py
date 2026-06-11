import os
import glob
import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from joblib import Parallel, delayed
from src.unmix_factory import get_unmixer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False

def get_spectral_features(df):
    """Get spectral channels"""
    return [c for c in df.columns if c.startswith('Area_') and c.endswith('nm')]

def get_scatter_features(df):
    """Get scatter (FSC/SSC) channels"""
    return [c for c in df.columns if ('FSC' in c or 'SSC' in c) and ('Area' in c or 'Height' in c)]

def save_unmixing_plot(raw_af_vals, raw_stain_vals, af_vals, stain_vals, stain_name, plot_path, title):
    """Save 2D scatter plot comparing raw vs unmixed results"""
    raw_af_plot = np.arcsinh(raw_af_vals / 150.0)
    raw_stain_plot = np.arcsinh(raw_stain_vals / 150.0)
    af_plot = np.arcsinh(af_vals / 150.0)
    stain_plot = np.arcsinh(stain_vals / 150.0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=150, sharex=True, sharey=True)
    
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

    def fit(self, X_neg, X_stain, scatter_neg=None):
        """ネガティブコントロールと染色サンプルから参照スペクトルを学習する.

        Parameters
        ----------
        X_neg : ndarray, shape (n_neg, n_channels)
            ネガティブコントロール(未染色)のスペクトルデータ.
        X_stain : ndarray, shape (n_stain, n_channels)
            染色サンプルのスペクトルデータ.
        scatter_neg : ndarray, optional, shape (n_neg, n_scatter_features)
            AF予測用の散乱光(FSC/SSC)データ.
        """
        # 1. Autofluorescence (AF) Reference (Global median)
        self.S_AF = np.median(X_neg, axis=0)
        self.S_AF = self.S_AF / (np.sum(self.S_AF) + 1e-9)

        # 1.5. Dynamic AF Predictor (Random Forest)
        self.af_predictor = None
        self.scatter_scaler = None
        if False: # DISABLED MLP PREDICTOR FOR NOW
            from sklearn.preprocessing import StandardScaler
            self.scatter_scaler = StandardScaler()
            scatter_neg_scaled = self.scatter_scaler.fit_transform(scatter_neg)
            
            # Predict raw X_neg spectra (both shape and intensity)
            # We will use this prediction as a FIXED background during unmixing
            if has_torch:
                class AFPredictorMLP(torch.nn.Module):
                    def __init__(self, input_dim, output_dim, hidden_layers=[256, 256], dropout_rate=0.2):
                        super().__init__()
                        layers = []
                        prev_dim = input_dim
                        for h_dim in hidden_layers:
                            layers.append(torch.nn.Linear(prev_dim, h_dim))
                            layers.append(torch.nn.BatchNorm1d(h_dim))
                            layers.append(torch.nn.ReLU())
                            if dropout_rate > 0:
                                layers.append(torch.nn.Dropout(dropout_rate))
                            prev_dim = h_dim
                        layers.append(torch.nn.Linear(prev_dim, output_dim))
                        self.net = torch.nn.Sequential(*layers)
                        
                    def forward(self, x):
                        return self.net(x)

                class TorchAFPredictor:
                    def __init__(self, n_trials=20, epochs=800):
                        self.n_trials = n_trials
                        self.epochs = epochs
                        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        self.best_model = None
                        self.y_scaler = StandardScaler()
                        
                    def fit(self, X, Y):
                        import copy
                        import random
                        Y_scaled = self.y_scaler.fit_transform(Y)
                        
                        # Train/Validation Split (90% Train, 10% Val)
                        indices = np.arange(len(X))
                        np.random.seed(42)
                        np.random.shuffle(indices)
                        split_idx = int(len(X) * 0.9)
                        train_idx, val_idx = indices[:split_idx], indices[split_idx:]
                        
                        X_train_t = torch.tensor(X[train_idx], dtype=torch.float32, device=self.device)
                        Y_train_t = torch.tensor(Y_scaled[train_idx], dtype=torch.float32, device=self.device)
                        X_val_t = torch.tensor(X[val_idx], dtype=torch.float32, device=self.device)
                        Y_val_t = torch.tensor(Y_scaled[val_idx], dtype=torch.float32, device=self.device)
                        
                        # Randomly generate hyperparameter combinations (Expanded Search Space)
                        random.seed(42)
                        trials = []
                        for _ in range(self.n_trials):
                            n_layers = random.choice([2, 3, 4, 5])
                            hidden_layers = [random.choice([64, 128, 256, 512, 1024]) for _ in range(n_layers)]
                            hidden_layers.sort(reverse=True) # Wide at start, narrow at end
                            lr = random.choice([0.01, 0.005, 0.001, 0.0005])
                            dropout = random.choice([0.0, 0.1, 0.2, 0.3, 0.4])
                            trials.append({'hidden_layers': hidden_layers, 'lr': lr, 'dropout': dropout})
                            
                        best_val_loss = float('inf')
                        best_model_state = None
                        
                        print(f"      [AF Predictor] Starting HP Search ({self.n_trials} trials) on {self.device}...")
                        criterion = torch.nn.MSELoss()
                        
                        for i, hp in enumerate(trials):
                            model = AFPredictorMLP(X.shape[1], Y.shape[1], 
                                                   hidden_layers=hp['hidden_layers'], 
                                                   dropout_rate=hp['dropout']).to(self.device)
                            optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'])
                            
                            for epoch in range(self.epochs):
                                model.train()
                                optimizer.zero_grad()
                                pred = model(X_train_t)
                                loss = criterion(pred, Y_train_t)
                                loss.backward()
                                optimizer.step()
                                
                            model.eval()
                            with torch.no_grad():
                                val_pred = model(X_val_t)
                                val_loss = criterion(val_pred, Y_val_t).item()
                                
                            print(f"        Trial {i+1}/{len(trials)} | Layers: {hp['hidden_layers']}, LR: {hp['lr']}, Drop: {hp['dropout']} -> Val Loss: {val_loss:.4f}")
                            
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                best_model_state = copy.deepcopy(model.state_dict())
                                self.best_model = model
                                
                        print(f"      [AF Predictor] ★ Selected Best Model with Val Loss: {best_val_loss:.4f} ★")
                        self.best_model.load_state_dict(best_model_state)
                        
                    def predict(self, X):
                        self.best_model.eval()
                        with torch.no_grad():
                            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
                            pred_scaled = self.best_model(X_t).cpu().numpy()
                            return self.y_scaler.inverse_transform(pred_scaled)

                self.af_predictor = TorchAFPredictor()
                self.af_predictor.fit(scatter_neg_scaled, X_neg)
            else:
                from sklearn.neural_network import MLPRegressor
                from sklearn.pipeline import Pipeline
                self.af_predictor = Pipeline([
                    ('scaler', StandardScaler()),
                    ('mlp', MLPRegressor(
                        hidden_layer_sizes=(128, 128),
                        activation='relu',
                        solver='adam',
                        max_iter=500,
                        random_state=42,
                        early_stopping=True,
                        validation_fraction=0.1,
                        verbose=True
                    ))
                ])
                # pipeline in sklearn handles scaling of X, but for Y we must use TransformedTargetRegressor
                from sklearn.compose import TransformedTargetRegressor
                self.af_predictor = TransformedTargetRegressor(
                    regressor=self.af_predictor.named_steps['mlp'],
                    transformer=StandardScaler()
                )
                self.af_predictor.fit(scatter_neg_scaled, X_neg)
                print("      [AF Predictor] Trained sklearn MLPRegressor (with Target Scaling) on FSC/SSC features.")

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

    def _unmix(self, X, scatter_val=None):
        """method パラメータに応じてアンミキシング手法を切り替える."""
        if self.method == 'poisson_glm':
            return self._unmix_poisson_glm(X, scatter_val=scatter_val)
        elif self.method == 'poisson':
            return self._unmix_poisson_irls(X, scatter_val=scatter_val)
        else:
            return self._unmix_ols_nnls(X, scatter_val=scatter_val)

    def _unmix_ols_nnls(self, X, scatter_val=None):
        """非負制約付き最小二乗法 (NNLS) によるアンミキシング.
        全波長を均等に評価(OLS)しつつ、マイナスの値を厳格に禁止する.
        """
        from scipy.optimize import nnls
        from joblib import Parallel, delayed
        import numpy as np
        
        # 参照スペクトル行列を作成 (M: [73, 2])
        M = np.column_stack([self.S_AF, self.S_Stain])
        
        # 全細胞(N個)に対して個別にNNLSを適用 (並列処理で高速化)
        def fit_cell(y):
            c, _ = nnls(M, y)
            return c
            
        # n_jobs=-1 でCPUの全コアを使って一気に計算
        C = Parallel(n_jobs=-1)(delayed(fit_cell)(y) for y in X)
        return np.array(C)

    def _unmix_poisson_irls(self, X, scatter_val=None):
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

        dynamic_AF = False
        if hasattr(self, 'af_predictor') and self.af_predictor is not None and scatter_val is not None:
            if len(scatter_val) == X.shape[0]:
                scatter_scaled = self.scatter_scaler.transform(scatter_val)
                S_AF_pred = self.af_predictor.predict(scatter_scaled)
                S_AF_pred = np.maximum(S_AF_pred, 0)
                sums = np.sum(S_AF_pred, axis=1, keepdims=True) + 1e-9
                S_AF_pred = S_AF_pred / sums
                dynamic_AF = True

        # ---- Step 1: c_af from tail channels (Poisson IRLS) ----
        X_tail = X[:, tail_mask]  # (N, T)

        if dynamic_AF:
            c_af = np.sum(S_AF_pred, axis=1)
            residual = X - S_AF_pred
        else:
            S_AF_tail = S_AF[tail_mask]
            denom_af_init = np.dot(S_AF_tail, S_AF_tail) + 1e-9
            c_af = X_tail @ S_AF_tail / denom_af_init  # (N,)

            for _ in range(self.irls_iter):
                predicted_tail = np.maximum(c_af[:, None] * S_AF_tail[None, :], eps)  # (N, T)
                W = 1.0 / predicted_tail  # (N, T)
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

    def _unmix_poisson_glm(self, X, scatter_val=None):
        """
        論文に準拠したポアソンデビアンス最小化によるアンミキシング (GLMアプローチ).
        PyTorchがインストールされている場合は、GPUを用いた一括バッチ最適化を行い劇的に高速化する。
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            has_torch = True
        except ImportError:
            has_torch = False

        if has_torch:
            return self._unmix_poisson_glm_torch(X, scatter_val=scatter_val)
        else:
            return self._unmix_poisson_glm_scipy(X, scatter_val=scatter_val)

    def _unmix_poisson_glm_torch(self, X, scatter_val=None):
        import torch
        import torch.nn as nn
        import torch.optim as optim

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        epsilon = 1e-6
        lambda_param = 0.0
        batch_size = 10000
        n_samples = X.shape[0]
        alpha_final_list = []

        # Predict dynamic AF if model is available
        dynamic_AF = False
        if hasattr(self, 'af_predictor') and self.af_predictor is not None and scatter_val is not None:
            if len(scatter_val) == n_samples:
                scatter_scaled = self.scatter_scaler.transform(scatter_val)
                S_AF_pred = self.af_predictor.predict(scatter_scaled)
                # Do NOT normalize. We want the exact raw predicted intensity & shape
                S_AF_pred = np.maximum(S_AF_pred, 0)
                dynamic_AF = True

        # Fallback static matrix
        M_np_static = np.maximum(self.S, 0)
        M_t_static = torch.tensor(M_np_static, dtype=torch.float64, device=device)

        # バッチ分割によるメモリ不足(OOM)回避
        for i in range(0, n_samples, batch_size):
            X_batch_np = X[i:i+batch_size]
            X_batch_t = torch.tensor(X_batch_np, dtype=torch.float64, device=device)
            X_safe = torch.clamp(X_batch_t, min=0.0) + epsilon
            
            if dynamic_AF:
                batch_S_AF_pred = torch.tensor(S_AF_pred[i:i+batch_size], dtype=torch.float64, device=device)
                S_Stain_t = torch.tensor(self.S_Stain, dtype=torch.float64, device=device)
                
                # We fix AF completely. We only fit c_stain (alpha has 1 dimension)
                # Initial estimate for c_stain
                R_np = X_batch_np - S_AF_pred[i:i+batch_size]
                S_Stain_peak = self.S_Stain[self._peak_mask]
                R_peak = R_np[:, self._peak_mask]
                denom = np.dot(S_Stain_peak, S_Stain_peak) + 1e-9
                c_stain_init = R_peak @ S_Stain_peak / denom
                alpha_init_np = np.clip(c_stain_init, 1e-3, None)[:, None] # (batch, 1)
                alpha_init = torch.tensor(alpha_init_np, dtype=torch.float64, device=device)
            else:
                M_init_np = M_np_static
                M_batch_t = M_t_static
                
                # 初期値の決定: numpyのlstsqで近似 (SVDの収束エラー回避)
                try:
                    alpha_init_np, _, _, _ = np.linalg.lstsq(M_init_np, X_batch_np.T, rcond=None)
                    alpha_init_np = alpha_init_np.T
                except np.linalg.LinAlgError:
                    # SVD convergence error fallback
                    alpha_init_np = np.zeros((X_batch_np.shape[0], M_init_np.shape[1]))
                    intensity_scale = np.maximum(X_batch_np.sum(axis=1), 0) / np.maximum(M_init_np.sum(), 1.0)
                    for j in range(M_init_np.shape[1]):
                        alpha_init_np[:, j] = intensity_scale
                    
                alpha_init_np = np.clip(alpha_init_np, 1e-3, None)
                alpha_init = torch.tensor(alpha_init_np, dtype=torch.float64, device=device)

            # Softplus関数を用いた非負制約の導入 (alpha = softplus(w))
            w_init = torch.where(
                alpha_init > 20.0, 
                alpha_init, 
                torch.log(torch.exp(alpha_init) - 1.0 + 1e-9)
            )
            w = nn.Parameter(w_init)

            optimizer = optim.Adam([w], lr=0.1)
            
            prev_loss = float('inf')
            patience = 5
            patience_counter = 0
            
            # 最大反復回数を設定し、Early Stoppingを導入
            for epoch in range(500):
                optimizer.zero_grad()
                # w から非負の alpha を生成
                alpha = torch.nn.functional.softplus(w)
                
                if dynamic_AF:
                    # alpha shape: (batch, 1)
                    # M_alpha = batch_S_AF_pred + alpha * S_Stain
                    M_alpha = batch_S_AF_pred + alpha * S_Stain_t
                else:
                    # M_alpha = alpha @ M.T
                    M_alpha = torch.matmul(alpha, M_batch_t.T)
                    
                M_alpha = torch.clamp(M_alpha, min=0.0) + epsilon
                
                # ポアソンデビアンスの主要項: sum( -r * log(Mα) + Mα )
                deviance_term = torch.sum(-X_safe * torch.log(M_alpha) + M_alpha)
                penalty_term = -lambda_param * torch.sum(alpha)
                
                loss = 2.0 * deviance_term + penalty_term
                loss.backward()
                optimizer.step()
                
                # Early Stoppingの判定
                current_loss = loss.item()
                if abs(prev_loss - current_loss) < 1e-4 * abs(prev_loss + 1e-9):
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
                else:
                    patience_counter = 0
                prev_loss = current_loss

            if dynamic_AF:
                alpha_stain_final = torch.nn.functional.softplus(w).detach().cpu().numpy() # (batch, 1)
                c_af_final = np.sum(S_AF_pred[i:i+batch_size], axis=1, keepdims=True) # (batch, 1)
                alpha_batch_final = np.hstack([c_af_final, alpha_stain_final]) # (batch, 2)
            else:
                alpha_batch_final = torch.nn.functional.softplus(w).detach().cpu().numpy()
            alpha_final_list.append(alpha_batch_final)
            
        return np.vstack(alpha_final_list)

    def _unmix_poisson_glm_scipy(self, X, scatter_val=None):
        """
        PyTorchがない場合のフォールバック用 (Scipy L-BFGS-B を使用したループ処理)
        """
        M_static = np.maximum(self.S, 0)
        lambda_param = 0.0
        epsilon = 1e-6
        
        dynamic_AF = False
        if hasattr(self, 'af_predictor') and self.af_predictor is not None and scatter_val is not None:
            if len(scatter_val) == X.shape[0]:
                scatter_scaled = self.scatter_scaler.transform(scatter_val)
                S_AF_pred = self.af_predictor.predict(scatter_scaled)
                S_AF_pred = np.maximum(S_AF_pred, 0)
                sums = np.sum(S_AF_pred, axis=1, keepdims=True) + 1e-9
                S_AF_pred = S_AF_pred / sums
                dynamic_AF = True
        
        def optimize_single_cell(i, r):
            if dynamic_AF:
                M_i = np.column_stack((S_AF_pred[i], self.S_Stain))
            else:
                M_i = M_static
                
            alpha_0, _ = nnls(M_i, r)
            
            def objective(alpha):
                r_safe = np.maximum(r, 0) + epsilon
                M_alpha = np.maximum(M_i @ alpha, 0) + epsilon
                deviance_term = np.sum(-r_safe * np.log(M_alpha) + M_alpha)
                penalty_term = -lambda_param * np.sum(alpha)
                return 2.0 * deviance_term + penalty_term
                
            bounds = [(0, None), (0, None)]
            res = minimize(objective, alpha_0, method='L-BFGS-B', bounds=bounds)
            return res.x
            
        # joblibのプロセス通信オーバーヘッドおよびAnacondaでのデッドロックを避けるため、
        # 単純なリスト内包表記を使用する (各セルの計算が1ms未満であるためこちらの方が圧倒的に高速)
        results = [optimize_single_cell(i, r) for i, r in enumerate(X)]
        return np.array(results)

    def transform(self, X, scatter_val=None):
        """アンミキシングを実行し、漏れ込み補正済みの係数を返す.

        Returns
        -------
        C_af : ndarray, shape (n,)
            自家蛍光の強度係数.
        C_stain_corrected : ndarray, shape (n,)
            漏れ込み補正済みの色素強度係数.
        """
        C = self._unmix(X, scatter_val=scatter_val)
        C_af = C[:, 0]
        C_stain = C[:, 1]
        C_stain_corrected = C_stain - self.slope * C_af - self.bg
        return C_af, C_stain_corrected

    def get_raw_coefficients(self, X, scatter_val=None):
        """補正前の生の係数 [c_af, c_stain] を返す."""
        return self._unmix(X, scatter_val=scatter_val)

    def remove_stain_component(self, X, scatter_val=None):
        """スペクトルから色素成分を除去し、純粋な自家蛍光スペクトルを返す."""
        C = self._unmix(X, scatter_val=scatter_val)
        return X - C[:, 1][:, None] * self.S_Stain[None, :]

def run_unmixing_group(results_base_dir, stain_name, method='poisson', retrain=False):
    neg_csv_paths = glob.glob(os.path.join(results_base_dir, "Negative_*", "*_wavelength.csv")) + \
                    glob.glob(os.path.join(results_base_dir, "negative_*", "*_wavelength.csv"))
    stain_csv_paths = glob.glob(os.path.join(results_base_dir, f"{stain_name}_*", "*_wavelength.csv")) + \
                      glob.glob(os.path.join(results_base_dir, f"{stain_name.lower()}_*", "*_wavelength.csv")) + \
                      glob.glob(os.path.join(results_base_dir, f"{stain_name.upper()}_*", "*_wavelength.csv"))
    
    neg_csv_paths = sorted(list(set([p for p in neg_csv_paths if "scarf_embeddings" not in p])))
    stain_csv_paths = sorted(list(set([p for p in stain_csv_paths if "scarf_embeddings" not in p])))
    
    if not neg_csv_paths or not stain_csv_paths:
        print("    Warning: Missing CSV files for unmixing.")
        return
        
    print("Loading unmixing data for fit...")
    X_neg_all = []
    scatter_neg_all = []
    for path in neg_csv_paths:
        df = pd.read_csv(path)
        wl_features = get_spectral_features(df)
        scat_features = get_scatter_features(df)
        X_neg_all.append(df[wl_features].values)
        if len(scat_features) > 0:
            scatter_neg_all.append(df[scat_features].values)
    X_neg_all = np.vstack(X_neg_all)
    if len(scatter_neg_all) > 0:
        scatter_neg_all = np.vstack(scatter_neg_all)
    else:
        scatter_neg_all = None
    
    X_stain_all = []
    for path in stain_csv_paths:
        df = pd.read_csv(path)
        wl_features = get_spectral_features(df)
        X_stain_all.append(df[wl_features].values)
    X_stain_all = np.vstack(X_stain_all)
    
    # --- Downsample for training (max 10000 cells) ---
    MAX_TRAIN_CELLS = 10000
    
    neg_idx = np.arange(len(X_neg_all))
    if len(X_neg_all) > MAX_TRAIN_CELLS:
        rng = np.random.default_rng(42)
        neg_idx = rng.choice(len(X_neg_all), MAX_TRAIN_CELLS, replace=False)
        X_neg_all = X_neg_all[neg_idx]
        if scatter_neg_all is not None:
            scatter_neg_all = scatter_neg_all[neg_idx]
        print(f"    Downsampled Negative to {MAX_TRAIN_CELLS} cells for training.")

    stain_idx = np.arange(len(X_stain_all))
    if len(X_stain_all) > MAX_TRAIN_CELLS:
        rng = np.random.default_rng(42)
        stain_idx = rng.choice(len(X_stain_all), MAX_TRAIN_CELLS, replace=False)
        X_stain_all = X_stain_all[stain_idx]
        print(f"    Downsampled Stain to {MAX_TRAIN_CELLS} cells for training.")
    
    # We extract date_str from the first neg_csv_path
    parts_neg = os.path.normpath(neg_csv_paths[0]).split(os.sep)
    date_str = parts_neg[-3]
    
    unmixer = get_unmixer(method, X_neg_all, X_stain_all, date_str=date_str, retrain=retrain, scatter_neg=scatter_neg_all)
        
    print(f"    Calculated autoflour leakage slope: {unmixer.slope:.6f}")
    
    peak_af_idx = np.argmax(unmixer.S_AF)
    peak_stain_idx = np.argmax(unmixer.S_Stain)
    
    for path in neg_csv_paths + stain_csv_paths:
        df = pd.read_csv(path)
        wl_features = get_spectral_features(df)
        scat_features = get_scatter_features(df)
        X_val = df[wl_features].values
        
        scatter_val = None
        if len(scat_features) > 0:
            scatter_val = df[scat_features].values
            
        raw_af_vals = X_val[:, peak_af_idx]
        raw_stain_vals = X_val[:, peak_stain_idx]
        
        af, stain_corr = unmixer.transform(X_val, scatter_val=scatter_val)

        
        df['Unmixed_AF'] = np.maximum(af, 0)
        df[f'Unmixed_{stain_name}'] = np.maximum(stain_corr, 0)
        
        # De-fragmentation warning workaround
        df = df.copy()
        df.to_csv(path, index=False)
        
        plot_path = os.path.join(os.path.dirname(path), "unmixing_scatter.png")
        save_unmixing_plot(raw_af_vals, raw_stain_vals, df['Unmixed_AF'].values, df[f'Unmixed_{stain_name}'].values, stain_name, plot_path, os.path.basename(path))
        print(f"    Unmixing complete: {os.path.basename(path)}")
