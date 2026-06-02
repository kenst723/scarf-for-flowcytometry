import numpy as np
from sklearn.neighbors import NearestNeighbors
from src.unmix_spectral import PoissonUnmixer

class ScarfKnnUnmixer(PoissonUnmixer):
    """
    SCARF Embeddingとk-NN探索を用いて、細胞ごとにカスタマイズされた
    自家蛍光参照スペクトル (S_AF_i) を推定し、Poisson IRLSでアンミキシングを行うヨ.
    """
    def __init__(self, k_neighbors=10, max_iter=5, tol=1e-4, irls_iter=3):
        super().__init__(max_iter=max_iter, tol=tol, method='poisson', irls_iter=irls_iter)
        self.k_neighbors = k_neighbors
        self.knn_model = None
        self.X_neg_raw = None
        
    def fit_knn(self, emb_neg, X_neg_raw):
        """
        NegativeサンプルのEmbeddingと生スペクトルを保存し、k-NNモデルを構築する.
        
        Parameters
        ----------
        emb_neg : ndarray, shape (N_neg, D)
            NegativeサンプルのSCARF Embedding.
        X_neg_raw : ndarray, shape (N_neg, M)
            Negativeサンプルの生スペクトル.
        """
        self.X_neg_raw = X_neg_raw
        self.knn_model = NearestNeighbors(n_neighbors=self.k_neighbors, algorithm='auto')
        self.knn_model.fit(emb_neg)
        
        # S_AFの初期値として全体の中央値を計算 (S_Stain精製などで使用するため)
        self.S_AF = np.median(self.X_neg_raw, axis=0)
        self.S_AF /= np.sum(self.S_AF)
        
    def get_personalized_saf(self, emb_stain):
        """
        染色細胞のEmbeddingを入力として、各細胞固有のS_AF_iを生成する.
        
        Parameters
        ----------
        emb_stain : ndarray, shape (N_stain, D)
            染色サンプルのSCARF Embedding.
            
        Returns
        -------
        S_AF_personalized : ndarray, shape (N_stain, M)
            細胞ごとに推定された純粋な自家蛍光スペクトル(各行が合計1に正規化されている).
        """
        if self.knn_model is None:
            raise ValueError("knn_model is not fitted. Call fit_knn first.")
            
        distances, indices = self.knn_model.kneighbors(emb_stain)
        
        # N_stain x k_neighbors x M
        nearest_spectra = self.X_neg_raw[indices]
        
        # k近傍の中央値をとり、細胞固有のS_AFを生成
        S_AF_personalized = np.median(nearest_spectra, axis=1)
        
        # 行ごとに正規化
        row_sums = np.sum(S_AF_personalized, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-9
        S_AF_personalized /= row_sums
        
        return S_AF_personalized

    def _unmix_poisson_irls_personalized(self, X, S_AF_personalized):
        """
        細胞ごとに異なる S_AF_personalized (N, M) を用いて Poisson IRLS を行う.
        """
        S_Stain = self.S_Stain
        tail_mask = self._tail_mask
        peak_mask = self._peak_mask
        eps = 1.0
        
        N = X.shape[0]

        # ---- Step 1: c_af from tail channels (Poisson IRLS) ----
        S_AF_tail = S_AF_personalized[:, tail_mask]  # (N, T)
        X_tail = X[:, tail_mask]  # (N, T)

        # 初期推定 (OLS)
        denom_af_init = np.sum(S_AF_tail * S_AF_tail, axis=1) + 1e-9  # (N,)
        c_af = np.sum(X_tail * S_AF_tail, axis=1) / denom_af_init  # (N,)

        for _ in range(self.irls_iter):
            predicted_tail = np.maximum(c_af[:, None] * S_AF_tail, eps)  # (N, T)
            W = 1.0 / predicted_tail  # (N, T)

            numerator = np.sum(W * X_tail * S_AF_tail, axis=1)    # (N,)
            denominator = np.sum(W * S_AF_tail ** 2, axis=1) + 1e-9  # (N,)
            c_af = numerator / denominator

        # ---- Step 2: c_stain from peak channels (Poisson IRLS) ----
        residual = X - c_af[:, None] * S_AF_personalized  # (N, M)
        S_Stain_peak = S_Stain[peak_mask]  # (P,)
        R_peak = residual[:, peak_mask]  # (N, P)

        # 初期推定 (OLS)
        denom_stain_init = np.dot(S_Stain_peak, S_Stain_peak) + 1e-9
        c_stain = R_peak @ S_Stain_peak / denom_stain_init  # (N,)

        for _ in range(self.irls_iter):
            predicted_peak = np.maximum(c_stain[:, None] * S_Stain_peak[None, :], eps)  # (N, P)
            W = 1.0 / predicted_peak  # (N, P)

            numerator = np.sum(W * R_peak * S_Stain_peak[None, :], axis=1)    # (N,)
            denominator = np.sum(W * S_Stain_peak[None, :] ** 2, axis=1) + 1e-9  # (N,)
            c_stain = numerator / denominator

        C = np.column_stack((c_af, c_stain))
        return C

    def transform_with_scarf(self, X, emb_stain):
        """
        SCARF Embeddingを用いたパーソナライズド・アンミキシングを実行し、
        漏れ込み補正済みの係数を返す.
        
        Parameters
        ----------
        X : ndarray, shape (N, M)
            染色サンプルの生スペクトル.
        emb_stain : ndarray, shape (N, D)
            染色サンプルのSCARF Embedding.
            
        Returns
        -------
        C_af : ndarray, shape (N,)
            自家蛍光の強度係数.
        C_stain_corrected : ndarray, shape (N,)
            漏れ込み補正済みの色素強度係数.
        """
        S_AF_personalized = self.get_personalized_saf(emb_stain)
        
        C = self._unmix_poisson_irls_personalized(X, S_AF_personalized)
        C_af = C[:, 0]
        C_stain = C[:, 1]
        C_stain_corrected = C_stain - self.slope * C_af - self.bg
        return C_af, C_stain_corrected
