import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict
from sklearn.metrics import silhouette_score

def calculate_mmd(X: torch.Tensor, Y: torch.Tensor, sigma: float = None) -> Tuple[float, float]:
    """
    RBFカーネルを用いたMMD (Maximum Mean Discrepancy) の計算。
    元の高次元空間のまま分布間の距離を測ります。
    
    Args:
        X (torch.Tensor): データ群1 (形状: [n_samples_X, n_features])
        Y (torch.Tensor): データ群2 (形状: [n_samples_Y, n_features])
        sigma (float, optional): RBFカーネルの幅。Noneの場合はMedian heuristicにより自動決定。
        
    Returns:
        float: MMD値
        float: 計算に使用されたsigma値
    """
    # 距離計算の効率化のため cdist を用いてペアワイズユークリッド距離の平方を計算
    # 浮動小数点誤差による微小なマイナス値が出ないように clamp(0) をかける
    XX = torch.clamp(torch.cdist(X, X, p=2) ** 2, min=0.0)
    YY = torch.clamp(torch.cdist(Y, Y, p=2) ** 2, min=0.0)
    XY = torch.clamp(torch.cdist(X, Y, p=2) ** 2, min=0.0)
    
    # 【Median Heuristic による sigma の決定】
    if sigma is None:
        median_sq_dist = torch.median(XY)
        sigma = torch.sqrt(median_sq_dist / 2.0).item()
        
    gamma = 1.0 / (2 * (sigma ** 2) + 1e-8)
    
    # カーネル行列の計算
    K_XX = torch.exp(-gamma * XX)
    K_YY = torch.exp(-gamma * YY)
    K_XY = torch.exp(-gamma * XY)
    
    # MMDの計算 (経験推定値)
    mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    
    # 数値誤差で微小なマイナスになることを防ぐ
    return max(float(mmd.item()), 0.0), sigma


def evaluate_unmixing_mmd(
    X_unstain: np.ndarray, 
    X_af: np.ndarray, 
    n_samples: int = 2000, 
    n_permutations: int = 100,
    random_state: int = 42
) -> Dict[str, float]:
    """
    UnstainデータとアンミキシングAFデータのMMDを計算し、Null分布を用いた統計検定を行う。
    
    Args:
        X_unstain (np.ndarray): アンミキシング前のネガティブコントロール細胞スペクトル
        X_af (np.ndarray): アンミキシング後の染色サンプルの推定AF細胞スペクトル
        n_samples (int): 各群からサンプリングする細胞数（計算コスト削減のため）
        n_permutations (int): Null分布を作成するためのサンプリング繰り返し回数
        random_state (int): 乱数シード
        
    Returns:
        dict: 検定結果(target_mmd, null_mmds, p_value等)を含む辞書
    """
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_u = len(X_unstain)
    n_a = len(X_af)
    
    # サンプル数の安全処理 (Null分布を作るためにUnstainは最低でも2 * n_samples 必要)
    actual_sample = min(n_samples, n_u // 2)
    actual_sample_af = min(n_a, actual_sample)
    
    if actual_sample < n_samples:
        print(f"  [MMD Evaluation] Warning: Reduced sample size to {actual_sample} due to limited unstain data.")

    # PyTorchテンソルへの変換
    X_unstain_full = torch.tensor(X_unstain, dtype=torch.float32, device=device)
    X_af_full = torch.tensor(X_af, dtype=torch.float32, device=device)
    
    # ---------------------------------------------------------
    # 1. 比較対象MMDの計算 (Unstain vs 予測AF)
    # ---------------------------------------------------------
    idx_u = torch.randperm(n_u)[:actual_sample]
    idx_a = torch.randperm(n_a)[:actual_sample_af]
    
    X_u_sample = X_unstain_full[idx_u]
    X_a_sample = X_af_full[idx_a]
    
    target_mmd, sigma = calculate_mmd(X_u_sample, X_a_sample, sigma=None)
    
    # ---------------------------------------------------------
    # 2. ベースライン(Null) MMDの作成
    # ---------------------------------------------------------
    null_mmds = []
    
    for _ in range(n_permutations):
        # unstainの中から 2グループ分 のインデックスをランダムに取得
        idx_null = torch.randperm(n_u)[:actual_sample * 2]
        
        X_null_1 = X_unstain_full[idx_null[:actual_sample]]
        X_null_2 = X_unstain_full[idx_null[actual_sample:]]
        
        # スケールを統一するため、固定の sigma を使用する
        mmd_val, _ = calculate_mmd(X_null_1, X_null_2, sigma=sigma)
        null_mmds.append(mmd_val)
        
    null_mmds = np.array(null_mmds)
    
    # ---------------------------------------------------------
    # 3. 統計検定 (経験的p値の算出)
    # ---------------------------------------------------------
    p_value = np.mean(null_mmds >= target_mmd)
    # ---------------------------------------------------------
    # 4. シルエットスコア (Silhouette Score) の算出
    # ---------------------------------------------------------
    X_u_np = X_u_sample.cpu().numpy()
    X_a_np = X_a_sample.cpu().numpy()
    X_combined = np.vstack([X_u_np, X_a_np])
    labels = np.concatenate([np.zeros(len(X_u_np)), np.ones(len(X_a_np))])
    
    sil_score = silhouette_score(X_combined, labels)
    
    return {
        "target_mmd": target_mmd,
        "null_mmds": null_mmds,
        "p_value": p_value,
        "sigma": sigma,
        "n_samples": actual_sample,
        "silhouette_score": sil_score
    }


def plot_mmd_results(results: Dict[str, float], output_path: str = None):
    """
    MMDの計算結果（Null分布とターゲット値）をヒストグラムで可視化する。
    """
    target_mmd = results["target_mmd"]
    null_mmds = results["null_mmds"]
    p_value = results["p_value"]
    
    plt.figure(figsize=(10, 6), dpi=150)
    
    sns.histplot(null_mmds, bins=30, kde=True, color='skyblue', stat='density', 
                 label='Null Distribution\n(Unstain vs Unstain)')
    
    plt.axvline(target_mmd, color='red', linestyle='dashed', linewidth=2, 
                label=f'Target MMD: {target_mmd:.5f}\n(Unstain vs Unmixed AF)')
    
    title_color = 'red' if p_value < 0.05 else 'black'
    
    # シルエットスコアの取得（過去の互換性のためにgetを使用）
    sil_score = results.get("silhouette_score", None)
    if sil_score is not None:
        title_text = f"Evaluation (MMD p-value: {p_value:.4f} | Silhouette Score: {sil_score:.4f})"
    else:
        title_text = f"MMD Evaluation of Unmixing Quality (Empirical p-value: {p_value:.4f})"
        
    plt.title(title_text, fontsize=14, color=title_color)
    plt.xlabel("Maximum Mean Discrepancy (MMD)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()
