"""
Spectral Unmixing (Autofluorescence Separation)
Negativeコントロールから自家蛍光スペクトルを抽出し、染色サンプルから対象色素のスペクトルを抽出し、
各細胞が持つ33チャネルの波長データから「自家蛍光強度」と「染色色素強度」を数学的に分離（アンミキシング）する。
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_spectral_features(df):
    """638.6nmを除外したスペクトルチャネルのリストを取得"""
    return [c for c in df.columns if c.startswith('Area_') and c.endswith('nm') and '638.6nm' not in c]

def compute_reference_spectra(neg_csv_paths, stain_csv_paths):
    """
    Negativeサンプルと染色サンプルから、それぞれの基準スペクトル（Reference Spectra）を計算する。
    """
    print("Computing Reference Spectra...")
    
    # --- 1. Autofluorescence (AF) Reference ---
    af_spectra_list = []
    for path in neg_csv_paths:
        df = pd.read_csv(path)
        wl_features = get_spectral_features(df)
        af_spectra_list.append(df[wl_features].values)
    
    X_neg_all = np.vstack(af_spectra_list)
    # 中央値スペクトルを計算し、合計が1になるように正規化
    S_AF = np.median(X_neg_all, axis=0)
    S_AF = S_AF / np.sum(S_AF)
    
    # --- 2. Stain (e.g. PI) Reference ---
    stain_spectra_list = []
    for path in stain_csv_paths:
        df = pd.read_csv(path)
        wl_features = get_spectral_features(df)
        stain_spectra_list.append(df[wl_features].values)
    
    X_stain_all = np.vstack(stain_spectra_list)
    
    # 染色細胞のうち、シグナルが極めて強い上位1%の細胞を抽出
    # （自家蛍光成分が相対的に無視できるほど染色シグナルが強い細胞）
    total_intensity = np.sum(X_stain_all, axis=1)
    threshold = np.percentile(total_intensity, 99)
    bright_cells = X_stain_all[total_intensity >= threshold]
    
    S_Stain = np.median(bright_cells, axis=0)
    # 純粋な色素スペクトルを得るため、AF成分を近似的に引く（簡略化のため定数引き算後にclip）
    S_Stain = np.maximum(S_Stain - (S_AF * np.min(S_Stain / (S_AF + 1e-9))), 0)
    S_Stain = S_Stain / np.sum(S_Stain)
    
    return S_AF, S_Stain

def apply_unmixing(csv_path, S_AF, S_Stain, stain_name="Stain"):
    """
    指定されたCSVファイルの全細胞に対してアンミキシングを適用し、結果を保存する。
    """
    df = pd.read_csv(csv_path)
    wl_features = get_spectral_features(df)
    X = df[wl_features].values
    
    # S行列 (33 x 2) の作成
    S = np.column_stack((S_AF, S_Stain))
    
    # 最小二乗法による分解 (OLS)
    # C = X * (S^T S)^-1 S^T
    pinv_S = np.linalg.pinv(S)  # 2 x 33
    C = X @ pinv_S.T            # (N x 33) @ (33 x 2) = (N x 2)
    
    # 物理的に負の発光はあり得ないため、0以下をクリップ (NNLSの近似)
    C = np.maximum(C, 0)
    
    # 新しいカラムとして追加
    df['Unmixed_AF'] = C[:, 0]
    df[f'Unmixed_{stain_name}'] = C[:, 1]
    
    # CSVを上書き保存
    df.to_csv(csv_path, index=False)
    
    # 散布図プロットの生成 (2D FACS Plot)
    plot_dir = os.path.dirname(csv_path)
    plot_path = os.path.join(plot_dir, "unmixing_scatter.png")
    
    # 見やすくするためにArcSinh変換してプロット (cofactor=150)
    af_plot = np.arcsinh(C[:, 0] / 150)
    stain_plot = np.arcsinh(C[:, 1] / 150)
    
    plt.figure(figsize=(6, 5))
    # 密度が高い部分を見やすくするため、半透明でプロット
    plt.scatter(af_plot, stain_plot, s=2, alpha=0.3, c='blue', edgecolors='none')
    plt.xlabel("Unmixed Autofluorescence (ArcSinh)")
    plt.ylabel(f"Unmixed {stain_name} (ArcSinh)")
    plt.title(f"Spectral Unmixing Result\n{os.path.basename(csv_path)}")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"    Unmixing complete: {os.path.basename(csv_path)}")
    print(f"    Saved scatter plot to {plot_path}")

def run_unmixing_group(results_base_dir, stain_name="PI"):
    """
    1つの実験グループに対してアンミキシングを実行するエントリーポイント
    results_base_dir: 例 'analysis/results/2026-05-21'
    """
    # NegativeサンプルのCSVを探す
    neg_csv_paths = glob.glob(os.path.join(results_base_dir, "Negative_*", "*.csv"))
    # 染色サンプルのCSVを探す
    stain_csv_paths = glob.glob(os.path.join(results_base_dir, f"{stain_name}_*", "*.csv"))
    
    # すでにSCARF特徴量などの余計なCSVが混ざらないようフィルタ
    neg_csv_paths = [p for p in neg_csv_paths if "scarf_embeddings" not in p]
    stain_csv_paths = [p for p in stain_csv_paths if "scarf_embeddings" not in p]
    
    if not neg_csv_paths or not stain_csv_paths:
        print("    Warning: Missing CSV files for unmixing.")
        return
        
    # リファレンススペクトルの計算
    S_AF, S_Stain = compute_reference_spectra(neg_csv_paths, stain_csv_paths)
    
    # 各サンプルのCSVにアンミキシングを適用して上書き
    for csv_path in neg_csv_paths + stain_csv_paths:
        apply_unmixing(csv_path, S_AF, S_Stain, stain_name=stain_name)
