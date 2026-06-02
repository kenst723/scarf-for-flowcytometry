"""
プロジェクト共通設定

パス定義・定数・ヘルパー関数をここに集約する。
新しい実験データを追加する際はこのファイルだけ変更すればよい。
"""

import os

# ---------------------------------------------------------------------------
# プロジェクトルート
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# ディレクトリ
# ---------------------------------------------------------------------------
SCARF_EPOCHS = 10
SCARF_BATCH_SIZE = 128
SCARF_LEARNING_RATE = 0.001
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "analysis", "results")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# ---------------------------------------------------------------------------
# 実験データ定義
# ---------------------------------------------------------------------------
# data/ 以下のフォルダ名 → 正規化した日付文字列のマッピング
EXPERIMENTS = {
    "Experiment 2026!06!02 12!39": "Experiment 2026!06!02 12!39"
}

# ---------------------------------------------------------------------------
# 波長帯域幅の正規化乗数 (Sony SA3800 固有)
# ---------------------------------------------------------------------------
WAVELENGTH_MULTIPLIERS = [
    2.352933, 2.352944, 2.352939, 2.162156, 2.105267, 1.904761, 1.860462, 1.702126,
    1.632656, 1.509433, 1.428573, 1.333333, 1.269838, 1.212123, 1.142858, 1.066665,
    1.025644, 0.97561,  0.909088, 0.869567, 0.816325, 0.769229, 0.727273, 0.68968,
    0.655738, 0.625005, 0.601506, 0.575537, 0.555556, 0.529801, 0.503132, 0.484842,
    0.45715,  0.439574,
]

# ArcSinh 変換の cofactor
COFACTOR = 150.0


# ---------------------------------------------------------------------------
# ヘルパー関数
# ---------------------------------------------------------------------------
def get_experiment_data_dir(experiment_folder: str) -> str:
    """実験フォルダのフルパスを返す。"""
    return os.path.join(DATA_DIR, experiment_folder)


def get_results_dir(experiment_folder: str, sample_label: str) -> str:
    """
    結果出力先ディレクトリのパスを返す。存在しなければ作成する。

    Parameters
    ----------
    experiment_folder : str
        data/ 以下のフォルダ名 (例: "Experiment 2026!05!21 15!59")
    sample_label : str
        サンプルのラベル (例: "PI_A01")

    Returns
    -------
    str
        結果ディレクトリのフルパス (例: results/2026-05-21/PI_A01/)
    """
    date_str = EXPERIMENTS.get(experiment_folder, experiment_folder)
    result_dir = os.path.join(RESULTS_DIR, date_str, sample_label)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def find_sraw_files(directory: str) -> list:
    """指定ディレクトリ内の .sraw ファイルパスのリストを返す。"""
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"ディレクトリが見つかりません: {directory}")
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.endswith(".sraw")
    ]
