import os
import pandas as pd

def get_unmixer(method, X_neg, X_stain, 
                date_str=None, neg_label=None, stain_label=None, 
                retrain=False, scatter_neg=None, **kwargs):
    """
    指定されたアルゴリズム(method)のUnmixerを初期化し、
    モデルの学習やロードを完了した状態のインスタンスを返すファクトリー関数。
    
    Parameters:
        method (str): 'poisson', 'poisson_glm', 'ols'
        X_neg (ndarray): ネガティブコントロールのスペクトル
        X_stain (ndarray): 染色サンプルのスペクトル
        date_str (str): プロジェクト結果ディレクトリの日付文字列（モデルパス検索用）
        neg_label (str): Negativeサンプルのラベル（SCARF用）
        stain_label (str): Stainサンプルのラベル（SCARF用）
        retrain (bool): キャッシュを無視して再学習するかどうか
        scatter_neg (ndarray): AF予測用の散乱光(FSC/SSC)データ
        **kwargs: Transformerなどのハイパーパラメータ
    """
    
    # プロジェクトルートを取得
    # src/unmix_factory.py の親の親がルート
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if method in ['poisson', 'poisson_glm', 'ols']:
        from src.unmix_spectral import PoissonUnmixer
        # CPU実行時の高速化のため、初期推定(max_iter=0)のみを使用し反復最適化をスキップ
        unmixer = PoissonUnmixer(method=method, max_iter=0)
        unmixer.fit(X_neg, X_stain, scatter_neg=scatter_neg)
        
    else:
        raise ValueError(f"Unknown method: {method}")
        
    return unmixer
