import os
import pandas as pd

def get_unmixer(method, X_neg, X_stain, 
                date_str=None, neg_label=None, stain_label=None, 
                retrain=False, **kwargs):
    """
    指定されたアルゴリズム(method)のUnmixerを初期化し、
    モデルの学習やロードを完了した状態のインスタンスを返すファクトリー関数。
    
    Parameters:
        method (str): 'poisson', 'poisson_glm', 'autoencoder', 'transformer', 'scarf'
        X_neg (ndarray): ネガティブコントロールのスペクトル
        X_stain (ndarray): 染色サンプルのスペクトル
        date_str (str): プロジェクト結果ディレクトリの日付文字列（モデルパス検索用）
        neg_label (str): Negativeサンプルのラベル（SCARF用）
        stain_label (str): Stainサンプルのラベル（SCARF用）
        retrain (bool): キャッシュを無視して再学習するかどうか
        **kwargs: Transformerなどのハイパーパラメータ
    """
    
    # プロジェクトルートを取得
    # src/unmix_factory.py の親の親がルート
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if method in ['poisson', 'poisson_glm']:
        from src.unmix_spectral import PoissonUnmixer
        # CPU実行時の高速化のため、初期推定(max_iter=0)のみを使用し反復最適化をスキップ
        unmixer = PoissonUnmixer(method=method, max_iter=0)
        unmixer.fit(X_neg, X_stain)
        
    elif method == 'autoencoder':
        from src.unmix_autoencoder import AutoEncoderUnmixer
        unmixer = AutoEncoderUnmixer()
        unmixer.fit(X_neg, X_stain)
        
        if date_str:
            model_path = os.path.join(PROJECT_ROOT, "analysis", "results", date_str, "ae_model.pth")
            if os.path.exists(model_path) and not retrain:
                unmixer.load_model(model_path)
            else:
                if not retrain:
                    print(f"Warning: Missing AE model at {model_path}. Using random init.")
                    
    elif method == 'transformer':
        from src.unmix_autoencoder_v2 import TransformerAutoEncoderUnmixer
        unmixer = TransformerAutoEncoderUnmixer(**kwargs)
        unmixer.fit(X_neg, X_stain)
        
        if date_str:
            model_path = os.path.join(PROJECT_ROOT, "analysis", "results", date_str, "transformer_ae_model.pth")
            if os.path.exists(model_path) and not retrain:
                unmixer.load_model(model_path)
            else:
                if not retrain:
                    print(f"Warning: Missing TransformerAE model at {model_path}. Using random init.")
                    
    elif method == 'scarf':
        from src.unmix_scarf import ScarfKnnUnmixer
        unmixer = ScarfKnnUnmixer(k_neighbors=10)
        unmixer.fit(X_neg, X_stain)
        
        if date_str and neg_label and stain_label:
            emb_neg_path = os.path.join(PROJECT_ROOT, "learning", "results", date_str, neg_label, f"{neg_label}_scarf_embeddings.csv")
            emb_stain_path = os.path.join(PROJECT_ROOT, "learning", "results", date_str, stain_label, f"{stain_label}_scarf_embeddings.csv")
            
            if os.path.exists(emb_neg_path) and os.path.exists(emb_stain_path) and not retrain:
                emb_neg = pd.read_csv(emb_neg_path).values
                emb_stain = pd.read_csv(emb_stain_path).values
                unmixer.fit_knn(emb_neg, X_neg)
                
                S_AF_personalized = unmixer.get_personalized_saf(emb_stain)
                unmixer._S_AF_personalized = S_AF_personalized  # 保存して外部から使えるようにする
            else:
                if not retrain:
                    print("Warning: Missing embeddings for SCARF. Falling back to simple fit.")
    else:
        raise ValueError(f"Unknown unmixing method: {method}")

    return unmixer
