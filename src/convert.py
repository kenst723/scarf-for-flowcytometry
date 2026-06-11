"""
.sraw → CSV 変換モジュール

Sony スペクトラルフローサイトメーターの .sraw ファイルを解析し、
4系列 (Area/Height × Channel/Wavelength) のCSVに変換する。

Usage:
    python -m src.convert                           # config.py のデフォルト設定で実行
    python -m src.convert --sraw-dir <path>         # 指定ディレクトリの .sraw を変換
    python -m src.convert --sraw-dir <path> --output-dir <path>
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import glob
import fcsparser

# プロジェクトルートを path に追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WAVELENGTH_MULTIPLIERS, RESULTS_DIR, find_sraw_files
from src.parse_sraw import parse_sraw_interleaved


def convert_sraw_to_csv(filepath, output_dir=None):
    """
    .sraw ファイルを CSV に変換する。

    Parameters
    ----------
    filepath : str
        .sraw ファイルのパス
    output_dir : str, optional
        出力先ディレクトリ。None の場合は RESULTS_DIR 直下に出力する。

    Returns
    -------
    str
        生成された CSV ファイルのパス
    """
    result = parse_sraw_interleaved(filepath)

    num_events = result['num_events']
    num_channels = result['num_channels']
    channel_names = result['channel_names']
    wavelengths = result['wavelengths']
    data = result['data']  # (num_events, 3, num_channels)

    # ---------------------------------------------------------
    # Wavelength 正規化
    # ---------------------------------------------------------
    multipliers = np.array(WAVELENGTH_MULTIPLIERS)
    wavelength_data = data * multipliers

    # ---------------------------------------------------------
    # カラム名の構築
    # ---------------------------------------------------------
    cols_ch = []
    cols_wl = []

    for ch in channel_names:
        cols_ch.append(f'Area_{ch}')
    for ch in channel_names:
        cols_ch.append(f'Height_{ch}')

    for wl in wavelengths:
        cols_wl.append(f'Area_{wl:.1f}nm')
    for wl in wavelengths:
        cols_wl.append(f'Height_{wl:.1f}nm')

    area_ch = data[:, 0, :]
    height_ch = data[:, 1, :]
    area_wl = wavelength_data[:, 0, :]
    height_wl = wavelength_data[:, 1, :]

    df_ch = pd.DataFrame(np.hstack([area_ch, height_ch]), columns=cols_ch)
    df_wl = pd.DataFrame(np.hstack([area_wl, height_wl]), columns=cols_wl)

    df_ch.insert(0, 'event_id', range(num_events))
    df_wl.insert(0, 'event_id', range(num_events))

    # ---------------------------------------------------------
    # FCS データのマージ (存在する場合)
    # ---------------------------------------------------------
    sraw_basename = os.path.splitext(os.path.basename(filepath))[0]
    sraw_dir = os.path.dirname(filepath)
    fcs_pattern = os.path.join(sraw_dir, f"{sraw_basename}*.fcs")
    fcs_files = glob.glob(fcs_pattern)
    
    if fcs_files:
        fcs_file = fcs_files[0]
        try:
            meta, df_fcs = fcsparser.parse(fcs_file, reformat_meta=True)
            # SRAW と FCS でイベント数が一致するか確認
            if len(df_fcs) == num_events:
                # 重複しそうなカラム名があれば適宜リネームするかプレフィックスをつける（FCS側はそのままでOK）
                # 'Time' などの不要な列は残して良い（後で除外されるため）
                # 指定された4つのパラメータのみを抽出
                target_cols = ['FSC - Area', 'FSC - Height', 'SSC - Area', 'SSC - Height']
                # FCSファイル内の実際のカラム名と一致するものだけを取得 (大文字小文字の違いなどを考慮)
                cols_to_keep = [c for c in df_fcs.columns if c in target_cols]
                
                df_fcs_subset = df_fcs[cols_to_keep]
                
                # 列方向に結合 (concatenate along columns)
                df_ch = pd.concat([df_fcs_subset, df_ch], axis=1)
                df_wl = pd.concat([df_fcs_subset, df_wl], axis=1)
                print(f"    Merged FCS data: {len(cols_to_keep)} columns added ({cols_to_keep}).")
            else:
                print(f"    Warning: Event count mismatch. SRAW={num_events}, FCS={len(df_fcs)}. Skipping FCS merge.")
        except Exception as e:
            print(f"    Warning: Failed to parse FCS file {fcs_file}: {e}")

    # 出力先の決定
    if output_dir is None:
        output_dir = RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    
    csv_path_ch = os.path.join(output_dir, f'{base_name}_{timestamp}_channel.csv')
    df_ch.to_csv(csv_path_ch, index=False)
    
    csv_path_wl = os.path.join(output_dir, f'{base_name}_{timestamp}_wavelength.csv')
    df_wl.to_csv(csv_path_wl, index=False)

    return csv_path_ch, csv_path_wl, df_ch, df_wl


def main():
    parser = argparse.ArgumentParser(description='.sraw → CSV 変換')
    parser.add_argument('--sraw-dir', type=str, default=None,
                        help='.sraw ファイルが格納されたディレクトリ')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='CSV の出力先ディレクトリ')
    args = parser.parse_args()

    if args.sraw_dir is None:
        print("Error: --sraw-dir を指定してください。")
        print("例: python -m src.convert --sraw-dir data/Experiment\\ 2026!05!21\\ 15!59/24\\ Tube\\ Rack\\ \\(5mL\\)\\ -\\ 1/PI")
        sys.exit(1)

    sraw_files = find_sraw_files(args.sraw_dir)

    if not sraw_files:
        print(f"Warning: {args.sraw_dir} に .sraw ファイルが見つかりません。")
        sys.exit(1)

    print(f"Found {len(sraw_files)} .sraw file(s) in {args.sraw_dir}")

    for filepath in sraw_files:
        filename = os.path.basename(filepath)
        csv_path_ch, csv_path_wl, df_ch, df_wl = convert_sraw_to_csv(filepath, output_dir=args.output_dir)
        print(f'Processed {filename} -> {os.path.basename(csv_path_ch)}, {os.path.basename(csv_path_wl)}')
        print(f'  Output Ch: {csv_path_ch} (Shape: {df_ch.shape})')
        print(f'  Output Wl: {csv_path_wl} (Shape: {df_wl.shape})')


if __name__ == '__main__':
    main()
