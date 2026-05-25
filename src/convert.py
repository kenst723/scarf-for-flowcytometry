"""
.sraw → CSV 変換モジュール

Sony スペクトラルフローサイトメーターの .sraw ファイルを解析し、
6系列 (Area/Height/Width × Channel/Wavelength) のCSVに変換する。

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
    columns = []

    # 1. Area(Channel)
    for ch in channel_names:
        columns.append(f'Area_{ch}')
    # 2. Height(Channel)
    for ch in channel_names:
        columns.append(f'Height_{ch}')
    # 3. Width(Channel)
    for ch in channel_names:
        columns.append(f'Width_{ch}')

    # 4. Area(Wavelength)
    for wl in wavelengths:
        columns.append(f'Area_{wl:.1f}nm')
    # 5. Height(Wavelength)
    for wl in wavelengths:
        columns.append(f'Height_{wl:.1f}nm')
    # 6. Width(Wavelength)
    for wl in wavelengths:
        columns.append(f'Width_{wl:.1f}nm')

    # Raw (Channel) データ抽出
    area_ch = data[:, 0, :]
    height_ch = data[:, 1, :]
    width_ch = data[:, 2, :]

    # Normalized (Wavelength) データ抽出
    area_wl = wavelength_data[:, 0, :]
    height_wl = wavelength_data[:, 1, :]
    width_wl = wavelength_data[:, 2, :]

    # 6系列を結合
    flat_data = np.hstack([area_ch, height_ch, width_ch, area_wl, height_wl, width_wl])

    df = pd.DataFrame(flat_data, columns=columns)
    df.insert(0, 'event_id', range(num_events))

    # 出力先の決定
    if output_dir is None:
        output_dir = RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    csv_name = f'{base_name}_{timestamp}.csv'
    csv_path = os.path.join(output_dir, csv_name)
    df.to_csv(csv_path, index=False)

    return csv_path, df


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
        csv_path, df = convert_sraw_to_csv(filepath, output_dir=args.output_dir)
        print(f'Processed {filename} -> {os.path.basename(csv_path)}')
        print(f'  Output: {csv_path}')
        print(f'  Shape: {df.shape}')


if __name__ == '__main__':
    main()
