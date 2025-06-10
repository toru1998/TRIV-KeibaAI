import argparse
import pandas as pd

# CSVのカラム名。main.pyと同じ定義を使用する
COLUMNS = [
    "年","月","日","回次","場所","日次","レース番号","レース名","クラスコード",
    "芝・ダ","トラックコード","距離","馬場状態","馬名","性別","年齢","騎手名","斤量",
    "頭数","馬番","確定着順","入線着順","異常コード","着差タイム","人気順",
    "走破タイム","走破時計","補正タイム","通過順1","通過順2","通過順3",
    "通過順4","上がり3Fタイム","馬体重","調教師","所属地","賞金",
    "血統登録番号","騎手コード","調教師コード","レースID","馬主名","生産者名",
    "父馬名","母馬名","母の父馬名","毛色","生年月日","単勝オッズ",
    "馬印","レース印","PCI"
]

def load_raw_data(path: str) -> pd.DataFrame:
    """CSVを読み込み、基本的な前処理を行ったDataFrameを返す"""
    df = pd.read_csv(path, encoding='utf-8', header=None, names=COLUMNS)
    df = df[df['確定着順'] != 0]
    df['確定着順'] = pd.to_numeric(df['確定着順'], errors='coerce')
    df.dropna(subset=['確定着順'], inplace=True)
    df['確定着順'] = df['確定着順'].astype(int)
    df['単勝オッズ'] = pd.to_numeric(df['単勝オッズ'], errors='coerce')
    return df

def filter_by_odds_and_rank(df: pd.DataFrame, min_odds: float) -> pd.DataFrame:
    """指定したオッズ以上かつ3着以内の馬を抽出する"""
    return df[(df['単勝オッズ'] >= min_odds) & (df['確定着順'] <= 3)]

def analyze_trends(df: pd.DataFrame) -> None:
    """簡単な傾向を表示する

    場所別およびクラスコード別の割合を計算して表示する。
    """

    # 場所ごとの割合を計算
    location_ratio = df['場所'].value_counts(normalize=True) * 100
    print('場所別割合(%):')
    print(location_ratio.round(2))

    # クラスコードごとの割合を計算
    class_ratio = df['クラスコード'].value_counts(normalize=True) * 100
    print('\nクラスコード別割合(%):')
    print(class_ratio.round(2))

    print('\n平均オッズ:', df['単勝オッズ'].mean())

def main() -> None:
    parser = argparse.ArgumentParser(description='オッズ条件付きデータ分析')
    parser.add_argument('csv_path', help='解析するCSVファイルへのパス')
    parser.add_argument('--min_odds', type=float, default=50.0, help='抽出する最小オッズ')
    args = parser.parse_args()

    df = load_raw_data(args.csv_path)
    filtered = filter_by_odds_and_rank(df, args.min_odds)
    print(f'抽出件数: {len(filtered)}')
    analyze_trends(filtered)

if __name__ == '__main__':
    main()
