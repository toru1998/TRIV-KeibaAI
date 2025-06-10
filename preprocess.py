import argparse
import pandas as pd

# analysis.py と同じカラム定義
COLUMNS = [
    "年","月","日","回次","場所","日次","レース番号","レース名","クラスコード",
    "芝・ダ","トラックコード","距離","馬場状態","馬名","性別","年齢","騎手名","斤量",
    "頭数","馬番","確定着順","入線着順","異常コード","着差タイム","人気順",
    "走破タイム","走破時計","補正タイム","通過順1","通過順2","通過順3",
    "通過順4","上がり3Fタイム","馬体重","調教師","所属地","賞金",
    "血統登録番号","騎手コード","調教師コード","レースID","馬主名","生産者名",
    "父馬名","母馬名","母の父馬名","毛色","生年月日","単勝オッズ",
    "馬印","レース印","PCI",
]

def preprocess_data(input_path: str, output_path: str) -> None:
    """CSV ファイルを読み込み基本的な前処理を行う"""
    df = pd.read_csv(input_path, header=None, names=COLUMNS, encoding="utf-8")

    # 確定着順が 0 (出走取消など) の行を除去
    df = df[df["確定着順"] != 0]

    # 数値に変換できない確定着順は NaN にして削除
    df["確定着順"] = pd.to_numeric(df["確定着順"], errors="coerce")
    df.dropna(subset=["確定着順"], inplace=True)
    df["確定着順"] = df["確定着順"].astype(int)

    # 単勝オッズも数値に変換
    df["単勝オッズ"] = pd.to_numeric(df["単勝オッズ"], errors="coerce")

    # 前処理結果を保存
    df.to_csv(output_path, index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV の前処理スクリプト")
    parser.add_argument("input_csv", help="入力CSVファイルのパス")
    parser.add_argument("output_csv", help="出力CSVファイルのパス")
    args = parser.parse_args()

    preprocess_data(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
