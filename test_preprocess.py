import os
import tempfile
import pandas as pd
import unittest

from preprocess import preprocess_data

class TestPreprocessData(unittest.TestCase):
    def setUp(self):
        # テスト用の小さなCSVデータを作成
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_path = os.path.join(self.temp_dir.name, "input.csv")
        self.output_path = os.path.join(self.temp_dir.name, "output.csv")

        data = [
            [2024, 1, 1, 1, "東京", 1, 1, "レース", 1, "芝", 1, 1200, "良", "ウマ", "牡", 3, "騎手", 55,
             10, 1, 1, 1, "", "", 1.0, "1:10.0", "", "", 1, 1, 1, 1, "", 480, "調教師", "", 0,
             "", "", "", "0001A1", "馬主", "生産者", "父", "母", "父父", "鹿毛", "2019-01-01", 2.3,
             "", "", ""]
        ]
        df = pd.DataFrame(data)
        df.to_csv(self.input_path, index=False, header=False, encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_preprocess_creates_output(self):
        preprocess_data(self.input_path, self.output_path)
        self.assertTrue(os.path.exists(self.output_path))
        df = pd.read_csv(self.output_path)
        # 確定着順が整数で保存されているか確認
        self.assertEqual(df.loc[0, "確定着順"], 1)

if __name__ == "__main__":
    unittest.main()
