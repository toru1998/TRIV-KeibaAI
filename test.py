from main import load_data

# データ読み込み
X, y, groups = load_data('data/2000_2024_fulldata.csv')

# 先頭20件のグループ（レースID）を表示
print("First 20 race IDs (groups):")
print(X.head(20))
