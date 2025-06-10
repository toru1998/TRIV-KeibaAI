import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from main import load_data


def train_baseline(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """ベースラインとなるロジスティック回帰モデルの学習"""
    X, y, _ = load_data(csv_path)

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ロジスティック回帰による学習
    model = LogisticRegression(max_iter=1000, multi_class="multinomial")
    model.fit(X_train_scaled, y_train)

    # 予測と評価
    preds = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    loss = log_loss(y_test, proba)

    print(f"Accuracy: {acc:.4f}")
    print(f"Log Loss: {loss:.4f}")

    return model


if __name__ == "__main__":
    # データファイルのパスは環境に合わせて変更してください
    train_baseline("data/2000_2024_fulldata.csv")
