import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss
import numpy as np

# --- 定数定義 ---
RANDOM_STATE = 42
N_TRIALS_OPTUNA = 3  # Optunaの試行回数
EARLY_STOPPING_ROUNDS = 10 #早期終了のラウンド数

# データ読み込み・前処理
def load_data(path: str):
    # カラム定義（ユーザー提供のものをそのまま使用）
    columns = [
        "年","月","日","回次","場所","日次","レース番号","レース名","クラスコード",
        "芝・ダ","トラックコード","距離","馬場状態","馬名","性別","年齢","騎手名","斤量",
        "頭数","馬番","確定着順","入線着順","異常コード","着差タイム","人気順",
        "走破タイム","走破時計","補正タイム","通過順1","通過順2","通過順3",
        "通過順4","上がり3Fタイム","馬体重","調教師","所属地","賞金",
        "血統登録番号","騎手コード","調教師コード","レースID","馬主名","生産者名",
        "父馬名","母馬名","母の父馬名","毛色","生年月日","単勝オッズ",
        "馬印","レース印","PCI"
    ]
    df = pd.read_csv(path, encoding='utf-8', header=None, names=columns)
    #出走取り消しした馬を削除（確定着順が0になっている）
    df = df[df['確定着順'] != 0]
    # 確定着順を数値型に変換、変換できないものはNaNとし、該当行を削除
    df['確定着順'] = pd.to_numeric(df['確定着順'], errors='coerce')
    df.dropna(subset=['確定着順'], inplace=True)
    df['確定着順'] = df['確定着順'].astype(int)

    # ラベル生成：1-3着はそのまま、4着以下をクラス4とする (1-indexed)
    df['label'] = df['確定着順'].apply(lambda x: x if x <= 3 else 4)
    
    # 特徴量として使用しないカラムのリスト
    # これらには、ターゲット変数、ID、結果そのもの、オッズ、高カーディナリティな文字列などが含まれる
    cols_to_drop_for_features = [
        "確定着順", "入線着順", "異常コード", "着差タイム", "人気順",
        "走破タイム", "走破時計", "補正タイム", "上がり3Fタイム", "賞金",
        "単勝オッズ", "馬印", "レース印", "PCI", # 結果やオッズ、予測指数など
        "label",      # 生成した目的変数
        "レースID",   # グループ情報として別途使用
        # 以下は文字列型が主でありselect_dtypes(include=np.number)で除外されるが、明示性のため記述
        "レース名", "馬名", "騎手名", "調教師", "所属地",
        "血統登録番号", "騎手コード", "調教師コード",
        "馬主名", "生産者名", "父馬名", "母馬名", "母の父馬名",
        "毛色", "生年月日"
        # 注意: "場所", "芝・ダ", "馬場状態", "性別", "クラスコード", "トラックコード" などは
        # 数値でエンコードされていれば特徴量に含まれる可能性がある。
        # これらをカテゴリ特徴として適切に扱うことで更なる改善が期待できる。
    ]

    features_df = df.drop(columns=cols_to_drop_for_features, errors='ignore')
    
    # 特徴量：数値型をそのまま、欠損は0埋め
    X = features_df.select_dtypes(include=np.number).fillna(0)
    X = X.loc[:, ~X.columns.duplicated()] # 重複したカラム名があれば削除

    # グループ情報（レース単位）
    df['レースID'] = df['レースID'].astype(str)
    # horse number appended to レースIDを削除し、レース固有IDのみ保持
    df['レースID'] = df['レースID'].str[:-2]
    groups = df['レースID']
    y = df['label']
    return X, y, groups

# Optuna 目的関数（Group-based hold-out + Early Stopping）
def objective(trial, X, y, groups):
    params = {
        'objective': 'multiclass',
        'num_class': 4, # クラス数 (0, 1, 2, 3)
        'metric': 'multi_logloss',
        'random_state': RANDOM_STATE,
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'n_jobs': -1,
        'verbose': -1, # LightGBMのログ出力を抑制
        'device_type': 'gpu'
    }
    # レースIDでグループ単位に分割
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, valid_idx = next(gss.split(X, y, groups))
    
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # LightGBMは0-indexedのラベルを期待するため、yから1を引く
    y_train_lgb = y_train - 1
    y_valid_lgb = y_valid - 1

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train_lgb,
        eval_set=[(X_valid, y_valid_lgb)],
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS,verbose=-1)],
        # categorical_feature='auto', # 数値のみなので基本不要だが、明示的にカテゴリ特徴を指定する場合に使う
    )
    proba = model.predict_proba(X_valid)
    n_classes = params['num_class']
    proba_full = np.zeros((proba.shape[0], n_classes))
    for idx, cls in enumerate(model.classes_):
        proba_full[:, cls] = proba[:, idx]
    return log_loss(y_valid_lgb, proba_full, labels=np.arange(n_classes))


def main():
    # データ読み込み
    X, y, groups = load_data('data/2000_2024_fulldata.csv') # ファイルパスは適宜変更してください

    # ハイパーパラメータ探索
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    study.optimize(lambda t: objective(t, X, y, groups), n_trials=N_TRIALS_OPTUNA)

    print('Best params:', study.best_params)
    print('Best Log Loss:', study.best_value)

    # 最適モデルで再学習
    best_params = study.best_params.copy()
    # objectiveとnum_class, random_stateは固定なので、best_paramsになくても追加/上書き
    best_params.update({
        'objective': 'multiclass',
        'num_class': 4,
        'metric': 'multi_logloss', # 評価指標も指定
        'random_state': RANDOM_STATE,
        'device_type': 'gpu', # GPUを使用する場合はこの行を追加
    })
    best_params['n_jobs'] = -1
    
    # グループ分割 (最終評価用)
    # Optunaでの分割とは独立したテストセットを作成することが望ましい
    # ここでは同じ random_state を使うが、データ全体に対する分割比率や回数を変えることを検討しても良い
    gss_final = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE + 1) # 念のため異なるseedで分割
    train_idx, test_idx = next(gss_final.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # LightGBMは0-indexedのラベルを期待するため、yから1を引く
    y_train_lgb = y_train - 1
    y_test_lgb = y_test - 1
    
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(
        X_train, y_train_lgb,
        eval_set=[(X_test, y_test_lgb)],
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=-1)],
        # categorical_feature='auto',
    )

    # 評価
    proba = final_model.predict_proba(X_test)
    n_classes = best_params['num_class']
    proba_full = np.zeros((proba.shape[0], n_classes))
    for idx, cls in enumerate(final_model.classes_):
        proba_full[:, cls] = proba[:, idx]
    final_loss = log_loss(y_test_lgb, proba_full, labels=np.arange(n_classes))
    print(f'Final Log Loss on Test Set: {final_loss:.4f}')

    # 確率表示
    cols = ['prob_1st', 'prob_2nd', 'prob_3rd', 'prob_4th_plus']
    df_res = pd.DataFrame(proba, columns=cols, index=X_test.index) # インデックスを合わせておくと元のデータと紐付けやすい
    print("\nPredicted probabilities for the first few test samples:")
    print(df_res.head())

    # 特徴量の重要度を表示 (オプション)
    if hasattr(final_model, 'feature_importances_'):
        feature_importances = pd.Series(final_model.feature_importances_, index=X_train.columns)
        print("\nTop 10 Feature Importances:")
        print(feature_importances.sort_values(ascending=False).head(10))

if __name__ == '__main__':
    main()
