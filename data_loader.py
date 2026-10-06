from pathlib import Path
import numpy as np


# プロジェクト内のprocessedフォルダ
PROCESSED_DIR = Path(__file__).resolve().parent / "data" / "processed"


def load_split(split_name):
    """
    train / validation / test のNPZファイルを読み込む
    """

    split_dir = PROCESSED_DIR / split_name

    if not split_dir.exists():
        raise FileNotFoundError(
            f"フォルダが見つかりません: {split_dir}"
        )

    npz_files = sorted(split_dir.rglob("*.npz"))

    if len(npz_files) == 0:
        raise FileNotFoundError(
            f"NPZファイルが見つかりません: {split_dir}"
        )

    x_list = []
    y_list = []

    for file_path in npz_files:
        data = np.load(file_path)

        voltage = data["Voltage"]
        current = data["Current"]
        soc = data["SOC"]
        sot = data["SOT"]

        # 入力：Voltage, Current
        x = np.column_stack((voltage, current))

        # 教師データ：SOC, SOT
        y = np.column_stack((soc, sot))

        x_list.append(x)
        y_list.append(y)

    X = np.concatenate(x_list, axis=0)
    Y = np.concatenate(y_list, axis=0)

    return X, Y

def fit_standardizer(data):
    """
    Trainingデータから平均値と標準偏差を計算する
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    return mean, std


def standardize(data, mean, std):
    """
    Trainingデータから求めた平均値・標準偏差で標準化する
    """
    return (data - mean) / std

if __name__ == "__main__":

    X_train, Y_train = load_split("train")
    X_val, Y_val = load_split("validation")

    print("=== HG2 データ読み込み確認 ===")

    print("\nTraining")
    print("X :", X_train.shape)
    print("Y :", Y_train.shape)

    print("\nValidation")
    print("X :", X_val.shape)
    print("Y :", Y_val.shape)

    print("\nTraining 入力先頭1行")
    print(X_train[0])

    print("\nTraining 教師データ先頭1行")
    print(Y_train[0])

    # Trainingデータだけから平均値と標準偏差を計算
    x_mean, x_std = fit_standardizer(X_train)
    y_mean, y_std = fit_standardizer(Y_train)

    # TrainingとValidationを同じ係数で標準化
    X_train_std = standardize(X_train, x_mean, x_std)
    Y_train_std = standardize(Y_train, y_mean, y_std)

    X_val_std = standardize(X_val, x_mean, x_std)
    Y_val_std = standardize(Y_val, y_mean, y_std)

    print("\n=== 標準化係数（Trainingデータから計算）===")

    print("入力平均 [Voltage, Current] :", x_mean)
    print("入力標準偏差 [Voltage, Current] :", x_std)

    print("教師平均 [SOC, SOT] :", y_mean)
    print("教師標準偏差 [SOC, SOT] :", y_std)

    print("\n=== 標準化後 Training ===")
    print("X 平均 :", np.mean(X_train_std, axis=0))
    print("X 標準偏差 :", np.std(X_train_std, axis=0))
    print("Y 平均 :", np.mean(Y_train_std, axis=0))
    print("Y 標準偏差 :", np.std(Y_train_std, axis=0))