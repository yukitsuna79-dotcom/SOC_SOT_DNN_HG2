from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt


PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

SAMPLING_FREQUENCY_HZ = 1.0

# Vidalらを参考にしたLPFカットオフ周波数
CUTOFF_05_MHZ_HZ = 0.0005
CUTOFF_5_MHZ_HZ = 0.005


def lowpass_filter(signal, cutoff_hz, fs=SAMPLING_FREQUENCY_HZ):
    """
    1次Butterworthローパスフィルタ。

    各NPZ（各時系列セグメント）を独立に処理する。
    将来情報を使用しない因果フィルタとしてsosfiltを使用する。
    """

    signal = np.asarray(signal, dtype=np.float64)

    sos = butter(
        N=1,
        Wn=cutoff_hz,
        btype="lowpass",
        fs=fs,
        output="sos",
    )

    # 初期値による極端な立ち上がりを避けるため、
    # 信号の最初の値を基準としてフィルタリングする。
    filtered = (
        sosfilt(
            sos,
            signal - signal[0],
        )
        + signal[0]
    )

    return filtered


def create_filtered_inputs(voltage, current):
    """
    6入力を生成する。

    0: Voltage
    1: Current
    2: Voltage 0.5 mHz LPF
    3: Current 0.5 mHz LPF
    4: Voltage 5 mHz LPF
    5: Current 5 mHz LPF
    """

    voltage = np.asarray(voltage, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)

    if len(voltage) != len(current):
        raise ValueError(
            "VoltageとCurrentのサンプル数が一致していません。"
        )

    voltage_05mhz = lowpass_filter(
        voltage,
        CUTOFF_05_MHZ_HZ,
    )

    current_05mhz = lowpass_filter(
        current,
        CUTOFF_05_MHZ_HZ,
    )

    voltage_5mhz = lowpass_filter(
        voltage,
        CUTOFF_5_MHZ_HZ,
    )

    current_5mhz = lowpass_filter(
        current,
        CUTOFF_5_MHZ_HZ,
    )

    X = np.column_stack(
        (
            voltage,
            current,
            voltage_05mhz,
            current_05mhz,
            voltage_5mhz,
            current_5mhz,
        )
    )

    return X


def load_npz_file(file_path):
    """
    1つのNPZを読み込み、6入力とSOC/SOT教師データを生成する。
    """

    data = np.load(file_path)

    voltage = data["Voltage"]
    current = data["Current"]
    soc = data["SOC"]
    sot = data["SOT"]

    X = create_filtered_inputs(
        voltage,
        current,
    )

    Y = np.column_stack(
        (
            soc,
            sot,
        )
    )

    return X, Y


def load_split(split_name):
    """
    train / validation / test のNPZを読み込む。

    フィルタ処理はNPZごとに独立して行い、
    異なる走行データ間でフィルタ状態を引き継がない。
    """

    split_dir = PROCESSED_DIR / split_name

    if not split_dir.exists():
        raise FileNotFoundError(
            f"フォルダが見つかりません: {split_dir}"
        )

    npz_files = sorted(
        split_dir.rglob("*.npz")
    )

    if len(npz_files) == 0:
        raise FileNotFoundError(
            f"NPZファイルが見つかりません: {split_dir}"
        )

    x_list = []
    y_list = []

    for file_path in npz_files:

        X, Y = load_npz_file(
            file_path
        )

        x_list.append(X)
        y_list.append(Y)

    X = np.concatenate(
        x_list,
        axis=0,
    )

    Y = np.concatenate(
        y_list,
        axis=0,
    )

    return X, Y


def fit_standardizer(data):
    mean = np.mean(
        data,
        axis=0,
    )

    std = np.std(
        data,
        axis=0,
    )

    if np.any(std == 0):
        raise ValueError(
            "標準偏差が0の入力または出力があります。"
        )

    return mean, std


def standardize(data, mean, std):
    return (
        data - mean
    ) / std


if __name__ == "__main__":

    X_train, Y_train = load_split(
        "train"
    )

    X_val, Y_val = load_split(
        "validation"
    )

    print("=== Filtered Input データ確認 ===")

    print()
    print("Training")
    print("X :", X_train.shape)
    print("Y :", Y_train.shape)

    print()
    print("Validation")
    print("X :", X_val.shape)
    print("Y :", Y_val.shape)

    print()
    print("入力順序")
    print(
        "[V, I, V_0.5mHz, I_0.5mHz, "
        "V_5mHz, I_5mHz]"
    )

    print()
    print("Training先頭1行")
    print(X_train[0])

    print()
    print("NaN確認")
    print("X NaN :", np.isnan(X_train).any())
    print("Y NaN :", np.isnan(Y_train).any())

    print()
    print("Inf確認")
    print("X Inf :", np.isinf(X_train).any())
    print("Y Inf :", np.isinf(Y_train).any())