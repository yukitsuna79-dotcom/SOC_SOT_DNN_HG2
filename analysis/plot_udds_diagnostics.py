from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from model import BaselineDFNN


# ============================================================
# 基本設定
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data" / "processed" / "test"
RESULTS_DIR = PROJECT_DIR / "results"

OUTPUT_DIR = RESULTS_DIR / "baseline_udds_diagnostics"

SEEDS = [1, 2, 3]

TEMPERATURES = [
    "-10degC",
    "0degC",
    "10degC",
    "25degC",
]

CYCLE_FILE = "UDDS_segment0_1Hz.npz"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# PowerPointや卒論に貼りやすい解像度
SAVE_DPI = 300


# ============================================================
# モデルと標準化情報の読み込み
# ============================================================

def load_model_and_standardizer(seed):

    result_dir = RESULTS_DIR / f"baseline_seed_{seed}"

    model_path = result_dir / "best_model.pth"
    standardizer_path = result_dir / "standardizer.npz"

    if not model_path.exists():
        raise FileNotFoundError(
            f"モデルが見つかりません: {model_path}"
        )

    if not standardizer_path.exists():
        raise FileNotFoundError(
            f"standardizer.npz が見つかりません: "
            f"{standardizer_path}"
        )

    model = BaselineDFNN().to(DEVICE)

    state_dict = torch.load(
        model_path,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)
    model.eval()

    standardizer = np.load(standardizer_path)

    x_mean = standardizer["x_mean"]
    x_std = standardizer["x_std"]

    y_mean = standardizer["y_mean"]
    y_std = standardizer["y_std"]

    return model, x_mean, x_std, y_mean, y_std


# ============================================================
# 推論
# ============================================================

def predict_udds(
    model,
    file_path,
    x_mean,
    x_std,
    y_mean,
    y_std
):

    data = np.load(file_path)

    time = data["Time"]
    voltage = data["Voltage"]
    current = data["Current"]

    soc_true = data["SOC"]
    sot_true = data["SOT"]

    # Baseline入力
    X = np.column_stack(
        (voltage, current)
    )

    # 学習データから求めた平均・標準偏差を使用
    X_std = (X - x_mean) / x_std

    X_tensor = torch.tensor(
        X_std,
        dtype=torch.float32,
        device=DEVICE
    )

    with torch.no_grad():
        prediction_std = model(X_tensor)

    prediction_std = (
        prediction_std
        .detach()
        .cpu()
        .numpy()
    )

    # 元の物理量へ戻す
    prediction = (
        prediction_std * y_std + y_mean
    )

    soc_pred = prediction[:, 0]
    sot_pred = prediction[:, 1]

    # SOCは%表示
    soc_true_percent = soc_true * 100.0
    soc_pred_percent = soc_pred * 100.0

    # 符号付き誤差
    # Error = Prediction - Target
    soc_error = (
        soc_pred_percent - soc_true_percent
    )

    sot_error = (
        sot_pred - sot_true
    )

    return {
        "time": time,
        "voltage": voltage,
        "current": current,

        "soc_true": soc_true_percent,
        "soc_pred": soc_pred_percent,
        "soc_error": soc_error,

        "sot_true": sot_true,
        "sot_pred": sot_pred,
        "sot_error": sot_error,
    }


# ============================================================
# ① Voltage + Current
# ============================================================

def save_input_figure(
    result,
    output_path,
    temperature,
    seed
):

    time = result["time"]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=True
    )

    # Voltage
    axes[0].plot(
        time,
        result["voltage"],
        linewidth=1.0
    )

    axes[0].set_ylabel("Voltage [V]")
    axes[0].grid(True, alpha=0.3)

    # Current
    axes[1].plot(
        time,
        result["current"],
        linewidth=1.0
    )

    axes[1].axhline(
        0,
        linewidth=0.8,
        linestyle="--"
    )

    axes[1].set_ylabel("Current [A]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"UDDS Input Waveforms "
        f"({temperature}, Seed {seed})"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=SAVE_DPI,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# ② SOC推定 + SOC誤差
# ============================================================

def save_soc_figure(
    result,
    output_path,
    temperature,
    seed
):

    time = result["time"]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=True
    )

    # SOC prediction
    axes[0].plot(
        time,
        result["soc_true"],
        label="True SOC",
        linewidth=1.5
    )

    axes[0].plot(
        time,
        result["soc_pred"],
        label="Predicted SOC",
        linewidth=1.2
    )

    axes[0].set_ylabel("SOC [%]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # SOC error
    axes[1].plot(
        time,
        result["soc_error"],
        linewidth=1.0
    )

    axes[1].axhline(
        0,
        linewidth=0.8,
        linestyle="--"
    )

    axes[1].set_ylabel("SOC Error [%]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"UDDS SOC Estimation "
        f"({temperature}, Seed {seed})"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=SAVE_DPI,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# ③ SOT推定 + SOT誤差
# ============================================================

def save_sot_figure(
    result,
    output_path,
    temperature,
    seed
):

    time = result["time"]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=True
    )

    # SOT prediction
    axes[0].plot(
        time,
        result["sot_true"],
        label="True SOT",
        linewidth=1.5
    )

    axes[0].plot(
        time,
        result["sot_pred"],
        label="Predicted SOT",
        linewidth=1.2
    )

    axes[0].set_ylabel("SOT [degC]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # SOT error
    axes[1].plot(
        time,
        result["sot_error"],
        linewidth=1.0
    )

    axes[1].axhline(
        0,
        linewidth=0.8,
        linestyle="--"
    )

    axes[1].set_ylabel("SOT Error [degC]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"UDDS SOT Estimation "
        f"({temperature}, Seed {seed})"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=SAVE_DPI,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# メイン処理
# ============================================================

def main():

    print("=" * 60)
    print("UDDS Baseline Diagnostic Waveforms")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for seed in SEEDS:

        print()
        print(f"Seed {seed} を処理します")

        model, x_mean, x_std, y_mean, y_std = (
            load_model_and_standardizer(seed)
        )

        seed_output_dir = (
            OUTPUT_DIR / f"seed_{seed}"
        )

        seed_output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for temperature in TEMPERATURES:

            file_path = (
                DATA_DIR
                / temperature
                / CYCLE_FILE
            )

            if not file_path.exists():
                print(
                    f"  [SKIP] "
                    f"{file_path} がありません"
                )
                continue

            print(
                f"  {temperature} / UDDS"
            )

            result = predict_udds(
                model,
                file_path,
                x_mean,
                x_std,
                y_mean,
                y_std
            )

            # 温度ごとの保存フォルダ
            temperature_output_dir = (
                seed_output_dir / temperature
            )

            temperature_output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            # ① Voltage + Current
            input_path = (
                temperature_output_dir
                / "UDDS_Input.png"
            )

            save_input_figure(
                result,
                input_path,
                temperature,
                seed
            )

            # ② SOC
            soc_path = (
                temperature_output_dir
                / "UDDS_SOC.png"
            )

            save_soc_figure(
                result,
                soc_path,
                temperature,
                seed
            )

            # ③ SOT
            sot_path = (
                temperature_output_dir
                / "UDDS_SOT.png"
            )

            save_sot_figure(
                result,
                sot_path,
                temperature,
                seed
            )

            print(
                f"    保存完了:"
                f" Input / SOC / SOT"
            )

    print()
    print("=" * 60)
    print("UDDS診断波形の作成が完了しました")
    print("=" * 60)

    print(
        f"保存先: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()