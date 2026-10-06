from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from model import BaselineDFNN


# ============================================================
# 1. 設定
# ============================================================

SEEDS = [1, 2, 3]
TEMPERATURES = ["10degC", "-10degC"]
CYCLE_FILE = "HWFET_segment0_1Hz.npz"

DEVICE = torch.device("cpu")

PROJECT_DIR = Path(__file__).resolve().parent

TEST_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "test"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "baseline_diagnostics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. モデル・標準化係数読み込み
# ============================================================

def load_model_and_standardizer(seed):

    result_dir = (
        RESULTS_DIR
        / f"baseline_seed_{seed}"
    )

    model_file = (
        result_dir
        / "best_model.pth"
    )

    standardizer_file = (
        result_dir
        / "standardizer.npz"
    )

    if not model_file.exists():
        raise FileNotFoundError(
            f"モデルが見つかりません: {model_file}"
        )

    if not standardizer_file.exists():
        raise FileNotFoundError(
            f"標準化係数が見つかりません: {standardizer_file}"
        )

    model = BaselineDFNN().to(DEVICE)

    state_dict = torch.load(
        model_file,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)
    model.eval()

    standardizer = np.load(
        standardizer_file
    )

    x_mean = standardizer["x_mean"]
    x_std = standardizer["x_std"]
    y_mean = standardizer["y_mean"]
    y_std = standardizer["y_std"]

    return (
        model,
        x_mean,
        x_std,
        y_mean,
        y_std
    )


# ============================================================
# 3. Seed 1～3
# ============================================================

print()
print("=" * 70)
print("SOT診断波形作成")
print("=" * 70)

for seed in SEEDS:

    (
        model,
        x_mean,
        x_std,
        y_mean,
        y_std
    ) = load_model_and_standardizer(seed)


    # ========================================================
    # 4. 10℃ / -10℃
    # ========================================================

    for temperature in TEMPERATURES:

        file_path = (
            TEST_DIR
            / temperature
            / CYCLE_FILE
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Testファイルが見つかりません: {file_path}"
            )

        data = np.load(file_path)

        time = data["Time"]
        voltage = data["Voltage"]
        current = data["Current"]

        sot_true = data["SOT"]


        # ----------------------------------------------------
        # Baseline入力
        # ----------------------------------------------------

        X = np.column_stack(
            (
                voltage,
                current
            )
        )


        # ----------------------------------------------------
        # 学習データの統計量で標準化
        # ----------------------------------------------------

        X_standardized = (
            (X - x_mean)
            / x_std
        )

        X_tensor = torch.tensor(
            X_standardized,
            dtype=torch.float32
        ).to(DEVICE)


        # ----------------------------------------------------
        # 推論
        # ----------------------------------------------------

        with torch.no_grad():

            prediction_standardized = (
                model(X_tensor)
                .cpu()
                .numpy()
            )


        # ----------------------------------------------------
        # 逆標準化
        # ----------------------------------------------------

        prediction = (
            prediction_standardized
            * y_std
            + y_mean
        )

        sot_pred = prediction[:, 1]


        # ----------------------------------------------------
        # SOT誤差
        #
        # Prediction - Target
        # ----------------------------------------------------

        sot_error = (
            sot_pred
            - sot_true
        )


        # ====================================================
        # 5. 診断グラフ
        # ====================================================

        fig, axes = plt.subplots(
            4,
            1,
            figsize=(14, 12),
            sharex=True
        )


        # ----------------------------------------------------
        # Voltage
        # ----------------------------------------------------

        axes[0].plot(
            time,
            voltage,
            linewidth=1.0
        )

        axes[0].set_ylabel(
            "Voltage [V]"
        )

        axes[0].grid(True)


        # ----------------------------------------------------
        # Current
        # ----------------------------------------------------

        axes[1].plot(
            time,
            current,
            linewidth=1.0
        )

        axes[1].axhline(
            y=0,
            linewidth=0.8
        )

        axes[1].set_ylabel(
            "Current [A]"
        )

        axes[1].grid(True)


        # ----------------------------------------------------
        # SOT
        # ----------------------------------------------------

        axes[2].plot(
            time,
            sot_true,
            label="True SOT",
            linewidth=1.5
        )

        axes[2].plot(
            time,
            sot_pred,
            label="Predicted SOT",
            linewidth=1.0
        )

        axes[2].set_ylabel(
            "SOT [degC]"
        )

        axes[2].legend()

        axes[2].grid(True)


        # ----------------------------------------------------
        # SOT Error
        # ----------------------------------------------------

        axes[3].plot(
            time,
            sot_error,
            linewidth=1.0
        )

        axes[3].axhline(
            y=0,
            linewidth=0.8
        )

        axes[3].set_ylabel(
            "SOT Error [degC]"
        )

        axes[3].set_xlabel(
            "Time [s]"
        )

        axes[3].grid(True)


        # ----------------------------------------------------
        # 全体タイトル
        # ----------------------------------------------------

        fig.suptitle(
            f"SOT Diagnostic "
            f"(Seed {seed}) - "
            f"{temperature} - HWFET",
            fontsize=16
        )

        plt.tight_layout(
            rect=[0, 0, 1, 0.97]
        )


        # ----------------------------------------------------
        # 保存
        # ----------------------------------------------------

        seed_output_dir = (
            OUTPUT_DIR
            / f"seed_{seed}"
        )

        seed_output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = (
            seed_output_dir
            / f"{temperature}_HWFET_SOT_diagnostic.png"
        )

        plt.savefig(
            output_file,
            dpi=200
        )

        plt.close()


        print(
            f"Seed {seed} | "
            f"{temperature} | "
            f"保存完了"
        )


print()
print("=" * 70)
print("SOT診断波形の作成が完了しました")
print("=" * 70)
print(f"保存先: {OUTPUT_DIR}")
print()