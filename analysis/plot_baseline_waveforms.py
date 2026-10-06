from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from model import BaselineDFNN


# ============================================================
# 1. 設定
# ============================================================

SEEDS = [1, 2, 3]
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

WAVEFORM_DIR = (
    RESULTS_DIR
    / "baseline_waveforms"
)

WAVEFORM_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. 学習済みモデルと標準化係数を読み込む関数
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

    # モデル
    model = BaselineDFNN().to(DEVICE)

    state_dict = torch.load(
        model_file,
        map_location=DEVICE
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    # 標準化係数
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
# 3. Testファイル一覧
# ============================================================

test_files = sorted(
    TEST_DIR.rglob("*.npz")
)

print()
print("=" * 70)
print("Baseline DFNN Seed 1～3 波形作成")
print("=" * 70)
print(f"Device         : {DEVICE}")
print(f"Seeds          : {SEEDS}")
print(f"Test NPZ files : {len(test_files)}")
print(f"保存先         : {WAVEFORM_DIR}")
print("=" * 70)


# ============================================================
# 4. Seed 1～3
# ============================================================

for seed in SEEDS:

    print()
    print("=" * 70)
    print(f"Seed {seed} 波形作成開始")
    print("=" * 70)

    (
        model,
        x_mean,
        x_std,
        y_mean,
        y_std
    ) = load_model_and_standardizer(seed)


    # Seedごとの保存フォルダ
    seed_dir = (
        WAVEFORM_DIR
        / f"seed_{seed}"
    )

    seed_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # 5. 19個のTest NPZ
    # ========================================================

    for file_path in test_files:

        data = np.load(
            file_path
        )

        # ----------------------------------------------------
        # データ読み込み
        # ----------------------------------------------------

        time = data["Time"]

        voltage = data["Voltage"]
        current = data["Current"]

        soc_true = data["SOC"]
        sot_true = data["SOT"]


        # ----------------------------------------------------
        # Baseline入力
        #
        # [Voltage, Current]
        # ----------------------------------------------------

        X = np.column_stack(
            (
                voltage,
                current
            )
        )


        # ----------------------------------------------------
        # Training統計量を使って標準化
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
                model(
                    X_tensor
                )
                .cpu()
                .numpy()
            )


        # ----------------------------------------------------
        # 物理単位へ逆変換
        # ----------------------------------------------------

        prediction = (
            prediction_standardized
            * y_std
            + y_mean
        )


        soc_pred = (
            prediction[:, 0]
        )

        sot_pred = (
            prediction[:, 1]
        )


        # ----------------------------------------------------
        # SOCを[%]へ変換
        # ----------------------------------------------------

        soc_true_percent = (
            soc_true
            * 100.0
        )

        soc_pred_percent = (
            soc_pred
            * 100.0
        )


        # ----------------------------------------------------
        # 符号付き推定誤差
        #
        # Error = Prediction - Target
        # ----------------------------------------------------

        soc_error = (
            soc_pred_percent
            - soc_true_percent
        )

        sot_error = (
            sot_pred
            - sot_true
        )


        # ----------------------------------------------------
        # 温度・走行サイクル名
        # ----------------------------------------------------

        temperature = (
            file_path.parent.name
        )

        cycle_name = (
            file_path.stem
            .replace(
                "_1Hz",
                ""
            )
        )


        # ----------------------------------------------------
        # 保存先
        # ----------------------------------------------------

        output_dir = (
            seed_dir
            / temperature
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        # ====================================================
        # 6. SOC 真値・推定値
        # ====================================================

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            time,
            soc_true_percent,
            label="True SOC",
            linewidth=1.5
        )

        plt.plot(
            time,
            soc_pred_percent,
            label="Predicted SOC",
            linewidth=1.2
        )

        plt.xlabel(
            "Time [s]"
        )

        plt.ylabel(
            "SOC [%]"
        )

        plt.title(
            f"SOC Estimation "
            f"(Seed {seed}) - "
            f"{temperature} - "
            f"{cycle_name}"
        )

        plt.legend()

        plt.grid(
            True
        )

        plt.tight_layout()

        plt.savefig(
            output_dir
            / f"{cycle_name}_SOC_prediction.png",
            dpi=200
        )

        plt.close()


        # ====================================================
        # 7. SOC 誤差波形
        # ====================================================

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            time,
            soc_error,
            linewidth=1.2
        )

        plt.axhline(
            y=0,
            linewidth=1.0
        )

        plt.xlabel(
            "Time [s]"
        )

        plt.ylabel(
            "SOC Error [%]"
        )

        plt.title(
            f"SOC Error "
            f"(Seed {seed}) - "
            f"{temperature} - "
            f"{cycle_name}"
        )

        plt.grid(
            True
        )

        plt.tight_layout()

        plt.savefig(
            output_dir
            / f"{cycle_name}_SOC_error.png",
            dpi=200
        )

        plt.close()


        # ====================================================
        # 8. SOT 真値・推定値
        # ====================================================

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            time,
            sot_true,
            label="True SOT",
            linewidth=1.5
        )

        plt.plot(
            time,
            sot_pred,
            label="Predicted SOT",
            linewidth=1.2
        )

        plt.xlabel(
            "Time [s]"
        )

        plt.ylabel(
            "SOT [degC]"
        )

        plt.title(
            f"SOT Estimation "
            f"(Seed {seed}) - "
            f"{temperature} - "
            f"{cycle_name}"
        )

        plt.legend()

        plt.grid(
            True
        )

        plt.tight_layout()

        plt.savefig(
            output_dir
            / f"{cycle_name}_SOT_prediction.png",
            dpi=200
        )

        plt.close()


        # ====================================================
        # 9. SOT 誤差波形
        # ====================================================

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            time,
            sot_error,
            linewidth=1.2
        )

        plt.axhline(
            y=0,
            linewidth=1.0
        )

        plt.xlabel(
            "Time [s]"
        )

        plt.ylabel(
            "SOT Error [degC]"
        )

        plt.title(
            f"SOT Error "
            f"(Seed {seed}) - "
            f"{temperature} - "
            f"{cycle_name}"
        )

        plt.grid(
            True
        )

        plt.tight_layout()

        plt.savefig(
            output_dir
            / f"{cycle_name}_SOT_error.png",
            dpi=200
        )

        plt.close()


        print(
            f"Seed {seed} | "
            f"{temperature:>7s} | "
            f"{cycle_name}"
        )


    print()
    print(
        f"Seed {seed} 完了"
    )


# ============================================================
# 10. 完了
# ============================================================

print()
print("=" * 70)
print("Seed 1～3 全波形の作成が完了しました")
print("=" * 70)
print(f"保存先: {WAVEFORM_DIR}")
print()