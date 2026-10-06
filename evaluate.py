from pathlib import Path
import csv

import numpy as np
import torch

from model import BaselineDFNN


# ============================================================
# 1. 評価条件
# ============================================================

DEVICE = torch.device("cpu")
SEEDS = [1, 2, 3]


# ============================================================
# 2. フォルダ
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

TEST_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "test"
)

RESULTS_DIR = PROJECT_DIR / "results"

EVALUATION_DIR = RESULTS_DIR / "baseline_evaluation"
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. 学習済みモデルと標準化係数の読み込み
# ============================================================

def load_trained_model(seed):

    result_dir = RESULTS_DIR / f"baseline_seed_{seed}"

    model_file = result_dir / "best_model.pth"
    standardizer_file = result_dir / "standardizer.npz"

    if not model_file.exists():
        raise FileNotFoundError(
            f"Best Modelが見つかりません: {model_file}"
        )

    if not standardizer_file.exists():
        raise FileNotFoundError(
            f"Standardizerが見つかりません: {standardizer_file}"
        )

    model = BaselineDFNN().to(DEVICE)

    state_dict = torch.load(
        model_file,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)
    model.eval()

    standardizer = np.load(standardizer_file)

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
# 4. Test NPZを1ファイル読み込む
# ============================================================

def load_test_file(file_path):

    data = np.load(file_path)

    voltage = data["Voltage"]
    current = data["Current"]
    soc = data["SOC"]
    sot = data["SOT"]

    X = np.column_stack(
        (voltage, current)
    )

    Y = np.column_stack(
        (soc, sot)
    )

    return X, Y


# ============================================================
# 5. Testデータを推論する
# ============================================================

def predict(
    model,
    X,
    x_mean,
    x_std,
    y_mean,
    y_std
):

    # Trainingデータから求めた係数で標準化
    X_std = (
        (X - x_mean)
        / x_std
    )

    X_tensor = torch.tensor(
        X_std,
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():

        prediction_std = model(
            X_tensor
        ).cpu().numpy()

    # 物理単位へ戻す
    prediction = (
        prediction_std * y_std
        + y_mean
    )

    return prediction


# ============================================================
# 6. 誤差を計算する
# ============================================================

def calculate_errors(
    prediction,
    target
):

    # SOC
    soc_error = (
        prediction[:, 0]
        - target[:, 0]
    )

    # SOT
    sot_error = (
        prediction[:, 1]
        - target[:, 1]
    )

    return soc_error, sot_error


# ============================================================
# 7. RMSE・MAEを計算する
# ============================================================

def calculate_metrics(
    soc_error,
    sot_error
):

    # SOC
    soc_rmse = (
        np.sqrt(
            np.mean(soc_error ** 2)
        )
        * 100.0
    )

    soc_mae = (
        np.mean(
            np.abs(soc_error)
        )
        * 100.0
    )

    # SOT
    sot_rmse = np.sqrt(
        np.mean(sot_error ** 2)
    )

    sot_mae = np.mean(
        np.abs(sot_error)
    )

    return (
        soc_rmse,
        soc_mae,
        sot_rmse,
        sot_mae
    )


# ============================================================
# 8. ファイル名から走行サイクル名を取得
# ============================================================

def get_cycle_name(file_path):

    file_name = file_path.stem

    # 例：
    # UDDS_segment0_1Hz
    # LA92_segment6_1Hz

    cycle_name = file_name.split(
        "_segment"
    )[0]

    return cycle_name


# ============================================================
# 9. 温度フォルダ名を取得
# ============================================================

def get_temperature_name(file_path):

    return file_path.parent.name


# ============================================================
# 10. Testファイル一覧
# ============================================================

test_files = sorted(
    TEST_DIR.rglob("*.npz")
)

if len(test_files) == 0:
    raise FileNotFoundError(
        f"Test NPZが見つかりません: {TEST_DIR}"
    )


print("============================================================")
print("Baseline DFNN 正式Test評価")
print("============================================================")

print("Device :", DEVICE)
print("Seeds  :", SEEDS)
print("Test NPZ files :", len(test_files))

print("============================================================")


# ============================================================
# 11. 全Seed評価
# ============================================================

all_file_results = []
all_group_results = []
all_seed_results = []


for seed in SEEDS:

    print()
    print("============================================================")
    print(f"Seed {seed} 評価開始")
    print("============================================================")

    (
        model,
        x_mean,
        x_std,
        y_mean,
        y_std
    ) = load_trained_model(seed)


    # --------------------------------------------------------
    # Seed全体の誤差
    # --------------------------------------------------------

    seed_soc_errors = []
    seed_sot_errors = []


    # --------------------------------------------------------
    # 温度×走行サイクルごとの誤差
    # --------------------------------------------------------

    grouped_errors = {}


    # --------------------------------------------------------
    # NPZを1つずつ評価
    # --------------------------------------------------------

    for file_path in test_files:

        temperature = get_temperature_name(
            file_path
        )

        cycle = get_cycle_name(
            file_path
        )

        X_test, Y_test = load_test_file(
            file_path
        )

        prediction = predict(
            model,
            X_test,
            x_mean,
            x_std,
            y_mean,
            y_std
        )

        (
            soc_error,
            sot_error
        ) = calculate_errors(
            prediction,
            Y_test
        )

        (
            soc_rmse,
            soc_mae,
            sot_rmse,
            sot_mae
        ) = calculate_metrics(
            soc_error,
            sot_error
        )


        # ----------------------------------------------------
        # NPZ単位の結果
        # ----------------------------------------------------

        all_file_results.append([
            seed,
            temperature,
            cycle,
            file_path.name,
            len(Y_test),
            soc_rmse,
            soc_mae,
            sot_rmse,
            sot_mae
        ])


        # ----------------------------------------------------
        # Seed全体用
        # ----------------------------------------------------

        seed_soc_errors.append(
            soc_error
        )

        seed_sot_errors.append(
            sot_error
        )


        # ----------------------------------------------------
        # 温度×サイクル単位で集約
        #
        # 10degC LA92の複数segmentも
        # ここで同じグループにまとめる
        # ----------------------------------------------------

        key = (
            temperature,
            cycle
        )

        if key not in grouped_errors:

            grouped_errors[key] = {
                "soc": [],
                "sot": []
            }

        grouped_errors[key]["soc"].append(
            soc_error
        )

        grouped_errors[key]["sot"].append(
            sot_error
        )


        print(
            f"{temperature:>7s} | "
            f"{cycle:<5s} | "
            f"{file_path.name:<25s} | "
            f"SOC RMSE {soc_rmse:7.3f}% | "
            f"SOC MAE {soc_mae:7.3f}% | "
            f"SOT RMSE {sot_rmse:7.3f} degC | "
            f"SOT MAE {sot_mae:7.3f} degC"
        )


    # ========================================================
    # 12. 温度×走行サイクル単位の集約
    # ========================================================

    print()
    print(
        f"--- Seed {seed} "
        "温度×走行サイクル集約 ---"
    )

    for key in sorted(
        grouped_errors.keys()
    ):

        temperature, cycle = key

        group_soc_error = np.concatenate(
            grouped_errors[key]["soc"]
        )

        group_sot_error = np.concatenate(
            grouped_errors[key]["sot"]
        )

        (
            soc_rmse,
            soc_mae,
            sot_rmse,
            sot_mae
        ) = calculate_metrics(
            group_soc_error,
            group_sot_error
        )

        all_group_results.append([
            seed,
            temperature,
            cycle,
            len(group_soc_error),
            soc_rmse,
            soc_mae,
            sot_rmse,
            sot_mae
        ])

        print(
            f"{temperature:>7s} | "
            f"{cycle:<5s} | "
            f"N {len(group_soc_error):6d} | "
            f"SOC RMSE {soc_rmse:7.3f}% | "
            f"SOC MAE {soc_mae:7.3f}% | "
            f"SOT RMSE {sot_rmse:7.3f} degC | "
            f"SOT MAE {sot_mae:7.3f} degC"
        )


    # ========================================================
    # 13. Seed全Test総合評価
    # ========================================================

    seed_soc_error = np.concatenate(
        seed_soc_errors
    )

    seed_sot_error = np.concatenate(
        seed_sot_errors
    )

    (
        seed_soc_rmse,
        seed_soc_mae,
        seed_sot_rmse,
        seed_sot_mae
    ) = calculate_metrics(
        seed_soc_error,
        seed_sot_error
    )

    all_seed_results.append([
        seed,
        len(seed_soc_error),
        seed_soc_rmse,
        seed_soc_mae,
        seed_sot_rmse,
        seed_sot_mae
    ])


    print()
    print(
        f"=== Seed {seed} Test総合 ==="
    )

    print(
        f"SOC RMSE : "
        f"{seed_soc_rmse:.3f}%"
    )

    print(
        f"SOC MAE  : "
        f"{seed_soc_mae:.3f}%"
    )

    print(
        f"SOT RMSE : "
        f"{seed_sot_rmse:.3f} degC"
    )

    print(
        f"SOT MAE  : "
        f"{seed_sot_mae:.3f} degC"
    )


# ============================================================
# 14. CSV保存：NPZ単位
# ============================================================

file_result_csv = (
    EVALUATION_DIR
    / "test_metrics_by_file.csv"
)

with open(
    file_result_csv,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Seed",
        "Temperature",
        "Cycle",
        "File",
        "Samples",
        "SOC_RMSE_percent",
        "SOC_MAE_percent",
        "SOT_RMSE_degC",
        "SOT_MAE_degC"
    ])

    writer.writerows(
        all_file_results
    )


# ============================================================
# 15. CSV保存：温度×走行サイクル単位
# ============================================================

group_result_csv = (
    EVALUATION_DIR
    / "test_metrics_by_temperature_cycle.csv"
)

with open(
    group_result_csv,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Seed",
        "Temperature",
        "Cycle",
        "Samples",
        "SOC_RMSE_percent",
        "SOC_MAE_percent",
        "SOT_RMSE_degC",
        "SOT_MAE_degC"
    ])

    writer.writerows(
        all_group_results
    )


# ============================================================
# 16. CSV保存：Seed総合
# ============================================================

seed_result_csv = (
    EVALUATION_DIR
    / "test_metrics_by_seed.csv"
)

with open(
    seed_result_csv,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Seed",
        "Samples",
        "SOC_RMSE_percent",
        "SOC_MAE_percent",
        "SOT_RMSE_degC",
        "SOT_MAE_degC"
    ])

    writer.writerows(
        all_seed_results
    )


# ============================================================
# 17. Seed 1～3 平均・標準偏差
# ============================================================

seed_array = np.array(
    [
        row[2:]
        for row in all_seed_results
    ],
    dtype=float
)

mean_metrics = np.mean(
    seed_array,
    axis=0
)

# 標本標準偏差
std_metrics = np.std(
    seed_array,
    axis=0,
    ddof=1
)


summary_csv = (
    EVALUATION_DIR
    / "test_metrics_seed_summary.csv"
)

with open(
    summary_csv,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Metric",
        "Mean",
        "SD"
    ])

    metric_names = [
        "SOC_RMSE_percent",
        "SOC_MAE_percent",
        "SOT_RMSE_degC",
        "SOT_MAE_degC"
    ]

    for name, mean, std in zip(
        metric_names,
        mean_metrics,
        std_metrics
    ):

        writer.writerow([
            name,
            mean,
            std
        ])


# ============================================================
# 18. 最終表示
# ============================================================

print()
print("============================================================")
print("Baseline DFNN 正式Test評価 完了")
print("============================================================")

print()
print("Seed 1～3 平均 ± SD")

print(
    f"SOC RMSE : "
    f"{mean_metrics[0]:.3f} "
    f"± {std_metrics[0]:.3f}%"
)

print(
    f"SOC MAE  : "
    f"{mean_metrics[1]:.3f} "
    f"± {std_metrics[1]:.3f}%"
)

print(
    f"SOT RMSE : "
    f"{mean_metrics[2]:.3f} "
    f"± {std_metrics[2]:.3f} degC"
)

print(
    f"SOT MAE  : "
    f"{mean_metrics[3]:.3f} "
    f"± {std_metrics[3]:.3f} degC"
)

print()
print("保存先 :")
print(EVALUATION_DIR)

print()
print("NPZ単位 :")
print(file_result_csv)

print()
print("温度×走行サイクル単位 :")
print(group_result_csv)

print()
print("Seed単位 :")
print(seed_result_csv)

print()
print("Seed平均・SD :")
print(summary_csv)