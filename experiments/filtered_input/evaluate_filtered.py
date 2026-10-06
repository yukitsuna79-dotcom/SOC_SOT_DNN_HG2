from pathlib import Path
import csv

import numpy as np
import torch

from model_filtered import FilteredInputDFNN
from data_loader_filtered import load_npz_file


DEVICE = torch.device("cpu")
SEEDS = [1, 2, 3]

PROJECT_DIR = Path(__file__).resolve().parents[2]

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

EVALUATION_DIR = (
    RESULTS_DIR
    / "filtered_input_evaluation"
)

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_trained_model(seed):

    result_dir = (
        RESULTS_DIR
        / f"filtered_input_seed_{seed}"
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
            f"Best Modelがありません: {model_file}"
        )

    if not standardizer_file.exists():
        raise FileNotFoundError(
            f"Standardizerがありません: {standardizer_file}"
        )

    model = FilteredInputDFNN().to(
        DEVICE
    )

    state_dict = torch.load(
        model_file,
        map_location=DEVICE,
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    standardizer = np.load(
        standardizer_file
    )

    return (
        model,
        standardizer["x_mean"],
        standardizer["x_std"],
        standardizer["y_mean"],
        standardizer["y_std"],
    )


def predict(
    model,
    X,
    x_mean,
    x_std,
    y_mean,
    y_std,
):

    X_std = (
        X - x_mean
    ) / x_std

    X_tensor = torch.tensor(
        X_std,
        dtype=torch.float32,
    ).to(DEVICE)

    with torch.no_grad():

        prediction_std = (
            model(X_tensor)
            .cpu()
            .numpy()
        )

    return (
        prediction_std
        * y_std
        + y_mean
    )


def calculate_metrics(
    prediction,
    target,
):

    soc_error = (
        prediction[:, 0]
        - target[:, 0]
    )

    sot_error = (
        prediction[:, 1]
        - target[:, 1]
    )

    soc_rmse = (
        np.sqrt(
            np.mean(
                soc_error ** 2
            )
        )
        * 100.0
    )

    soc_mae = (
        np.mean(
            np.abs(
                soc_error
            )
        )
        * 100.0
    )

    sot_rmse = np.sqrt(
        np.mean(
            sot_error ** 2
        )
    )

    sot_mae = np.mean(
        np.abs(
            sot_error
        )
    )

    return (
        soc_rmse,
        soc_mae,
        sot_rmse,
        sot_mae,
        soc_error,
        sot_error,
    )


def get_cycle_name(file_path):

    return (
        file_path
        .stem
        .split("_segment")[0]
    )


def get_temperature_name(file_path):

    return file_path.parent.name


test_files = sorted(
    TEST_DIR.rglob("*.npz")
)

if len(test_files) == 0:
    raise FileNotFoundError(
        f"Test NPZがありません: {TEST_DIR}"
    )


all_file_results = []
all_seed_results = []


print("=" * 70)
print("Filtered Input DFNN Test評価")
print("=" * 70)
print("Test NPZ files :", len(test_files))


for seed in SEEDS:

    print()
    print("=" * 70)
    print(f"Seed {seed}")
    print("=" * 70)

    (
        model,
        x_mean,
        x_std,
        y_mean,
        y_std,
    ) = load_trained_model(
        seed
    )

    seed_soc_errors = []
    seed_sot_errors = []

    for file_path in test_files:

        temperature = (
            get_temperature_name(
                file_path
            )
        )

        cycle = get_cycle_name(
            file_path
        )

        X_test, Y_test = (
            load_npz_file(
                file_path
            )
        )

        prediction = predict(
            model,
            X_test,
            x_mean,
            x_std,
            y_mean,
            y_std,
        )

        (
            soc_rmse,
            soc_mae,
            sot_rmse,
            sot_mae,
            soc_error,
            sot_error,
        ) = calculate_metrics(
            prediction,
            Y_test,
        )

        seed_soc_errors.append(
            soc_error
        )

        seed_sot_errors.append(
            sot_error
        )

        all_file_results.append(
            [
                seed,
                temperature,
                cycle,
                file_path.name,
                len(Y_test),
                soc_rmse,
                soc_mae,
                sot_rmse,
                sot_mae,
            ]
        )

        print(
            f"{temperature:>7s} | "
            f"{cycle:<5s} | "
            f"SOC RMSE {soc_rmse:7.3f}% | "
            f"SOC MAE {soc_mae:7.3f}% | "
            f"SOT RMSE {sot_rmse:7.3f} degC | "
            f"SOT MAE {sot_mae:7.3f} degC"
        )

    seed_soc_error = np.concatenate(
        seed_soc_errors
    )

    seed_sot_error = np.concatenate(
        seed_sot_errors
    )

    seed_soc_rmse = (
        np.sqrt(
            np.mean(
                seed_soc_error ** 2
            )
        )
        * 100.0
    )

    seed_soc_mae = (
        np.mean(
            np.abs(
                seed_soc_error
            )
        )
        * 100.0
    )

    seed_sot_rmse = np.sqrt(
        np.mean(
            seed_sot_error ** 2
        )
    )

    seed_sot_mae = np.mean(
        np.abs(
            seed_sot_error
        )
    )

    all_seed_results.append(
        [
            seed,
            len(seed_soc_error),
            seed_soc_rmse,
            seed_soc_mae,
            seed_sot_rmse,
            seed_sot_mae,
        ]
    )

    print()
    print(f"=== Seed {seed} 全Test ===")
    print(
        f"SOC RMSE : {seed_soc_rmse:.3f}%"
    )
    print(
        f"SOC MAE  : {seed_soc_mae:.3f}%"
    )
    print(
        f"SOT RMSE : {seed_sot_rmse:.3f} degC"
    )
    print(
        f"SOT MAE  : {seed_sot_mae:.3f} degC"
    )


# ============================================================
# CSV
# ============================================================

file_csv = (
    EVALUATION_DIR
    / "test_metrics_by_file.csv"
)

with open(
    file_csv,
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "Seed",
            "Temperature",
            "Cycle",
            "File",
            "Samples",
            "SOC_RMSE_percent",
            "SOC_MAE_percent",
            "SOT_RMSE_degC",
            "SOT_MAE_degC",
        ]
    )

    writer.writerows(
        all_file_results
    )


seed_csv = (
    EVALUATION_DIR
    / "test_metrics_by_seed.csv"
)

with open(
    seed_csv,
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "Seed",
            "Samples",
            "SOC_RMSE_percent",
            "SOC_MAE_percent",
            "SOT_RMSE_degC",
            "SOT_MAE_degC",
        ]
    )

    writer.writerows(
        all_seed_results
    )


seed_array = np.array(
    [
        row[2:]
        for row in all_seed_results
    ],
    dtype=float,
)

mean_metrics = np.mean(
    seed_array,
    axis=0,
)

std_metrics = np.std(
    seed_array,
    axis=0,
    ddof=1,
)


summary_csv = (
    EVALUATION_DIR
    / "test_metrics_seed_summary.csv"
)

metric_names = [
    "SOC_RMSE_percent",
    "SOC_MAE_percent",
    "SOT_RMSE_degC",
    "SOT_MAE_degC",
]

with open(
    summary_csv,
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "Metric",
            "Mean",
            "SD",
        ]
    )

    for name, mean, std in zip(
        metric_names,
        mean_metrics,
        std_metrics,
    ):

        writer.writerow(
            [
                name,
                mean,
                std,
            ]
        )


print()
print("=" * 70)
print("Filtered Input Seed 1-3 平均 ± SD")
print("=" * 70)

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
print("保存先 :", EVALUATION_DIR)