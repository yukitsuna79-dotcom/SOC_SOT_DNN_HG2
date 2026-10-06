from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. パス設定
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_DIR
    / "results"
    / "baseline_evaluation"
    / "test_metrics_by_file.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "baseline_evaluation"
)

OUTPUT_BY_SEED = (
    OUTPUT_DIR
    / "test_metrics_by_temperature_seed.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "test_metrics_temperature_seed_summary.csv"
)


# ============================================================
# 2. CSV読み込み
# ============================================================

df = pd.read_csv(INPUT_FILE)

print()
print("=" * 75)
print("Baseline DFNN 温度別性能集計")
print("=" * 75)
print(f"入力ファイル : {INPUT_FILE}")
print(f"行数         : {len(df)}")
print()


# ============================================================
# 3. 温度 × Seed ごとに集計
# ============================================================

results = []

for (seed, temperature), group in df.groupby(
    ["Seed", "Temperature"],
    sort=False
):

    n = group["Samples"].to_numpy(dtype=float)

    total_samples = int(
        np.sum(n)
    )


    # --------------------------------------------------------
    # SOC RMSE
    #
    # sqrt(
    #   sum(N_j * RMSE_j^2)
    #   / sum(N_j)
    # )
    # --------------------------------------------------------

    soc_rmse = np.sqrt(
        np.sum(
            n
            * group[
                "SOC_RMSE_percent"
            ].to_numpy(dtype=float) ** 2
        )
        / np.sum(n)
    )


    # --------------------------------------------------------
    # SOC MAE
    # --------------------------------------------------------

    soc_mae = (
        np.sum(
            n
            * group[
                "SOC_MAE_percent"
            ].to_numpy(dtype=float)
        )
        / np.sum(n)
    )


    # --------------------------------------------------------
    # SOT RMSE
    # --------------------------------------------------------

    sot_rmse = np.sqrt(
        np.sum(
            n
            * group[
                "SOT_RMSE_degC"
            ].to_numpy(dtype=float) ** 2
        )
        / np.sum(n)
    )


    # --------------------------------------------------------
    # SOT MAE
    # --------------------------------------------------------

    sot_mae = (
        np.sum(
            n
            * group[
                "SOT_MAE_degC"
            ].to_numpy(dtype=float)
        )
        / np.sum(n)
    )


    results.append(
        {
            "Seed": seed,
            "Temperature": temperature,
            "Samples": total_samples,
            "SOC_RMSE_percent": soc_rmse,
            "SOC_MAE_percent": soc_mae,
            "SOT_RMSE_degC": sot_rmse,
            "SOT_MAE_degC": sot_mae,
        }
    )


temperature_seed_df = pd.DataFrame(
    results
)


# ============================================================
# 4. 温度順に並べる
# ============================================================

temperature_order = {
    "-10degC": 0,
    "0degC": 1,
    "10degC": 2,
    "25degC": 3,
}

temperature_seed_df["Temperature_order"] = (
    temperature_seed_df["Temperature"]
    .map(temperature_order)
)

temperature_seed_df = (
    temperature_seed_df
    .sort_values(
        [
            "Temperature_order",
            "Seed"
        ]
    )
    .drop(
        columns="Temperature_order"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 5. Seed別結果を表示
# ============================================================

print("=" * 75)
print("【温度別 × Seed別】")
print("=" * 75)

for temperature in [
    "-10degC",
    "0degC",
    "10degC",
    "25degC"
]:

    temp_df = temperature_seed_df[
        temperature_seed_df["Temperature"]
        == temperature
    ]

    print()
    print(f"--- {temperature} ---")

    for _, row in temp_df.iterrows():

        print(
            f"Seed {int(row['Seed'])}: "
            f"SOC RMSE "
            f"{row['SOC_RMSE_percent']:.3f}% | "
            f"SOC MAE "
            f"{row['SOC_MAE_percent']:.3f}% | "
            f"SOT RMSE "
            f"{row['SOT_RMSE_degC']:.3f} degC | "
            f"SOT MAE "
            f"{row['SOT_MAE_degC']:.3f} degC"
        )


# ============================================================
# 6. 各温度について3 Seed平均 ± 標準偏差
#
# ddof=1:
# 標本標準偏差
# ============================================================

summary_rows = []

metric_columns = [
    "SOC_RMSE_percent",
    "SOC_MAE_percent",
    "SOT_RMSE_degC",
    "SOT_MAE_degC",
]

for temperature in [
    "-10degC",
    "0degC",
    "10degC",
    "25degC"
]:

    temp_df = temperature_seed_df[
        temperature_seed_df["Temperature"]
        == temperature
    ]

    row = {
        "Temperature": temperature
    }

    for metric in metric_columns:

        values = (
            temp_df[metric]
            .to_numpy(dtype=float)
        )

        row[f"{metric}_mean"] = (
            np.mean(values)
        )

        row[f"{metric}_SD"] = (
            np.std(
                values,
                ddof=1
            )
        )

    summary_rows.append(row)


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# 7. 3 Seed平均 ± SD 表示
# ============================================================

print()
print("=" * 75)
print("【温度別 3 Seed平均 ± SD】")
print("=" * 75)

for _, row in summary_df.iterrows():

    print()
    print(
        f"{row['Temperature']}"
    )

    print(
        f"  SOC RMSE : "
        f"{row['SOC_RMSE_percent_mean']:.3f}"
        f" ± "
        f"{row['SOC_RMSE_percent_SD']:.3f} %"
    )

    print(
        f"  SOC MAE  : "
        f"{row['SOC_MAE_percent_mean']:.3f}"
        f" ± "
        f"{row['SOC_MAE_percent_SD']:.3f} %"
    )

    print(
        f"  SOT RMSE : "
        f"{row['SOT_RMSE_degC_mean']:.3f}"
        f" ± "
        f"{row['SOT_RMSE_degC_SD']:.3f} degC"
    )

    print(
        f"  SOT MAE  : "
        f"{row['SOT_MAE_degC_mean']:.3f}"
        f" ± "
        f"{row['SOT_MAE_degC_SD']:.3f} degC"
    )


# ============================================================
# 8. CSV保存
# ============================================================

temperature_seed_df.to_csv(
    OUTPUT_BY_SEED,
    index=False
)

summary_df.to_csv(
    OUTPUT_SUMMARY,
    index=False
)


print()
print("=" * 75)
print("温度別集計が完了しました")
print("=" * 75)
print(f"Seed別結果 : {OUTPUT_BY_SEED}")
print(f"平均±SD    : {OUTPUT_SUMMARY}")
print()