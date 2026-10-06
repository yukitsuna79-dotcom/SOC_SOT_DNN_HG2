from pathlib import Path
import numpy as np
from scipy.io import loadmat


# ==========================================
# HG2 Mixed走行データ比較
# ==========================================

PROJECT_ROOT = Path(r"E:\SOC_SOT_DNN_HG2")

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "LG_HG2_Original"
)

TEMPERATURES = {
    "-10": "n10degC",
    "0": "0degC",
    "10": "10degC",
    "25": "25degC",
}

MIXED_CYCLES = [
    "Mixed1",
    "Mixed2",
    "Mixed4",
    "Mixed5",
    "Mixed6",
    "Mixed7",
    "Mixed8",
]

print("HG2 Mixed走行データ比較")
print("=" * 60)

print("対象温度:")
for temp, folder in TEMPERATURES.items():
    print(f"  {temp} degC -> {folder}")

print("\n比較対象:")
for cycle in MIXED_CYCLES:
    print(f"  {cycle}")

# ==========================================
# STEP 2 : 対象MATファイルの存在確認
# ==========================================

print("\nMATファイル存在確認")
print("=" * 60)

found_count = 0
missing_count = 0

for temp, folder_name in TEMPERATURES.items():

    temp_dir = RAW_DIR / folder_name

    for cycle in MIXED_CYCLES:

        matches = list(temp_dir.glob(f"*{cycle}*.mat"))

        if len(matches) == 1:
            print(
                f"[OK] {temp:>3} degC | "
                f"{cycle:<6} | {matches[0].name}"
            )
            found_count += 1

        elif len(matches) == 0:
            print(
                f"[MISSING] {temp:>3} degC | "
                f"{cycle}"
            )
            missing_count += 1

        else:
            print(
                f"[MULTIPLE] {temp:>3} degC | "
                f"{cycle} | {len(matches)} files"
            )

print("=" * 60)
print(f"見つかったファイル数 : {found_count}")
print(f"見つからないファイル数 : {missing_count}")

# ==========================================
# STEP 4 : 各Mixedデータの特性を取得
# ==========================================

results = []

print("\nMixedデータ特性")
print("=" * 120)

print(
    f"{'Temp':>6} "
    f"{'Cycle':<7} "
    f"{'Time[s]':>10} "
    f"{'V min':>9} "
    f"{'V max':>9} "
    f"{'I min':>9} "
    f"{'I max':>9} "
    f"{'SOC min':>9} "
    f"{'SOC max':>9} "
    f"{'SOT min':>9} "
    f"{'SOT max':>9} "
    f"{'ΔSOT':>9}"
)

print("-" * 120)

for temp, folder_name in TEMPERATURES.items():

    temp_dir = RAW_DIR / folder_name

    for cycle in MIXED_CYCLES:

        mat_path = list(temp_dir.glob(f"*{cycle}*.mat"))[0]

        mat = loadmat(mat_path)

        # HG2 MATファイル内のデータを取得
        data = mat["meas"]

        time = np.asarray(data["Time"][0, 0]).squeeze()
        voltage = np.asarray(data["Voltage"][0, 0]).squeeze()
        current = np.asarray(data["Current"][0, 0]).squeeze()
        ah = np.asarray(data["Ah"][0, 0]).squeeze()
        sot = np.asarray(data["Battery_Temp_degC"][0, 0]).squeeze()

        # 有限値だけを比較に使用
        valid = (
            np.isfinite(time)
            & np.isfinite(voltage)
            & np.isfinite(current)
            & np.isfinite(ah)
            & np.isfinite(sot)
        )

        time = time[valid]
        voltage = voltage[valid]
        current = current[valid]
        ah = ah[valid]
        sot = sot[valid]

        # これまでと同じSOC定義
        soc = 1.0 + ah / 3.0

        duration = np.max(time) - np.min(time)
        sot_change = np.max(sot) - np.min(sot)

        results.append({
            "temp": float(temp),
            "cycle": cycle,
            "duration": duration,
            "voltage_range": np.max(voltage) - np.min(voltage),
            "current_range": np.max(current) - np.min(current),
            "soc_range": np.max(soc) - np.min(soc),
            "sot_range": sot_change,
        })

        print(
            f"{temp:>6} "
            f"{cycle:<7} "
            f"{duration:10.1f} "
            f"{np.min(voltage):9.4f} "
            f"{np.max(voltage):9.4f} "
            f"{np.min(current):9.3f} "
            f"{np.max(current):9.3f} "
            f"{np.min(soc):9.4f} "
            f"{np.max(soc):9.4f} "
            f"{np.min(sot):9.3f} "
            f"{np.max(sot):9.3f} "
            f"{sot_change:9.3f}"
        )

# ==========================================
# STEP 5 : Mixedごとの4温度集計
# ==========================================

print("\n")
print("Mixedごとの4温度集計")
print("=" * 100)

print(
    f"{'Cycle':<7} "
    f"{'平均Time[s]':>12} "
    f"{'平均ΔV[V]':>12} "
    f"{'平均ΔI[A]':>12} "
    f"{'平均ΔSOC':>12} "
    f"{'平均ΔSOT[℃]':>14} "
    f"{'ΔSOT SD':>10}"
)

print("-" * 100)

for cycle in MIXED_CYCLES:

    cycle_results = [
        r for r in results
        if r["cycle"] == cycle
    ]

    durations = np.array(
        [r["duration"] for r in cycle_results]
    )

    voltage_ranges = np.array(
        [r["voltage_range"] for r in cycle_results]
    )

    current_ranges = np.array(
        [r["current_range"] for r in cycle_results]
    )

    soc_ranges = np.array(
        [r["soc_range"] for r in cycle_results]
    )

    sot_ranges = np.array(
        [r["sot_range"] for r in cycle_results]
    )

    print(
        f"{cycle:<7} "
        f"{np.mean(durations):12.1f} "
        f"{np.mean(voltage_ranges):12.4f} "
        f"{np.mean(current_ranges):12.3f} "
        f"{np.mean(soc_ranges):12.4f} "
        f"{np.mean(sot_ranges):14.3f} "
        f"{np.std(sot_ranges, ddof=1):10.3f}"
    )