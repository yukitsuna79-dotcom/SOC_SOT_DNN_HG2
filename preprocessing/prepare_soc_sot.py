from pathlib import Path
from scipy.io import loadmat
import numpy as np


# ==========================================
# パス設定
# ==========================================
project_dir = Path(__file__).resolve().parent

raw_dir = (
    project_dir
    / "data"
    / "raw"
    / "LG_HG2_Original"
)

processed_dir = (
    project_dir
    / "data"
    / "processed"
)


# ==========================================
# 1ファイルを処理する関数
# ==========================================
def process_file(mat_file):

    data = loadmat(
        mat_file,
        squeeze_me=True,
        struct_as_record=False
    )

    meas = data["meas"]

    time = np.asarray(meas.Time, dtype=float)
    voltage = np.asarray(meas.Voltage, dtype=float)
    current = np.asarray(meas.Current, dtype=float)
    ah = np.asarray(meas.Ah, dtype=float)
    temperature = np.asarray(
        meas.Battery_Temp_degC,
        dtype=float
    )
        # ======================================
    # NaNを含む行を除外
    # ======================================
    valid_mask = (
        np.isfinite(time)
        & np.isfinite(voltage)
        & np.isfinite(current)
        & np.isfinite(ah)
        & np.isfinite(temperature)
    )

    time = time[valid_mask]
    voltage = voltage[valid_mask]
    current = current[valid_mask]
    ah = ah[valid_mask]
    temperature = temperature[valid_mask]
    
        # ======================================
    # 同一時刻の重複データを1つに整理
    # ======================================
    keep = np.concatenate((
        [True],
        np.diff(time) != 0
    ))

    time = time[keep]
    voltage = voltage[keep]
    current = current[keep]
    ah = ah[keep]
    temperature = temperature[keep]
    
        # ======================================
    # 連続区間の境界を検出
    # ======================================
    time_diff = np.diff(time)

    split_indices = np.where(
        (time_diff > 5.0) | (time_diff < 0.0)
    )[0] + 1

    segments = np.split(
        np.arange(len(time)),
        split_indices
    )

    # 空の区間を除外
    segments = [
        segment
        for segment in segments
        if len(segment) > 0
    ]

        # ======================================
    # 各連続区間を1 Hz化し、SOC・SOTを作成
    # ======================================
    processed_segments = []

    nominal_capacity_ah = 3.0

    for segment_id, segment in enumerate(segments):

        segment_time = time[segment]
        segment_voltage = voltage[segment]
        segment_current = current[segment]
        segment_ah = ah[segment]
        segment_temperature = temperature[segment]

        # 元データが存在する範囲内だけ1 Hz化
        start_time = np.ceil(segment_time[0])
        end_time = np.floor(segment_time[-1])

        if end_time < start_time:
            continue

        time_1hz = np.arange(
            start_time,
            end_time + 1.0,
            1.0
        )

        voltage_1hz = np.interp(
            time_1hz,
            segment_time,
            segment_voltage
        )

        current_1hz = np.interp(
            time_1hz,
            segment_time,
            segment_current
        )

        ah_1hz = np.interp(
            time_1hz,
            segment_time,
            segment_ah
        )

        temperature_1hz = np.interp(
            time_1hz,
            segment_time,
            segment_temperature
        )

        # SOC：公称容量3 Ahを基準に算出
        soc_1hz = 1.0 + (ah_1hz / nominal_capacity_ah)

        # SOT：HG2で実測されたケース温度
        sot_1hz = temperature_1hz.copy()

        processed_segments.append({
            "segment_id": segment_id,
            "Time": time_1hz,
            "Voltage": voltage_1hz,
            "Current": current_1hz,
            "SOC": soc_1hz,
            "SOT": sot_1hz,
        })

    return processed_segments
    
# ==========================================
# 使用する温度条件
# ==========================================
temperature_folders = {
    "-10degC": "n10degC",
    "0degC": "0degC",
    "10degC": "10degC",
    "25degC": "25degC",
}


# ==========================================
# 温度フォルダの確認
# ==========================================
print("対象温度フォルダ")

for temperature_name, folder_name in temperature_folders.items():

    folder = raw_dir / folder_name

    print(
        temperature_name,
        ":",
        folder,
        "| 存在 =", folder.exists()
    )

# ==========================================
# 使用する走行データ
# ==========================================
target_cycles = [
    "Mixed1",
    "Mixed2",
    "Mixed4",
    "Mixed5",
    "Mixed6",
    "Mixed7",
    "Mixed8",
    "UDDS",
    "LA92",
    "US06",
    "HWFET",
]


# ==========================================
# データ分割
# ==========================================
dataset_split = {
    "train": [
        "Mixed2",
        "Mixed4",
        "Mixed5",
        "Mixed6",
        "Mixed7",
        "Mixed8",
    ],

    "validation": [
        "Mixed1",
    ],

    "test": [
        "UDDS",
        "LA92",
        "US06",
        "HWFET",
    ],
}


# ==========================================
# 対象MATファイルを確認
# ==========================================
print("\n対象MATファイル")

for temperature_name, folder_name in temperature_folders.items():

    folder = raw_dir / folder_name
    mat_files = list(folder.glob("*.mat"))

    print(f"\n[{temperature_name}]")

    for cycle in target_cycles:

        matched_files = [
            file
            for file in mat_files
            if cycle in file.name
        ]

        print(
            cycle,
            ":",
            len(matched_files),
            "ファイル"
        )

# ==========================================
# 全対象ファイルの品質情報を確認
# ==========================================
print("\n全対象ファイルの品質確認")

for temperature_name, folder_name in temperature_folders.items():

    folder = raw_dir / folder_name
    mat_files = list(folder.glob("*.mat"))

    for cycle in target_cycles:

        matched_files = [
            file
            for file in mat_files
            if cycle in file.name
        ]

        mat_file = matched_files[0]

        data = loadmat(
            mat_file,
            squeeze_me=True,
            struct_as_record=False
        )

        meas = data["meas"]

        time = np.asarray(meas.Time)
        voltage = np.asarray(meas.Voltage)
        current = np.asarray(meas.Current)
        ah = np.asarray(meas.Ah)
        temperature = np.asarray(meas.Battery_Temp_degC)

        time_diff = np.diff(time)

        nan_count = (
            np.isnan(time).sum()
            + np.isnan(voltage).sum()
            + np.isnan(current).sum()
            + np.isnan(ah).sum()
            + np.isnan(temperature).sum()
        )

        duplicate_count = np.sum(time_diff == 0)
        backward_count = np.sum(time_diff < 0)

        print(
            temperature_name,
            cycle,
            "| NaN =", nan_count,
            "| 重複 =", duplicate_count,
            "| 逆戻り =", backward_count,
            "| 最大間隔 =", time_diff.max()
        )

        # ==========================================
# 全対象データを作成・保存
# ==========================================
print("\nSOC・SOTデータを作成します")

for split_name, cycles in dataset_split.items():

    for temperature_name, folder_name in temperature_folders.items():

        # 保存先フォルダ
        save_dir = (
            processed_dir
            / split_name
            / temperature_name
        )

        save_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # 元データの温度フォルダ
        source_folder = raw_dir / folder_name
        mat_files = list(source_folder.glob("*.mat"))

        for cycle in cycles:

            # 25℃ HWFETはテスト対象から除外
            if (
                split_name == "test"
                and temperature_name == "25degC"
                and cycle == "HWFET"
            ):
                print(
                    "除外:",
                    temperature_name,
                    cycle
                )
                continue

            matched_files = [
                file
                for file in mat_files
                if cycle in file.name
            ]

            if len(matched_files) != 1:
                print(
                    "確認が必要:",
                    temperature_name,
                    cycle,
                    "| ファイル数 =",
                    len(matched_files)
                )
                continue

            mat_file = matched_files[0]

            # これまで作成した処理を実行
            processed_segments = process_file(mat_file)

            # 連続区間ごとに別ファイルとして保存
            for segment in processed_segments:

                segment_id = segment["segment_id"]

                save_path = (
                    save_dir
                    / f"{cycle}_segment{segment_id}_1Hz.npz"
                )

                np.savez(
                    save_path,
                    Time=segment["Time"],
                    Voltage=segment["Voltage"],
                    Current=segment["Current"],
                    SOC=segment["SOC"],
                    SOT=segment["SOT"],
                )

                print(
                    "保存:",
                    split_name,
                    temperature_name,
                    cycle,
                    f"segment{segment_id}",
                    "| 点数 =",
                    len(segment["Time"])
                )

print("\nSOC・SOTデータ作成完了")