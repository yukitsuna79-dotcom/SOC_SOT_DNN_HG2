from pathlib import Path
import numpy as np


# ==========================================
# processedデータの場所
# ==========================================
project_dir = Path(__file__).resolve().parent
processed_dir = project_dir / "data" / "processed"


# ==========================================
# 最終確認
# ==========================================
npz_files = sorted(processed_dir.rglob("*.npz"))

print("==========================================")
print("HG2 SOC・SOTデータ 最終確認")
print("==========================================")
print("NPZファイル数 :", len(npz_files))

required_keys = {
    "Time",
    "Voltage",
    "Current",
    "SOC",
    "SOT",
}

problem_count = 0


for file in npz_files:

    data = np.load(file)

    keys = set(data.files)

    # 必要なデータが存在するか
    if keys != required_keys:
        print("\n[キー異常]", file)
        print("キー :", data.files)
        problem_count += 1
        continue

    time = data["Time"]
    voltage = data["Voltage"]
    current = data["Current"]
    soc = data["SOC"]
    sot = data["SOT"]

    # 各系列の長さ
    lengths = [
        len(time),
        len(voltage),
        len(current),
        len(soc),
        len(sot),
    ]

    if len(set(lengths)) != 1:
        print("\n[長さ異常]", file)
        print("長さ :", lengths)
        problem_count += 1

    # NaN / inf
    arrays = {
        "Time": time,
        "Voltage": voltage,
        "Current": current,
        "SOC": soc,
        "SOT": sot,
    }

    for name, array in arrays.items():

        if not np.all(np.isfinite(array)):
            print(
                "\n[NaN / inf]",
                file,
                name
            )
            problem_count += 1

    # 1 Hzになっているか
    if len(time) >= 2:

        time_diff = np.diff(time)

        if not np.allclose(time_diff, 1.0):
            print("\n[時間間隔異常]", file)
            print(
                "最小 =",
                time_diff.min(),
                "最大 =",
                time_diff.max()
            )
            problem_count += 1


print("\n==========================================")
print("確認結果")
print("==========================================")

if problem_count == 0:
    print("異常は検出されませんでした。")
else:
    print("確認が必要な項目 :", problem_count)


# ==========================================
# SOC・SOTの全体範囲
# ==========================================
all_soc = []
all_sot = []

for file in npz_files:

    data = np.load(file)

    if "SOC" in data.files:
        all_soc.append(data["SOC"])

    if "SOT" in data.files:
        all_sot.append(data["SOT"])


if all_soc and all_sot:

    all_soc = np.concatenate(all_soc)
    all_sot = np.concatenate(all_sot)

    print("\nSOC 全体範囲")
    print("最小 :", all_soc.min())
    print("最大 :", all_soc.max())

    print("\nSOT 全体範囲 [degC]")
    print("最小 :", all_sot.min())
    print("最大 :", all_sot.max())