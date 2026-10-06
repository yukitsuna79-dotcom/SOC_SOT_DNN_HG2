from pathlib import Path
from scipy.io import loadmat
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

file_path = (
    BASE_DIR
    / "data"
    / "raw"
    / "LG_HG2_Original"
    / "10degC"
    / "11-25-18_20.59 582_LA92_10degC_LGHG2.mat"
)

data = loadmat(
    file_path,
    squeeze_me=True,
    struct_as_record=False
)

meas = data["meas"]

print("Time              :", len(meas.Time))
print("Voltage           :", len(meas.Voltage))
print("Current           :", len(meas.Current))
print("Ah                :", len(meas.Ah))
print("Battery_Temp_degC :", len(meas.Battery_Temp_degC))

print("\n最初の10点:")

for i in range(10):
    print(
        i,
        "| Time =", meas.Time[i],
        "| Voltage =", meas.Voltage[i],
        "| Current =", meas.Current[i],
        "| Ah =", meas.Ah[i],
        "| Temp =", meas.Battery_Temp_degC[i]
    )

time_diff = np.diff(meas.Time)

print("\n時間間隔の確認:")
print("最小間隔 :", time_diff.min(), "秒")
print("最大間隔 :", time_diff.max(), "秒")
print("平均間隔 :", time_diff.mean(), "秒")

zero_diff_count = np.sum(time_diff == 0)

print("\n同一時刻の確認:")
print("時間間隔が0秒の箇所 :", zero_diff_count, "個")

zero_indices = np.where(time_diff == 0)[0]

print("\n同一時刻の詳細:")

for i in zero_indices[:5]:
    print("インデックス:", i, "と", i + 1)

    print(
        "1点目:",
        "Time =", meas.Time[i],
        "| Voltage =", meas.Voltage[i],
        "| Current =", meas.Current[i],
        "| Ah =", meas.Ah[i],
        "| Temp =", meas.Battery_Temp_degC[i]
    )

    print(
        "2点目:",
        "Time =", meas.Time[i + 1],
        "| Voltage =", meas.Voltage[i + 1],
        "| Current =", meas.Current[i + 1],
        "| Ah =", meas.Ah[i + 1],
        "| Temp =", meas.Battery_Temp_degC[i + 1]
    )

max_gap_index = np.argmax(time_diff)

print("\n最大時間間隔の詳細:")
print("インデックス:", max_gap_index, "と", max_gap_index + 1)

print(
    "直前:",
    "Time =", meas.Time[max_gap_index],
    "| Voltage =", meas.Voltage[max_gap_index],
    "| Current =", meas.Current[max_gap_index],
    "| Ah =", meas.Ah[max_gap_index],
    "| Temp =", meas.Battery_Temp_degC[max_gap_index]
)

print(
    "直後:",
    "Time =", meas.Time[max_gap_index + 1],
    "| Voltage =", meas.Voltage[max_gap_index + 1],
    "| Current =", meas.Current[max_gap_index + 1],
    "| Ah =", meas.Ah[max_gap_index + 1],
    "| Temp =", meas.Battery_Temp_degC[max_gap_index + 1]
)

print(
    "時間差 =",
    time_diff[max_gap_index],
    "秒"
)

print("\nNaNの個数:")
print("Voltage =", np.isnan(meas.Voltage).sum())
print("Current =", np.isnan(meas.Current).sum())
print("Ah =", np.isnan(meas.Ah).sum())
print("Temp =", np.isnan(meas.Battery_Temp_degC).sum())

nan_indices = np.where(
    np.isnan(meas.Voltage) |
    np.isnan(meas.Current) |
    np.isnan(meas.Ah) |
    np.isnan(meas.Battery_Temp_degC)
)[0]

print("\nNaNの位置:")
for i in nan_indices:
    print(
        "index =", i,
        "| Time =", meas.Time[i],
        "| Voltage =", meas.Voltage[i],
        "| Current =", meas.Current[i],
        "| Ah =", meas.Ah[i],
        "| Temp =", meas.Battery_Temp_degC[i]
    )
