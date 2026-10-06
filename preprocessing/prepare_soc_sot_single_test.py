import numpy as np
from pathlib import Path
from scipy.io import loadmat


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
# 動作確認用データ
# 0℃ Mixed1
# ==========================================
input_file = (
    raw_dir
    / "0degC"
    / "11-30-18_23.20 589_Mixed1_0degC_LGHG2.mat"
)


# ==========================================
# MATファイル読み込み
# ==========================================
data = loadmat(
    input_file,
    squeeze_me=True,
    struct_as_record=False
)

meas = data["meas"]

# ==========================================
# 必要なデータを取り出す
# ==========================================
time = meas.Time
voltage = meas.Voltage
current = meas.Current
ah = meas.Ah
temperature = meas.Battery_Temp_degC


print("読み込み成功")
print("ファイル :", input_file.name)
print("データ数 :", len(meas.Time))

print("\n各データの長さ")
print("Time        :", len(time))
print("Voltage     :", len(voltage))
print("Current     :", len(current))
print("Ah          :", len(ah))
print("Temperature :", len(temperature))

# ==========================================
# NaNの確認
# ==========================================
print("\nNaNの確認")
print("Time        :", np.isnan(time).sum())
print("Voltage     :", np.isnan(voltage).sum())
print("Current     :", np.isnan(current).sum())
print("Ah          :", np.isnan(ah).sum())
print("Temperature :", np.isnan(temperature).sum())

# ==========================================
# 時刻重複の確認
# ==========================================
time_diff = np.diff(time)

duplicate_count = np.sum(time_diff == 0)
backward_count = np.sum(time_diff < 0)

print("\n時間軸の確認")
print("同一時刻 :", duplicate_count)
print("時刻逆戻り :", backward_count)
print("最小時間間隔 :", time_diff.min())
print("最大時間間隔 :", time_diff.max())

# ==========================================
# 同一時刻の重複を除去
# ==========================================
keep = np.concatenate(([True], np.diff(time) != 0))

time = time[keep]
voltage = voltage[keep]
current = current[keep]
ah = ah[keep]
temperature = temperature[keep]

print("\n重複除去後")
print("データ数 :", len(time))
print("同一時刻 :", np.sum(np.diff(time) == 0))
print("時刻逆戻り :", np.sum(np.diff(time) < 0))

# ==========================================
# 1 Hz時間軸を作成
# ==========================================
start_time = np.ceil(time[0])
end_time = np.floor(time[-1])

time_1hz = np.arange(
    start_time,
    end_time + 1.0,
    1.0
)

print("\n1 Hz時間軸")
print("開始時刻 :", time_1hz[0])
print("終了時刻 :", time_1hz[-1])
print("データ数 :", len(time_1hz))
print("時間間隔 :", np.unique(np.diff(time_1hz)))

# ==========================================
# 各データを1 Hz時間軸へ対応
# ==========================================
voltage_1hz = np.interp(time_1hz, time, voltage)
current_1hz = np.interp(time_1hz, time, current)
ah_1hz = np.interp(time_1hz, time, ah)
temperature_1hz = np.interp(time_1hz, time, temperature)

print("\n1 Hz化後")
print("Time        :", len(time_1hz))
print("Voltage     :", len(voltage_1hz))
print("Current     :", len(current_1hz))
print("Ah          :", len(ah_1hz))
print("Temperature :", len(temperature_1hz))

print("\nNaN")
print("Voltage     :", np.isnan(voltage_1hz).sum())
print("Current     :", np.isnan(current_1hz).sum())
print("Ah          :", np.isnan(ah_1hz).sum())
print("Temperature :", np.isnan(temperature_1hz).sum())

# ==========================================
# SOC・SOT教師データを作成
# ==========================================
nominal_capacity_ah = 3.0

soc_1hz = 1.0 + (ah_1hz / nominal_capacity_ah)
sot_1hz = temperature_1hz.copy()

print("\nSOC・SOT")
print("SOC 開始 :", soc_1hz[0])
print("SOC 終了 :", soc_1hz[-1])
print("SOC 最小 :", soc_1hz.min())
print("SOC 最大 :", soc_1hz.max())

print("SOT 開始 :", sot_1hz[0])
print("SOT 終了 :", sot_1hz[-1])
print("SOT 最小 :", sot_1hz.min())
print("SOT 最大 :", sot_1hz.max())

# ==========================================
# 1 Hz SOC・SOTデータを保存
# ==========================================
output_file = processed_dir / "Mixed1_0degC_1Hz.npz"

np.savez(
    output_file,
    Time=time_1hz,
    Voltage=voltage_1hz,
    Current=current_1hz,
    SOC=soc_1hz,
    SOT=sot_1hz
)

print("\n保存完了")
print("保存先 :", output_file)