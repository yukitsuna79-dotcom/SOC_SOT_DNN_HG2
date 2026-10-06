from pathlib import Path

import numpy as np
from scipy.io import loadmat


# ==========================================
# 1. 読み込むHG2ファイル
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

mat_path = (
    BASE_DIR
    / "data"
    / "raw"
    / "LG_HG2_Original"
    / "0degC"
    / "11-30-18_23.20 589_Mixed1_0degC_LGHG2.mat"
)


# ==========================================
# 2. MATファイル読み込み
# ==========================================

data = loadmat(
    mat_path,
    squeeze_me=True,
    struct_as_record=False
)

meas = data["meas"]

time = np.asarray(meas.Time, dtype=float)
voltage = np.asarray(meas.Voltage, dtype=float)
current = np.asarray(meas.Current, dtype=float)
ah = np.asarray(meas.Ah, dtype=float)
temperature = np.asarray(meas.Battery_Temp_degC, dtype=float)


# ==========================================
# 3. 変換前の状態を確認
# ==========================================

print("====================================")
print("変換前")
print("====================================")

print(f"データ点数 : {len(time)}")
print(f"開始時刻   : {time[0]:.3f} s")
print(f"終了時刻   : {time[-1]:.3f} s")

dt = np.diff(time)

print(f"平均時間間隔 : {np.mean(dt):.6f} s")
print(f"中央値       : {np.median(dt):.6f} s")


# ==========================================
# 4. 1 Hzの時刻軸を作成
# ==========================================

time_1hz = np.arange(
    np.ceil(time[0]),
    np.floor(time[-1]) + 1,
    1.0
)


# ==========================================
# 5. 1 Hz時刻に合わせて線形補間
#    ※今回は動作確認用
# ==========================================

voltage_1hz = np.interp(time_1hz, time, voltage)
current_1hz = np.interp(time_1hz, time, current)
ah_1hz = np.interp(time_1hz, time, ah)
temperature_1hz = np.interp(time_1hz, time, temperature)


# ==========================================
# 6. 変換後の状態を確認
# ==========================================

print("\n====================================")
print("1 Hz変換後")
print("====================================")

print(f"データ点数 : {len(time_1hz)}")
print(f"開始時刻   : {time_1hz[0]:.1f} s")
print(f"終了時刻   : {time_1hz[-1]:.1f} s")

print("\n最初の10点:")

for i in range(min(10, len(time_1hz))):
    print(
        f"{i}: "
        f"Time={time_1hz[i]:.1f}, "
        f"Voltage={voltage_1hz[i]:.4f}, "
        f"Current={current_1hz[i]:.4f}, "
        f"Ah={ah_1hz[i]:.6f}, "
        f"Temp={temperature_1hz[i]:.3f}"
    )