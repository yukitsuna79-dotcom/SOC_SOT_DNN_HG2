from pathlib import Path
from scipy.io import loadmat

# プロジェクトフォルダ
BASE_DIR = Path(__file__).resolve().parent

# 0℃ Mixed1 のMATファイル
mat_path = (
    BASE_DIR
    / "data"
    / "raw"
    / "LG_HG2_Original"
    / "0degC"
    / "01-12-18_04.35 589_Mixed2_0degC_LGHG2.mat"
)

print("読み込みファイル:")
print(mat_path)

# MATファイルを読み込む
data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

print("\nMATファイル内の変数:")
print([key for key in data.keys() if not key.startswith("__")])

# meas構造体を取得
meas = data["meas"]

print("\nmeas内のデータ:")
print("Time               :", meas.Time.shape)
print("Voltage            :", meas.Voltage.shape)
print("Current            :", meas.Current.shape)
print("Ah                 :", meas.Ah.shape)
print("Battery_Temp_degC  :", meas.Battery_Temp_degC.shape)

print("\n最初の5点:")
for i in range(5):
    print(
        f"{i}: "
        f"Time={meas.Time[i]:.3f}, "
        f"Voltage={meas.Voltage[i]:.4f}, "
        f"Current={meas.Current[i]:.4f}, "
        f"Ah={meas.Ah[i]:.6f}, "
        f"Temp={meas.Battery_Temp_degC[i]:.3f}"
    )

print("\nAhの確認")
print("開始時 Ah =", meas.Ah[0])
print("終了時 Ah =", meas.Ah[-1])

print("Ah 最小値 =", meas.Ah.min())
print("Ah 最大値 =", meas.Ah.max())