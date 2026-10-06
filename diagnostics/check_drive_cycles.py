from pathlib import Path
from scipy.io import loadmat
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

data_dir = (
    BASE_DIR
    / "data"
    / "raw"
    / "LG_HG2_Original"
    / "25degC"
)

mat_files = list(data_dir.glob("*.mat"))

print("0℃フォルダのMATファイル数:", len(mat_files))

for mat_file in mat_files:
    print(mat_file.name)

target_names = [
    "Mixed1",
    "Mixed2",
    "Mixed4",
    "Mixed5",
    "Mixed6",
    "Mixed7",
    "UDDS",
    "HWFET",
    "LA92",
    "US06",
]

print("\n今回確認する走行データ:")

for mat_file in mat_files:
    if any(name in mat_file.name for name in target_names):
        print(mat_file.name)

print("\nAhの確認:")

for mat_file in mat_files:
    if any(name in mat_file.name for name in target_names):

        data = loadmat(
            mat_file,
            squeeze_me=True,
            struct_as_record=False
        )

        meas = data["meas"]

        print(
            mat_file.name,
            "| 開始 =", meas.Ah[0],
            "| 終了 =", meas.Ah[-1],
            "| 最小 =", meas.Ah.min(),
            "| 最大 =", meas.Ah.max()
        )

print("\n時間軸の一括確認:")

for mat_file in mat_files:
    if any(name in mat_file.name for name in target_names):

        data = loadmat(
            mat_file,
            squeeze_me=True,
            struct_as_record=False
        )

        meas = data["meas"]

        time_diff = np.diff(meas.Time)
        zero_count = np.sum(time_diff == 0)

        print(
            mat_file.name,
            "| データ数 =", len(meas.Time),
            "| 平均間隔 =", time_diff.mean(),
            "| 最大間隔 =", time_diff.max(),
            "| 0秒間隔 =", zero_count
        )