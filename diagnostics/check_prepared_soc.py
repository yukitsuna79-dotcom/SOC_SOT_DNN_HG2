from scipy.io import loadmat

mat_path = (
    r"E:\SOC_DNN_MathWorks_SGD\data\raw\LGHG2_dataset\Train"
    r"\TRAIN_LGHG2@n10degC_to_25degC_Norm_5Inputs.mat"
)

data = loadmat(mat_path)

print("MATファイル内の変数:")
print([key for key in data.keys() if not key.startswith("__")])

print("\nYの確認:")
print("Yの形 =", data["Y"].shape)
print("Y 最小値 =", data["Y"].min())
print("Y 最大値 =", data["Y"].max())

print("Y 最初の値 =", data["Y"][0, 0])
print("Y 最後の値 =", data["Y"][0, -1])