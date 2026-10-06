import random
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model import BaselineDFNN
from data_loader import load_split, fit_standardizer, standardize


# ============================================================
# 1. 予備学習条件
# ============================================================

SEED = 3
LEARNING_RATE = 0.01
BATCH_SIZE = 128
MAX_EPOCHS = 1200

# CPUを使用
DEVICE = torch.device("cpu")


# ============================================================
# 2. Seed固定
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 3. HG2データ読み込み
# ============================================================

X_train, Y_train = load_split("train")
X_val, Y_val = load_split("validation")


# ============================================================
# 4. Trainingデータだけから標準化係数を計算
# ============================================================

x_mean, x_std = fit_standardizer(X_train)
y_mean, y_std = fit_standardizer(Y_train)

X_train_std = standardize(X_train, x_mean, x_std)
Y_train_std = standardize(Y_train, y_mean, y_std)

X_val_std = standardize(X_val, x_mean, x_std)
Y_val_std = standardize(Y_val, y_mean, y_std)


# ============================================================
# 5. NumPy → PyTorch Tensor
# ============================================================

X_train_tensor = torch.tensor(
    X_train_std,
    dtype=torch.float32
)

Y_train_tensor = torch.tensor(
    Y_train_std,
    dtype=torch.float32
)

X_val_tensor = torch.tensor(
    X_val_std,
    dtype=torch.float32
)

Y_val_tensor = torch.tensor(
    Y_val_std,
    dtype=torch.float32
)


# ============================================================
# 6. Training用DataLoader
# ============================================================

train_dataset = TensorDataset(
    X_train_tensor,
    Y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================================
# 7. 全結合層Baseline DFNN
# ============================================================

model = BaselineDFNN().to(DEVICE)


# ============================================================
# 8. Loss と SGD
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 9. 保存先
# ============================================================

result_dir = Path("results") / f"baseline_seed_{SEED}"
result_dir.mkdir(parents=True, exist_ok=True)

history_file = result_dir / "training_history.csv"
model_file = result_dir / "best_model.pth"
standardizer_file = result_dir / "standardizer.npz"


# 標準化係数を保存
np.savez(
    standardizer_file,
    x_mean=x_mean,
    x_std=x_std,
    y_mean=y_mean,
    y_std=y_std
)


# ============================================================
# 10. 学習履歴
# ============================================================

history = []

best_val_score = float("inf")
best_epoch = 0


# ============================================================
# 11. 学習開始
# ============================================================

print("============================================================")
print("全結合層 Baseline DFNN 予備学習")
print("============================================================")

print(f"Device        : {DEVICE}")
print(f"Seed          : {SEED}")
print(f"Learning Rate : {LEARNING_RATE}")
print(f"Batch Size    : {BATCH_SIZE}")
print(f"Max Epochs    : {MAX_EPOCHS}")

print()
print("Network : 2 -> 110 -> 110 -> 2")
print("Activation : Tanh -> Tanh")
print("Optimizer : SGD")

print()
print("Training samples   :", len(X_train_tensor))
print("Validation samples :", len(X_val_tensor))

print("============================================================")


# ============================================================
# 12. Epochループ
# ============================================================

for epoch in range(1, MAX_EPOCHS + 1):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    train_loss_sum = 0.0

    for batch_X, batch_Y in train_loader:

        batch_X = batch_X.to(DEVICE)
        batch_Y = batch_Y.to(DEVICE)

        optimizer.zero_grad()

        prediction = model(batch_X)

        # SOC Loss
        loss_soc = criterion(
            prediction[:, 0],
            batch_Y[:, 0]
        )

        # SOT Loss
        loss_sot = criterion(
            prediction[:, 1],
            batch_Y[:, 1]
        )

        # SOCとSOTを同じ重みで学習
        total_loss = loss_soc + loss_sot

        total_loss.backward()

        optimizer.step()

        train_loss_sum += (
            total_loss.item() * len(batch_X)
        )


    train_loss = (
        train_loss_sum / len(train_dataset)
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        val_prediction_std = model(
            X_val_tensor.to(DEVICE)
        )

        val_prediction_std = (
            val_prediction_std.cpu().numpy()
        )


    # --------------------------------------------------------
    # 標準化を元に戻す
    # --------------------------------------------------------

    val_prediction = (
        val_prediction_std * y_std
        + y_mean
    )


    # --------------------------------------------------------
    # SOC評価
    # --------------------------------------------------------

    soc_error = (
        val_prediction[:, 0]
        - Y_val[:, 0]
    )

    soc_rmse = np.sqrt(
        np.mean(soc_error ** 2)
    )

    soc_mae = np.mean(
        np.abs(soc_error)
    )


    # --------------------------------------------------------
    # SOT評価
    # --------------------------------------------------------

    sot_error = (
        val_prediction[:, 1]
        - Y_val[:, 1]
    )

    sot_rmse = np.sqrt(
        np.mean(sot_error ** 2)
    )

    sot_mae = np.mean(
        np.abs(sot_error)
    )


    # --------------------------------------------------------
    # 暫定的なモデル保存判定
    # --------------------------------------------------------
    #
    # SOCとSOTは単位が違うため、
    # 物理単位のRMSEを直接足さない。
    #
    # 標準化空間のValidation MSEを使用する。
    # --------------------------------------------------------

    with torch.no_grad():

        val_target_std = Y_val_tensor.to(DEVICE)

        val_prediction_for_loss = model(
            X_val_tensor.to(DEVICE)
        )

        val_soc_loss = criterion(
            val_prediction_for_loss[:, 0],
            val_target_std[:, 0]
        )

        val_sot_loss = criterion(
            val_prediction_for_loss[:, 1],
            val_target_std[:, 1]
        )

        val_total_loss = (
            val_soc_loss + val_sot_loss
        ).item()


    # --------------------------------------------------------
    # Best model保存
    # --------------------------------------------------------

    if val_total_loss < best_val_score:

        best_val_score = val_total_loss
        best_epoch = epoch

        torch.save(
            model.state_dict(),
            model_file
        )


    # --------------------------------------------------------
    # 履歴保存用
    # --------------------------------------------------------

    history.append([
        epoch,
        train_loss,
        val_total_loss,
        soc_rmse * 100.0,
        soc_mae * 100.0,
        sot_rmse,
        sot_mae
    ])


    # --------------------------------------------------------
    # 30 Epochごとに表示
    # --------------------------------------------------------

    if epoch == 1 or epoch % 30 == 0:

        print(
            f"Epoch {epoch:4d} | "
            f"Train Loss {train_loss:.6f} | "
            f"Val Loss {val_total_loss:.6f} | "
            f"SOC RMSE {soc_rmse * 100:.3f}% | "
            f"SOC MAE {soc_mae * 100:.3f}% | "
            f"SOT RMSE {sot_rmse:.3f} degC | "
            f"SOT MAE {sot_mae:.3f} degC"
        )


# ============================================================
# 13. CSV保存
# ============================================================

with open(
    history_file,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Epoch",
        "Train_Loss",
        "Validation_Loss",
        "SOC_RMSE_percent",
        "SOC_MAE_percent",
        "SOT_RMSE_degC",
        "SOT_MAE_degC"
    ])

    writer.writerows(history)


# ============================================================
# 14. 終了表示
# ============================================================

print()
print("============================================================")
print("予備学習が完了しました")
print("============================================================")

print("Best Epoch :", best_epoch)
print("Best Validation Loss :", best_val_score)

print()
print("Best Model :")
print(model_file)

print()
print("Training History :")
print(history_file)

print()
print("Standardizer :")
print(standardizer_file)