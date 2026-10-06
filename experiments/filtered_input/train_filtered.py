import random
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model_filtered import FilteredInputDFNN
from data_loader_filtered import (
    load_split,
    fit_standardizer,
    standardize,
)


SEEDS = [1, 2, 3]

LEARNING_RATE = 0.01
BATCH_SIZE = 128
MAX_EPOCHS = 1200

DEVICE = torch.device("cpu")

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_DIR / "results"


# ============================================================
# データ読み込み
# ============================================================

print("Filtered Inputデータを読み込んでいます...")

X_train, Y_train = load_split(
    "train"
)

X_val, Y_val = load_split(
    "validation"
)


# ============================================================
# Trainingデータのみから標準化係数を計算
# ============================================================

x_mean, x_std = fit_standardizer(
    X_train
)

y_mean, y_std = fit_standardizer(
    Y_train
)

X_train_std = standardize(
    X_train,
    x_mean,
    x_std,
)

Y_train_std = standardize(
    Y_train,
    y_mean,
    y_std,
)

X_val_std = standardize(
    X_val,
    x_mean,
    x_std,
)

Y_val_std = standardize(
    Y_val,
    y_mean,
    y_std,
)


X_train_tensor = torch.tensor(
    X_train_std,
    dtype=torch.float32,
)

Y_train_tensor = torch.tensor(
    Y_train_std,
    dtype=torch.float32,
)

X_val_tensor = torch.tensor(
    X_val_std,
    dtype=torch.float32,
).to(DEVICE)

Y_val_tensor = torch.tensor(
    Y_val_std,
    dtype=torch.float32,
).to(DEVICE)


train_dataset = TensorDataset(
    X_train_tensor,
    Y_train_tensor,
)

criterion = nn.MSELoss()


# ============================================================
# Seedごとの学習
# ============================================================

for seed in SEEDS:

    print()
    print("=" * 70)
    print(
        f"Filtered Input DFNN - Seed {seed}"
    )
    print("=" * 70)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # SeedごとにDataLoaderを作成
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    model = FilteredInputDFNN().to(
        DEVICE
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    result_dir = (
        RESULTS_DIR
        / f"filtered_input_seed_{seed}"
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_file = (
        result_dir
        / "training_history.csv"
    )

    model_file = (
        result_dir
        / "best_model.pth"
    )

    standardizer_file = (
        result_dir
        / "standardizer.npz"
    )

    np.savez(
        standardizer_file,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )

    history = []

    best_val_score = float("inf")
    best_epoch = 0

    print("Network      : 6 -> 110 -> 110 -> 2")
    print("Activation   : Tanh -> Tanh")
    print("Optimizer    : SGD")
    print("Learning Rate:", LEARNING_RATE)
    print("Batch Size   :", BATCH_SIZE)
    print("Max Epochs   :", MAX_EPOCHS)
    print("Seed         :", seed)
    print(
        "Training samples   :",
        len(X_train_tensor),
    )
    print(
        "Validation samples :",
        len(X_val_tensor),
    )

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        # ====================================================
        # Training
        # ====================================================

        model.train()

        train_loss_sum = 0.0

        for batch_X, batch_Y in train_loader:

            batch_X = batch_X.to(
                DEVICE
            )

            batch_Y = batch_Y.to(
                DEVICE
            )

            optimizer.zero_grad()

            prediction = model(
                batch_X
            )

            loss_soc = criterion(
                prediction[:, 0],
                batch_Y[:, 0],
            )

            loss_sot = criterion(
                prediction[:, 1],
                batch_Y[:, 1],
            )

            total_loss = (
                loss_soc
                + loss_sot
            )

            total_loss.backward()

            optimizer.step()

            train_loss_sum += (
                total_loss.item()
                * len(batch_X)
            )

        train_loss = (
            train_loss_sum
            / len(train_dataset)
        )

        # ====================================================
        # Validation
        # ====================================================

        model.eval()

        with torch.no_grad():

            val_prediction_std = model(
                X_val_tensor
            )

            val_soc_loss = criterion(
                val_prediction_std[:, 0],
                Y_val_tensor[:, 0],
            )

            val_sot_loss = criterion(
                val_prediction_std[:, 1],
                Y_val_tensor[:, 1],
            )

            val_total_loss = (
                val_soc_loss
                + val_sot_loss
            ).item()

            val_prediction = (
                val_prediction_std
                .cpu()
                .numpy()
                * y_std
                + y_mean
            )

        # ====================================================
        # Validation物理単位評価
        # ====================================================

        soc_error = (
            val_prediction[:, 0]
            - Y_val[:, 0]
        )

        sot_error = (
            val_prediction[:, 1]
            - Y_val[:, 1]
        )

        soc_rmse = (
            np.sqrt(
                np.mean(
                    soc_error ** 2
                )
            )
            * 100.0
        )

        soc_mae = (
            np.mean(
                np.abs(
                    soc_error
                )
            )
            * 100.0
        )

        sot_rmse = np.sqrt(
            np.mean(
                sot_error ** 2
            )
        )

        sot_mae = np.mean(
            np.abs(
                sot_error
            )
        )

        # ====================================================
        # Best model
        # ====================================================

        if val_total_loss < best_val_score:

            best_val_score = (
                val_total_loss
            )

            best_epoch = epoch

            torch.save(
                model.state_dict(),
                model_file,
            )

        history.append(
            [
                epoch,
                train_loss,
                val_total_loss,
                soc_rmse,
                soc_mae,
                sot_rmse,
                sot_mae,
            ]
        )

        if (
            epoch == 1
            or epoch % 30 == 0
        ):

            print(
                f"Epoch {epoch:4d} | "
                f"Train Loss {train_loss:.6f} | "
                f"Val Loss {val_total_loss:.6f} | "
                f"SOC RMSE {soc_rmse:.3f}% | "
                f"SOC MAE {soc_mae:.3f}% | "
                f"SOT RMSE {sot_rmse:.3f} degC | "
                f"SOT MAE {sot_mae:.3f} degC"
            )

    # ========================================================
    # CSV保存
    # ========================================================

    with open(
        history_file,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "Epoch",
                "Train_Loss",
                "Validation_Loss",
                "SOC_RMSE_percent",
                "SOC_MAE_percent",
                "SOT_RMSE_degC",
                "SOT_MAE_degC",
            ]
        )

        writer.writerows(
            history
        )

    print()
    print(
        f"Seed {seed} 学習完了"
    )
    print(
        "Best Epoch :",
        best_epoch,
    )
    print(
        "Best Validation Loss :",
        best_val_score,
    )
    print(
        "Best Model :",
        model_file,
    )


print()
print("=" * 70)
print("Filtered Input Seed 1-3 学習完了")
print("=" * 70)