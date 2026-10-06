import torch
import torch.nn as nn


class BaselineDFNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 第1全結合層：Voltage, Current → 110ニューロン
        self.fc1 = nn.Linear(2, 110)
        self.tanh1 = nn.Tanh()

        # 第2全結合層：110 → 110ニューロン
        self.fc2 = nn.Linear(110, 110)
        self.tanh2 = nn.Tanh()

        # 出力層：SOC, SOT
        self.fc_out = nn.Linear(110, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh1(x)

        x = self.fc2(x)
        x = self.tanh2(x)

        x = self.fc_out(x)

        return x


if __name__ == "__main__":
    model = BaselineDFNN()

    # 動作確認用：VoltageとCurrentの2入力
    test_input = torch.randn(4, 2)
    test_output = model(test_input)

    print(model)
    print("入力サイズ :", test_input.shape)
    print("出力サイズ :", test_output.shape)