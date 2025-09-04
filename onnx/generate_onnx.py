import torch
import torch.nn as nn

# -----------------------------------------
# 定义第一个模型：输入 10 维，输出 7 维
# -----------------------------------------
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 7)
        )

    def forward(self, x):
        # x: [batch_size, 10]
        return self.fc(x)  # [batch_size, 7]

# -----------------------------------------
# 定义第二个模型：输入 12 维，输出 3 维
# -----------------------------------------
class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(12, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        # x: [batch_size, 12]
        return self.fc(x)  # [batch_size, 3]

# 实例化
model1 = Model1()
model2 = Model2()

# 切换到 eval 模式，导出时常用
model1.eval()
model2.eval()

# ------------------------------------------------
# 导出第一个 ONNX 模型
# ------------------------------------------------
# 假设 batch_size=1，输入维度 10
dummy_input1 = torch.randn(1, 10)
torch.onnx.export(
    model1,
    dummy_input1,
    "model1.onnx",
    input_names=["input10"],
    output_names=["output7"],
    opset_version=11
)
print("✅ 已导出 model1.onnx (10→7)")

# ------------------------------------------------
# 准备第二个 ONNX 的输入：前 5 维 + model1 输出的 7 维
# ------------------------------------------------
# 演示如何在 PyTorch 中拼接这两部分
with torch.no_grad():
    out1 = model1(dummy_input1)               # [1, 7]
    first5 = dummy_input1[:, :5]              # [1, 5]
    concat12 = torch.cat([first5, out1], dim=1)  # [1, 12]

# ------------------------------------------------
# 导出第二个 ONNX 模型
# ------------------------------------------------
torch.onnx.export(
    model2,
    concat12,
    "model2.onnx",
    input_names=["input12"],
    output_names=["output3"],
    opset_version=11
)
print("✅ 已导出 model2.onnx (12→3)")

# ------------------------------------------------
# （可选）完整调用流程示例
# ------------------------------------------------
# batch_size=1
x = torch.randn(1, 10)
y1 = model1(x)                             # [1,7]
x2 = torch.cat([x[:, :5], y1], dim=1)      # [1,12]
y2 = model2(x2)                            # [1,3]
print("模型串联输出：", y2)
