import wandb
import time
import random
import torch
import torch.nn as nn

# ------------------------------
# 1. 初始化 wandb
# ------------------------------
wandb.init(
    project="wandb_test",  # 你的项目名称
    name="test_run",  # 这次实验的名字
    config={
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 5
    },
    mode="offline"  # "offline" 离线模式，"online" 上传到 wandb.ai
)

config = wandb.config

# ------------------------------
# 2. 构建一个简单模型
# ------------------------------
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 2)
)

optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
loss_fn = nn.MSELoss()

# ------------------------------
# 3. 模拟训练
# ------------------------------
for epoch in range(config.epochs):
    # 模拟一批输入输出
    x = torch.rand(config.batch_size, 4)
    y = torch.rand(config.batch_size, 2)

    pred = model(x)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ------------------------------
    # 4. 记录指标到 wandb
    # ------------------------------
    wandb.log({
        "epoch": epoch,
        "loss": loss.item(),
        "random_metric": random.random()
    })

    print(f"Epoch {epoch}: loss={loss.item():.4f}")

# ------------------------------
# 5. 保存模型文件到 wandb
# ------------------------------
torch.save(model.state_dict(), "model.pt")
artifact = wandb.Artifact("model", type="model")
artifact.add_file("model.pt")
wandb.log_artifact(artifact)
# ------------------------------
# 6. 结束 wandb run
# ------------------------------
wandb.finish()
