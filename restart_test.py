import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# 简单数据
x = torch.randn(1000, 10)
y = torch.randn(1000, 1)
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 模型和优化器
model = nn.Linear(10, 1).cuda()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# checkpoint 路径
ckpt_path = "checkpoint.pth"
start_epoch = 0

# 恢复
if os.path.exists(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed from epoch {start_epoch}")

# 训练
num_epochs = 5
for epoch in range(start_epoch, num_epochs):
    epoch_loss = 0
    for inputs, targets in loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    # 保存
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, ckpt_path)
    print(f"Checkpoint saved at epoch {epoch+1}")
