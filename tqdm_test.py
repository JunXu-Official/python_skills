import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# 简单模型：单层全连接
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)
    def forward(self, x):
        return self.fc(x)

def main():
    # 构造假数据：1000个样本，输入维度10，标签1维
    x = torch.randn(1000, 10)
    y = torch.randn(1000, 1)

    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SimpleNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 5
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        loop = tqdm(dataloader, desc="Training", dynamic_ncols=True, file=sys.stdout)
        for inputs, targets in loop:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # 更新进度条描述，显示当前loss
            loop.set_postfix(loss=loss.item())

if __name__ == "__main__":
    main()
