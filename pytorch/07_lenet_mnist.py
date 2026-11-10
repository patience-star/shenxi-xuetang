"""第 7 课 · 卷积神经网络与 LeNet
经典案例：用 LeNet-5（LeCun 等，1998）识别手写数字，并和第 5 课的多层感知机比一比：
参数少得多、准确率更高、对平移更不敏感。

运行：python 07_lenet_mnist.py（CPU 上约 1~2 分钟）
"""

# %% 1. 读取数据（和第 5 课一样）
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

torch.manual_seed(0)
train_raw = datasets.MNIST('data', train=True, download=True)
test_raw = datasets.MNIST('data', train=False, download=True)
X_train = train_raw.data.float().div(255).unsqueeze(1)     # [60000, 1, 28, 28]
X_test = test_raw.data.float().div(255).unsqueeze(1)
y_train, y_test = train_raw.targets, test_raw.targets
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)

# %% 2. 定义 LeNet-5：两次「卷积 + 池化」提取特征，再用全连接层分类
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)   # 6 个 5×5 卷积核：[1, 28, 28] → [6, 28, 28]
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)             # 16 个 5×5×6 卷积核：[6, 14, 14] → [16, 10, 10]
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = (x - 0.1307) / 0.3081
        x = F.max_pool2d(torch.relu(self.conv1(x)), 2)   # 池化把边长减半：→ [6, 14, 14]
        x = F.max_pool2d(torch.relu(self.conv2(x)), 2)   # → [16, 5, 5]
        x = x.flatten(1)                                  # → 400
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = LeNet()
x = torch.zeros(1, 1, 28, 28)                       # 用一张空图走一遍，打印每一步的形状
print('输入        ', tuple(x.shape))
x = torch.relu(model.conv1(x)); print('conv1 + ReLU', tuple(x.shape))
x = F.max_pool2d(x, 2);         print('池化        ', tuple(x.shape))
x = torch.relu(model.conv2(x)); print('conv2 + ReLU', tuple(x.shape))
x = F.max_pool2d(x, 2);         print('池化        ', tuple(x.shape))
print('展平        ', tuple(x.flatten(1).shape))
for name, p in model.named_parameters():
    print(f'{name:12s} {str(tuple(p.shape)):18s} {p.numel():>7,} 个参数')
n_params = sum(p.numel() for p in model.parameters())
print(f'合计 {n_params:,} 个参数（第 5 课的 MLP 是 109,386 个）')

# %% 3. 训练：和 MLP 完全相同的设置，便于对比
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def accuracy(model, X, y, bs=2000):
    model.eval()
    with torch.no_grad():
        acc = sum((model(X[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item() for i in range(0, len(X), bs)) / len(X)
    model.train()
    return acc

history = {'train_loss': [], 'test_acc': []}
t0 = time.time()
for epoch in range(1, 9):
    total, n = 0.0, 0
    for xb, yb in train_loader:
        loss = F.cross_entropy(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
        n += len(xb)
    history['train_loss'].append(total / n)
    history['test_acc'].append(accuracy(model, X_test, y_test))
    print(f'第 {epoch} 轮  训练损失 {total / n:.4f}  测试准确率 {history["test_acc"][-1]:.2%}')
train_secs = time.time() - t0
print(f'训练用时 {train_secs:.0f} 秒')

# %% 4. 卷积核学到了什么？看看第一层的 6 个 5×5 卷积核
w = model.conv1.weight.detach()[:, 0]               # [6, 5, 5]
for k in range(6):
    print(f'卷积核 {k}：', '  '.join(''.join('+' if v > 0.15 else '-' if v < -0.15 else '.' for v in row) for row in w[k]))
print('「+」是正权重、「-」是负权重：正负并排的地方就是在找「边缘」——一侧亮、一侧暗。')

# %% 5. 平移测试：把测试图整体右移 3 个像素，谁更扛得住？
shifted = torch.roll(X_test, shifts=3, dims=3)       # 每张图向右平移 3 像素（MNIST 四周是黑边，平移不会丢笔画）
lenet_acc, lenet_shift = accuracy(model, X_test, y_test), accuracy(model, shifted, y_test)
print(f'LeNet：原图 {lenet_acc:.2%} → 右移 3 像素 {lenet_shift:.2%}')
mlp_acc = mlp_shift = None
if os.path.exists('data/mlp_mnist.pt'):              # 第 5 课保存的 MLP（没跑过第 5 课就跳过这项对比）
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1, self.fc2, self.fc3 = nn.Linear(784, 128), nn.Linear(128, 64), nn.Linear(64, 10)

        def forward(self, x):
            x = ((x - 0.1307) / 0.3081).flatten(1)
            return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

    mlp = MLP()
    mlp.load_state_dict(torch.load('data/mlp_mnist.pt', weights_only=True))
    mlp_acc, mlp_shift = accuracy(mlp, X_test, y_test), accuracy(mlp, shifted, y_test)
    print(f'MLP  ：原图 {mlp_acc:.2%} → 右移 3 像素 {mlp_shift:.2%}')
    print('卷积核在整张图上滑动（同一套权重处处共用），笔画挪了位置照样能被认出来；MLP 的每个权重只盯一个固定的像素。')

# %% 导出给网站（学习时可跳过）
model.eval()
with torch.no_grad():
    pred = torch.cat([model(X_test[i:i + 2000]).argmax(1) for i in range(0, 10000, 2000)])
wrong = (pred != y_test).nonzero().squeeze(1)
torch.manual_seed(1)
demo_idx = torch.randperm(10000)[:24]
site_export = {
    'model': model,
    'golden': (X_test[:8], lambda x: model(x)),
    'history': history,
    'summary': {
        'test_acc': f'{history["test_acc"][-1]:.2%}', 'params': f'{n_params:,}', 'secs': f'{train_secs:.0f}', 'n_wrong': len(wrong),
        'lenet_shift': f'{lenet_shift:.1%}', 'mlp_acc': f'{mlp_acc:.1%}' if mlp_acc else '—', 'mlp_shift': f'{mlp_shift:.1%}' if mlp_shift else '—',
    },
    'extras': {
        'demo_x': (X_test[demo_idx, 0] * 255).round().to(torch.uint8), 'demo_y': y_test[demo_idx].tolist(),
        'wrong_x': (X_test[wrong[:16], 0] * 255).round().to(torch.uint8),
        'wrong_y': y_test[wrong[:16]].tolist(), 'wrong_p': pred[wrong[:16]].tolist(),
    },
}
