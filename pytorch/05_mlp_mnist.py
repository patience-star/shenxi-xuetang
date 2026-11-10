"""第 5 课 · 多层感知机（MLP）
经典案例：手写数字识别（MNIST）—— 784 → 128 → 64 → 10 的全连接网络。

运行：python 05_mlp_mnist.py（CPU 上约 1 分钟；首次运行会自动下载 MNIST）
"""

# %% 1. 读取数据：MNIST 很小，直接整个读进内存
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

torch.manual_seed(0)
train_raw = datasets.MNIST('data', train=True, download=True)
test_raw = datasets.MNIST('data', train=False, download=True)
# .data 是 uint8 的 [样本数, 28, 28]；除以 255 变成 0~1，再加一个通道维 → [N, 1, 28, 28]
X_train = train_raw.data.float().div(255).unsqueeze(1)
X_test = test_raw.data.float().div(255).unsqueeze(1)
y_train, y_test = train_raw.targets, test_raw.targets
print('训练集', tuple(X_train.shape), '测试集', tuple(X_test.shape))
print('各数字的训练样本数：', torch.bincount(y_train).tolist())     # 大致均衡，准确率是合理的指标

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)

# %% 2. 定义网络：三个全连接层，中间用 ReLU
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)     # 第一层：784 个像素 → 128 个隐藏神经元
        self.fc2 = nn.Linear(128, 64)      # 第二层：128 → 64
        self.fc3 = nn.Linear(64, 10)       # 输出层：64 → 10 个数字各一个「得分」

    def forward(self, x):
        x = (x - 0.1307) / 0.3081          # 标准化写进模型里：部署时直接喂 0~1 的像素即可，不会忘
        x = x.flatten(1)                   # [N, 1, 28, 28] → [N, 784]：全连接层只认向量
        x = torch.relu(self.fc1(x))        # 线性变换后过 ReLU：没有它，三层叠起来仍然只是一个线性函数
        x = torch.relu(self.fc2(x))
        return self.fc3(x)                 # 输出 logits（未归一化的得分），softmax 交给损失函数做

model = MLP()
n_params = sum(p.numel() for p in model.parameters())
print(model)
print(f'参数总数：{n_params:,}（其中第一层就占了 784×128 + 128 = {784 * 128 + 128:,} 个）')

# %% 3. 训练：交叉熵损失 + Adam 优化器
loss_fn = nn.CrossEntropyLoss()            # = log_softmax + 负对数似然；输入 logits，标签是整数类别
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def accuracy(model, X, y):
    model.eval()                           # 切到评估模式（本模型无 Dropout/BN，但养成习惯）
    with torch.no_grad():                  # 评估不需要梯度，省时省内存
        return (model(X).argmax(1) == y).float().mean().item()

history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
t0 = time.time()
for epoch in range(1, 9):
    model.train()
    total, n = 0.0, 0
    for xb, yb in train_loader:
        loss = loss_fn(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
        n += len(xb)
    history['train_loss'].append(total / n)
    history['train_acc'].append(accuracy(model, X_train, y_train))
    history['test_acc'].append(accuracy(model, X_test, y_test))
    print(f'第 {epoch} 轮  训练损失 {total / n:.4f}  训练准确率 {history["train_acc"][-1]:.2%}  测试准确率 {history["test_acc"][-1]:.2%}')
train_secs = time.time() - t0
print(f'训练用时 {train_secs:.0f} 秒')

# %% 4. 看看模型在哪些地方犯错
model.eval()
with torch.no_grad():
    logits = model(X_test)
    probs = torch.softmax(logits, dim=1)   # softmax：把 10 个得分变成加起来等于 1 的概率
pred = probs.argmax(1)
wrong = (pred != y_test).nonzero().squeeze(1)
print(f'测试集 10000 张里错了 {len(wrong)} 张')
for d in range(10):                        # 每个数字的准确率
    m = y_test == d
    print(f'数字 {d}：准确率 {(pred[m] == d).float().mean().item():.2%}', end='   ' if d % 2 == 0 else '\n')
# 最常见的混淆：真实是 a、预测成 b
conf = torch.zeros(10, 10, dtype=torch.long)
for t, p in zip(y_test.tolist(), pred.tolist()):
    conf[t, p] += 1
pairs = sorted(((conf[a, b].item(), a, b) for a in range(10) for b in range(10) if a != b), reverse=True)[:5]
print('最常见的错误：', '；'.join(f'{a} 被认成 {b}（{c} 次）' for c, a, b in pairs))

# %% 5. 看一张图片的完整推理过程
i = wrong[0].item()
print(f'第 {i} 张测试图：真实标签 {y_test[i].item()}，模型预测 {pred[i].item()}')
print('10 个得分 logits：', [f'{v:.1f}' for v in logits[i].tolist()])
print('softmax 概率   ：', [f'{v:.3f}' for v in probs[i].tolist()])
print('得分大的类别概率高；概率最大的那一类就是预测结果')

# %% 6. 保存模型（以后加载就不用重新训练）
torch.save(model.state_dict(), 'data/mlp_mnist.pt')
print('已保存到 data/mlp_mnist.pt；加载方法：model = MLP(); model.load_state_dict(torch.load("data/mlp_mnist.pt", weights_only=True))')

# %% 导出给网站（学习时可跳过）
torch.manual_seed(1)
demo_idx = torch.randperm(10000)[:24]                     # 网页里「随机抽一张测试图」用
site_export = {
    'model': model,
    'golden': (X_test[:8], lambda x: model(x)),
    'history': history,
    'summary': {
        'test_acc': f'{history["test_acc"][-1]:.2%}', 'train_acc': f'{history["train_acc"][-1]:.2%}',
        'params': f'{n_params:,}', 'n_wrong': len(wrong), 'secs': f'{train_secs:.0f}',
        'top_confusion': f'{pairs[0][1]} 被认成 {pairs[0][2]}',
    },
    'extras': {
        'demo_x': (X_test[demo_idx, 0] * 255).round().to(torch.uint8), 'demo_y': y_test[demo_idx].tolist(),
        'wrong_x': (X_test[wrong[:16], 0] * 255).round().to(torch.uint8),
        'wrong_y': y_test[wrong[:16]].tolist(), 'wrong_p': pred[wrong[:16]].tolist(),
        'per_class': [round((pred[y_test == d] == d).float().mean().item(), 4) for d in range(10)],
    },
}
