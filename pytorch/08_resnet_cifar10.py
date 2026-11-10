"""第 8 课 · 经典 CNN 与残差网络（ResNet）
经典案例：CIFAR-10 图像分类——ResNet-20（He et al., 2016 论文里给 CIFAR-10 设计的版本）。

另外做一个小实验复现论文的核心发现：普通网络加深反而更差，加上残差连接就不会。

运行：python 08_resnet_cifar10.py（CPU 上约 35~45 分钟；首次运行会自动下载 CIFAR-10，约 170 MB）
"""

# %% 1. 读取 CIFAR-10：10 类、32×32 的彩色小图
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

torch.manual_seed(0)
CLASSES = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
train_raw = datasets.CIFAR10('data', train=True, download=True)
test_raw = datasets.CIFAR10('data', train=False, download=True)
# .data 是 uint8 的 [N, 32, 32, 3]（HWC）；PyTorch 卷积要 [N, 3, 32, 32]（CHW），所以 permute
X_train = torch.from_numpy(train_raw.data).permute(0, 3, 1, 2).float().div(255)
X_test = torch.from_numpy(test_raw.data).permute(0, 3, 1, 2).float().div(255)
y_train, y_test = torch.tensor(train_raw.targets), torch.tensor(test_raw.targets)
print('训练集', tuple(X_train.shape), '测试集', tuple(X_test.shape))
print('类别：', '、'.join(CLASSES))

# %% 2. 数据增强：随机裁剪 + 水平翻转
def augment(x):
    """每张图：四周补 4 像素后随机裁回 32×32（相当于随机平移），再以一半概率左右翻转。
    猫向左还是向右、往哪边挪几个像素，都还是猫——让模型学到这种不变性。"""
    n = len(x)
    flip = torch.rand(n) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    padded = F.pad(x, (4, 4, 4, 4), mode='reflect')
    out = torch.empty_like(x)
    for k, (i, j) in enumerate(torch.randint(0, 9, (n, 2)).tolist()):
        out[k] = padded[k, :, i:i + 32, j:j + 32]
    return out

# %% 3. 残差块：输出 = F(x) + x
class Block(nn.Module):
    def __init__(self, cin, cout, stride, residual=True):
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)   # 有 BN 时卷积不需要偏置
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        # 形状变化时（步幅 2 或通道数变了），捷径用 1×1 卷积把 x 变成同样的形状才能相加
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out = out + self.short(x)            # ← 残差连接：只要 F(x) 学成 0，这一块就是「什么也不做」
        return torch.relu(out)

class ResNet(nn.Module):
    """CIFAR 版 ResNet：3 个阶段（16、32、64 个通道），每阶段 n 个残差块，总层数 6n + 2。"""
    MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

    def __init__(self, n=3, residual=True):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        blocks, cin = [], 16
        for cout, stride in [(16, 1), (32, 2), (64, 2)]:
            for i in range(n):
                blocks.append(Block(cin, cout, stride if i == 0 else 1, residual))
                cin = cout
        self.layers = nn.Sequential(*blocks)
        self.fc = nn.Linear(64, 10)
        for m in self.modules():                   # He 初始化（第 6 课）
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = (x - self.MEAN) / self.STD             # 按通道标准化，写在模型里
        x = torch.relu(self.bn(self.conv(x)))      # [N, 16, 32, 32]
        x = self.layers(x)                         # → [N, 64, 8, 8]
        x = x.mean(dim=(2, 3))                     # 全局平均池化 → [N, 64]
        return self.fc(x)                          # → [N, 10]

model = ResNet(n=3)
n_params = sum(p.numel() for p in model.parameters())
print(f'ResNet-20：{sum(isinstance(m, nn.Conv2d) for m in model.modules()) - 2} 个 3×3 卷积层 + 1 个全连接层，参数 {n_params:,}')

def evaluate(model, X, y, bs=1000):
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(X[i:i + bs]).argmax(1) for i in range(0, len(X), bs)])
    model.train()
    return (pred == y).float().mean().item(), pred

# %% 4. 小实验：普通网络 vs 残差网络，20 层 vs 56 层（各用 1 万张图训练 4 轮）
def short_run(n, residual, epochs=4, n_train=10000):
    torch.manual_seed(0)
    net = ResNet(n, residual)
    opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    curve, running = [], []
    for _ in range(epochs):
        for idx in torch.randperm(n_train).split(128):
            loss = F.cross_entropy(net(augment(X_train[idx])), y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            running.append(loss.item())
            if len(running) == 10:
                curve.append(sum(running) / 10)
                running = []
    acc, _ = evaluate(net, X_test[:2000], y_test[:2000])
    return curve, acc

t0 = time.time()
deg = {}
for n, residual, name in [(3, False, '普通网络 20 层'), (9, False, '普通网络 56 层'), (3, True, '残差网络 20 层'), (9, True, '残差网络 56 层')]:
    deg[name] = short_run(n, residual)
    print(f'{name}：最后的训练损失 {deg[name][0][-1]:.3f}，测试准确率 {deg[name][1]:.1%}', flush=True)
print(f'用时 {time.time() - t0:.0f} 秒。普通网络从 20 层加到 56 层，连训练损失都更高了（不是过拟合，是更难优化）；')
print('加上残差连接，56 层网络的训练损失大幅下降。只训练 4 轮时更深的网络还没发挥出优势，')
print('论文里训练足够久之后，ResNet-56 的误差会反超 ResNet-20——深度终于变成了好处。')

# %% 5. 各层梯度的大小：普通网络 vs 残差网络（56 层，刚初始化时，用一批真实图片）
def grad_norms(residual):
    torch.manual_seed(0)
    net = ResNet(9, residual)
    loss = F.cross_entropy(net(X_train[:128]), y_train[:128])
    loss.backward()
    return [m.weight.grad.norm().item() for name, m in net.named_modules() if isinstance(m, nn.Conv2d) and 'short' not in name]

g_plain, g_res = grad_norms(False), grad_norms(True)
for name, g in [('普通网络', g_plain), ('残差网络', g_res)]:
    print(f'{name}：第 1 层卷积的梯度 {g[0]:.2e}，最后一层卷积 {g[-1]:.2e}，最大 / 最小 = {max(g) / min(g):.0f} 倍')

# %% 6. 正式训练 ResNet-20：SGD + Momentum + OneCycle 学习率 + 权重衰减 + 数据增强
EPOCHS, BS = 30, 128
torch.manual_seed(0)
model = ResNet(n=3)
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
steps = (len(X_train) + BS - 1) // BS
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, epochs=EPOCHS, steps_per_epoch=steps)   # 先升后降的学习率
history = {'train_loss': [], 'test_acc': [], 'lr': []}
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    total, n = 0.0, 0
    for idx in torch.randperm(len(X_train)).split(BS):
        loss = F.cross_entropy(model(augment(X_train[idx])), y_train[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()                               # 每一步都更新学习率
        total += loss.item() * len(idx)
        n += len(idx)
    acc, _ = evaluate(model, X_test, y_test)
    history['train_loss'].append(total / n)
    history['test_acc'].append(acc)
    history['lr'].append(sched.get_last_lr()[0])
    print(f'第 {epoch:2d} 轮  训练损失 {total / n:.3f}  测试准确率 {acc:.2%}  （已用 {(time.time() - t0) / 60:.1f} 分钟）', flush=True)
train_min = (time.time() - t0) / 60

# %% 7. 看看哪些类别最难
acc, pred = evaluate(model, X_test, y_test)
per_class = [(pred[y_test == c] == c).float().mean().item() for c in range(10)]
for c in range(10):
    print(f'{CLASSES[c]}：{per_class[c]:.1%}', end='   ' if c % 5 < 4 else '\n')
conf = torch.zeros(10, 10, dtype=torch.long)
for t, p in zip(y_test.tolist(), pred.tolist()):
    conf[t, p] += 1
pairs = sorted(((conf[a, b].item(), a, b) for a in range(10) for b in range(10) if a != b), reverse=True)[:4]
print('最常见的错误：', '；'.join(f'{CLASSES[a]}→{CLASSES[b]}（{c} 次）' for c, a, b in pairs))

# %% 导出给网站（学习时可跳过）
demo = torch.cat([(y_test == c).nonzero()[:3, 0] for c in range(10)])    # 每类 3 张测试图
site_export = {
    'model': model,
    'golden': (X_test[:4], lambda x: model(x)),
    'history': history,
    'summary': {
        'test_acc': f'{acc:.2%}', 'params': f'{n_params:,}', 'epochs': EPOCHS, 'minutes': f'{train_min:.0f}',
        'worst': CLASSES[min(range(10), key=lambda c: per_class[c])], 'best': CLASSES[max(range(10), key=lambda c: per_class[c])],
        'top_confusion': f'{CLASSES[pairs[0][1]]}被认成{CLASSES[pairs[0][2]]}',
        'p20': f'{deg["普通网络 20 层"][0][-1]:.3f}', 'p56': f'{deg["普通网络 56 层"][0][-1]:.3f}',
        'r20': f'{deg["残差网络 20 层"][0][-1]:.3f}', 'r56': f'{deg["残差网络 56 层"][0][-1]:.3f}',
        'gp_ratio': f'{max(g_plain) / min(g_plain):.0f}', 'gr_ratio': f'{max(g_res) / min(g_res):.0f}',
    },
    'extras': {
        'classes': CLASSES,
        'demo_x': torch.from_numpy(test_raw.data[demo.numpy()]),           # [30, 32, 32, 3] uint8
        'demo_y': y_test[demo].tolist(),
        'per_class': [round(v, 4) for v in per_class],
        'deg': [v[0] for v in deg.values()], 'deg_names': list(deg.keys()),
        'grad_plain': g_plain, 'grad_res': g_res,
    },
}
