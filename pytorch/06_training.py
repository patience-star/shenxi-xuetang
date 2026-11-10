"""第 6 课 · 训练技巧：优化器、初始化、正则化、BatchNorm
经典案例：在 MNIST 上做 4 组对照实验——每组只改变一个因素，其余完全相同，看它到底起什么作用。

运行：python 06_training.py（CPU 上约 2~3 分钟）
"""

# %% 1. 数据与通用工具
import time

import torch
import torch.nn as nn
from torchvision import datasets

train_raw = datasets.MNIST('data', train=True, download=True)
test_raw = datasets.MNIST('data', train=False, download=True)
X_train = ((train_raw.data.float() / 255 - 0.1307) / 0.3081).flatten(1)   # 直接标准化并展平成 [N, 784]
X_test = ((test_raw.data.float() / 255 - 0.1307) / 0.3081).flatten(1)
y_train, y_test = train_raw.targets, test_raw.targets

def mlp(sizes, act=nn.ReLU, bn=False, dropout=0.0):
    """按 sizes 搭一个全连接网络，例如 [784, 128, 64, 10]；可选 BatchNorm、Dropout。"""
    layers = []
    for i in range(len(sizes) - 2):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if bn:
            layers.append(nn.BatchNorm1d(sizes[i + 1]))   # 在激活函数之前做批归一化
        layers.append(act())
        if dropout:
            layers.append(nn.Dropout(dropout))            # 训练时随机把一部分神经元的输出置 0
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)

def accuracy(model, X, y):
    model.eval()                                          # 关键：评估时关掉 Dropout、BN 用训练时的统计量
    with torch.no_grad():
        acc = (model(X).argmax(1) == y).float().mean().item()
    model.train()
    return acc

def batches(n, size, gen):
    idx = torch.randperm(n, generator=gen)
    return [idx[i:i + size] for i in range(0, n, size)]

loss_fn = nn.CrossEntropyLoss()
print('数据准备好了：训练集', tuple(X_train.shape), '测试集', tuple(X_test.shape))

# %% 2. 实验一：优化器——同一个网络、同一个初始值，只换优化器
def run_optimizer(name, make_opt, epochs=2):
    torch.manual_seed(0)                                  # 三次实验的初始权重、数据顺序完全相同
    model = mlp([784, 128, 64, 10])
    opt = make_opt(model.parameters())
    gen, curve, running = torch.Generator().manual_seed(0), [], []
    for _ in range(epochs):
        for idx in batches(60000, 128, gen):
            loss = loss_fn(model(X_train[idx]), y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            running.append(loss.item())
            if len(running) == 20:                        # 每 20 步记一次平均损失，曲线更平滑
                curve.append(sum(running) / 20)
                running = []
    acc = accuracy(model, X_test, y_test)
    print(f'{name:16s} 最后的损失 {curve[-1]:.4f}   测试准确率 {acc:.2%}')
    return curve, acc

opt_curves, opt_acc = {}, {}
for name, make in [('SGD', lambda p: torch.optim.SGD(p, lr=0.01)),
                   ('SGD + Momentum', lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9)),
                   ('Adam', lambda p: torch.optim.Adam(p, lr=1e-3))]:
    opt_curves[name], opt_acc[name] = run_optimizer(name, make)
print('同样的学习步数，Momentum 和 Adam 下降得快得多：它们会「记住」之前的梯度方向，少走弯路。')

# %% 3. 实验二：过拟合与正则化——只用 1000 张训练图，网络却很大
def random_shift(x, max_shift=2):
    """数据增强：每张图随机平移最多 2 个像素（数字挪一挪还是同一个数字，相当于免费多出很多样本）"""
    imgs = x.view(-1, 1, 28, 28)
    padded = nn.functional.pad(imgs, (max_shift,) * 4, value=imgs.min().item())   # 四周用背景值填充
    out = torch.empty_like(imgs)
    for i, (dy, dx) in enumerate(torch.randint(0, 2 * max_shift + 1, (len(imgs), 2)).tolist()):
        out[i] = padded[i, :, dy:dy + 28, dx:dx + 28]
    return out.flatten(1)

def test_loss(model):
    model.eval()
    with torch.no_grad():
        loss = loss_fn(model(X_test), y_test).item()
    model.train()
    return loss

def run_regularization(name, n_train, dropout=0.0, weight_decay=0.0, augment=False, epochs=60):
    torch.manual_seed(0)
    model = mlp([784, 512, 512, 10], dropout=dropout)    # 66 万个参数 vs 1000 个样本：很容易死记硬背
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    gen, tr, te, tl = torch.Generator().manual_seed(0), [], [], []
    Xs, ys = X_train[:n_train], y_train[:n_train]
    for epoch in range(epochs):
        for idx in batches(n_train, 100, gen):
            xb = random_shift(Xs[idx]) if augment else Xs[idx]
            loss = loss_fn(model(xb), ys[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 3 == 2:                                 # 每 3 轮记录一次
            tr.append(accuracy(model, Xs, ys))
            te.append(accuracy(model, X_test, y_test))
            tl.append(test_loss(model))
    print(f'{name:24s} 训练准确率 {tr[-1]:.2%}  测试准确率 {te[-1]:.2%}  测试损失 {tl[-1]:.3f}（最低时 {min(tl):.3f}）')
    return tr, te, tl

t0 = time.time()
reg = {}
reg['无正则化'] = run_regularization('无正则化', 1000)
reg['Dropout 0.5'] = run_regularization('Dropout 0.5', 1000, dropout=0.5)
reg['权重衰减 0.01'] = run_regularization('权重衰减 0.01', 1000, weight_decay=0.01)
reg['数据增强'] = run_regularization('数据增强（随机平移）', 1000, augment=True)
reg['更多数据'] = run_regularization('更多数据（10000 张）', 10000)
print(f'用时 {time.time() - t0:.0f} 秒。')
print('训练准确率都到了 100%，测试准确率却低得多——这就是过拟合（模型把 1000 张图「背」下来了）。')
print('无正则化时测试损失先降后升：模型对答错的题越来越「自信」。Dropout、权重衰减能压住这种过度自信；')
print('数据增强和更多数据才真正提高了测试准确率：最有效的正则化永远是更多（更多样）的数据。')

# %% 4. 实验三：BatchNorm——4 个隐藏层的 sigmoid 网络，有没有 BN
def run_bn(bn):
    torch.manual_seed(0)
    model = mlp([784, 100, 100, 100, 100, 10], act=nn.Sigmoid, bn=bn)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    gen, curve, running = torch.Generator().manual_seed(0), [], []
    for idx in batches(60000, 64, gen):                    # 只训练 1 个 epoch
        loss = loss_fn(model(X_train[idx]), y_train[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        running.append(loss.item())
        if len(running) == 20:
            curve.append(sum(running) / 20)
            running = []
    acc = accuracy(model, X_test, y_test)
    print(f'{"有 BatchNorm" if bn else "没有 BatchNorm":14s} 1 个 epoch 后测试准确率 {acc:.2%}')
    return curve, acc

bn_curve, bn_acc = run_bn(True)
plain_curve, plain_acc = run_bn(False)
print('深层 sigmoid 网络本来很难训练；BN 让每层的输入保持在合适的范围，梯度能顺畅地传下去。')

# %% 5. 实验四：初始化——10 层网络里，每一层输出的「幅度」怎么变化
def layer_stds(init, act):
    torch.manual_seed(0)
    x = torch.randn(512, 256)                             # 一批随机输入
    stds = []
    for _ in range(10):
        W = torch.empty(256, 256)
        if init == 'small':
            nn.init.normal_(W, std=0.01)                  # 太小
        elif init == 'large':
            nn.init.normal_(W, std=1.0)                   # 太大
        elif init == 'xavier':
            nn.init.xavier_normal_(W)                     # 方差 = 2 / (输入数 + 输出数)，适合 tanh
        else:
            nn.init.kaiming_normal_(W, nonlinearity='relu')   # 方差 = 2 / 输入数，适合 ReLU
        x = act(x @ W)
        stds.append(x.std().item())
    return stds

inits = {
    'tanh + 太小 N(0, 0.01²)': layer_stds('small', torch.tanh),
    'tanh + 太大 N(0, 1)': layer_stds('large', torch.tanh),
    'tanh + Xavier': layer_stds('xavier', torch.tanh),
    'ReLU + Xavier': layer_stds('xavier', torch.relu),
    'ReLU + He（Kaiming）': layer_stds('kaiming', torch.relu),
}
for name, s in inits.items():
    print(f'{name:22s} 各层输出标准差：', ' '.join(f'{v:.3f}' for v in s))
print('太小：信号一层层衰减到 0（梯度也会消失）；太大：tanh 全部饱和在 ±1（梯度同样消失）；')
print('Xavier / He 初始化让每层的输出幅度保持稳定。PyTorch 的 nn.Linear 默认用的也是这一类初始化。')

# %% 导出给网站（学习时可跳过）
site_export = {
    'history': {
        'opt': list(opt_curves.values()), 'opt_names': list(opt_curves.keys()),
        'reg_train': [v[0] for v in reg.values()], 'reg_test': [v[1] for v in reg.values()], 'reg_loss': [v[2] for v in reg.values()],
        'reg_names': list(reg.keys()),
        'bn': bn_curve, 'plain': plain_curve,
        'init': list(inits.values()), 'init_names': list(inits.keys()),
    },
    'summary': {
        'sgd_acc': f'{opt_acc["SGD"]:.2%}', 'mom_acc': f'{opt_acc["SGD + Momentum"]:.2%}', 'adam_acc': f'{opt_acc["Adam"]:.2%}',
        'none_train': f'{reg["无正则化"][0][-1]:.1%}', 'none_test': f'{reg["无正则化"][1][-1]:.1%}',
        'drop_test': f'{reg["Dropout 0.5"][1][-1]:.1%}', 'wd_test': f'{reg["权重衰减 0.01"][1][-1]:.1%}',
        'aug_test': f'{reg["数据增强"][1][-1]:.1%}', 'more_test': f'{reg["更多数据"][1][-1]:.1%}',
        'none_loss': f'{reg["无正则化"][2][-1]:.3f}', 'none_loss_min': f'{min(reg["无正则化"][2]):.3f}',
        'drop_loss': f'{reg["Dropout 0.5"][2][-1]:.3f}', 'wd_loss': f'{reg["权重衰减 0.01"][2][-1]:.3f}', 'aug_loss': f'{reg["数据增强"][2][-1]:.3f}',
        'bn_acc': f'{bn_acc:.2%}', 'plain_acc': f'{plain_acc:.2%}',
    },
}
