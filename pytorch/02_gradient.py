"""第 2 课 · 导数、梯度与梯度下降
经典案例：一个神经元学会「摄氏度 → 华氏度」。

真实公式是 F = 1.8 × C + 32，但我们不告诉模型，只给它一些 (摄氏, 华氏) 的测量数据，
看它能不能靠梯度下降自己「找到」 1.8 和 32。顺便看清两个关键细节：
  ① 输入的数值范围差别大时（这里摄氏度从 −40 到 100），训练会变得很慢；标准化之后立刻变快；
  ② 学习率太小走得慢，太大会来回震荡甚至发散。

运行：python 02_gradient.py（几秒钟）
"""

# %% 1. 准备数据：40 个带测量误差的温度
import torch

torch.manual_seed(0)
C = torch.linspace(-40, 100, 40)                  # 摄氏度：-40 ~ 100 之间均匀取 40 个
F = 1.8 * C + 32 + torch.randn(40) * 2.0          # 华氏度 = 真实公式 + 一点测量误差（温度计不完美）
for c, f in list(zip(C, F))[::8]:
    print(f'{c.item():7.1f} °C  →  {f.item():7.1f} °F')

# %% 2. 先感受一下导数：autograd 自动求导 vs 手算
# 设模型只有一个参数 w：预测 = w × c，单个样本的损失 L = (w·c − f)²
w = torch.tensor(1.0, requires_grad=True)          # requires_grad=True：让 PyTorch 记录和它有关的计算
c, f = torch.tensor(10.0), torch.tensor(50.0)
L = (w * c - f) ** 2
L.backward()                                        # 自动求 dL/dw，结果存到 w.grad
print('autograd 算出的 dL/dw =', w.grad.item())
print('手算 2·c·(w·c − f)    =', (2 * c * (w * c - f)).item())   # 链式法则：外层 2(·) 乘内层导数 c
print('导数是负的 → 增大 w 能让损失变小（10 × w 离 50 还差得远）')

# %% 3. 手写梯度下降：直接用原始摄氏度训练 w 和 b
def train(x, y, lr, steps, log_every):
    """最朴素的梯度下降：每一步用全部 40 个样本算损失、求梯度、更新参数。"""
    w = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    losses = []
    for step in range(steps + 1):
        pred = w * x + b                            # 一个神经元：加权求和（没有激活函数）
        loss = ((pred - y) ** 2).mean()             # 均方误差（MSE）：误差平方的平均
        losses.append(loss.item())
        if log_every and step % log_every == 0:
            print(f'第 {step:5d} 步  loss = {loss.item():10.3f}   w = {w.item():.4f}   b = {b.item():.4f}')
        loss.backward()                             # 求 ∂loss/∂w 和 ∂loss/∂b
        with torch.no_grad():                       # 更新参数这一步不需要被记进计算图
            w -= lr * w.grad                        # 核心公式：参数 ← 参数 − 学习率 × 梯度
            b -= lr * b.grad
            w.grad.zero_()                          # PyTorch 会把梯度累加，所以用完要清零
            b.grad.zero_()
    return w.item(), b.item(), losses

# 摄氏度最大到 100，w 的梯度会比 b 的梯度大上百倍，学习率只能取得很小，否则发散
w_raw, b_raw, loss_raw = train(C, F, lr=2e-4, steps=3000, log_every=500)
print(f'\n原始输入训练 3000 步：w = {w_raw:.3f}（真值 1.8），b = {b_raw:.3f}（真值 32）← b 还差得远！')

# %% 4. 标准化输入：同样的问题，几十步就学会
mu, sigma = C.mean(), C.std()
Cn = (C - mu) / sigma                               # 标准化：均值 0、标准差 1
wn, bn, loss_norm = train(Cn, F, lr=0.1, steps=100, log_every=20)
# 换算回原来的单位：F = wn·(C − μ)/σ + bn = (wn/σ)·C + (bn − wn·μ/σ)
w_back, b_back = wn / sigma.item(), bn - wn * mu.item() / sigma.item()
print(f'\n标准化后只训练 100 步：换算回来 w = {w_back:.3f}，b = {b_back:.3f}')
print('原因：标准化后 w 和 b 的梯度大小相当，一个学习率对两个参数都合适。')

# %% 5. PyTorch 的标准写法：nn.Linear + 优化器
import torch.nn as nn

model = nn.Linear(1, 1)                             # 一个神经元：1 个输入、1 个输出，内含 w 和 b
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()
X, Y = Cn.unsqueeze(1), F.unsqueeze(1)              # nn.Linear 要求输入形状 [样本数, 特征数]
for epoch in range(100):
    optimizer.zero_grad()                           # ① 清空上一步的梯度
    loss = loss_fn(model(X), Y)                     # ② 前向：算预测和损失
    loss.backward()                                 # ③ 反向：算梯度
    optimizer.step()                                # ④ 更新：参数 ← 参数 − lr × 梯度
print(f'训练后 loss = {loss.item():.3f}')
with torch.no_grad():
    t = torch.tensor([[37.0]])
    print(f'体温 37 °C → 预测 {model((t - mu) / sigma).item():.1f} °F（真实公式给出 {1.8 * 37 + 32:.1f} °F）')

# %% 6. 学习率实验：同样 30 步，不同学习率
lr_curves = {}
for lr in [0.01, 0.1, 0.5, 0.95, 1.05]:
    *_, ls = train(Cn, F, lr=lr, steps=30, log_every=0)   # log_every=0：不打印过程
    lr_curves[lr] = ls
    trend = '发散 ✗' if ls[-1] > ls[0] else '收敛 ✓' if ls[-1] < 5 else '还没到谷底'
    print(f'学习率 {lr:<5}：第 0 步 loss {ls[0]:8.1f} → 第 30 步 {ls[-1]:12.2f}  {trend}')

# %% 导出给网站（学习时可跳过）
site_export = {
    'history': {
        'raw': loss_raw[::30],                      # 原始输入的损失（每 30 步取一个点）
        'norm': loss_norm,
    },
    'summary': {
        'w_raw': f'{w_raw:.3f}', 'b_raw': f'{b_raw:.2f}',
        'w_norm': f'{w_back:.3f}', 'b_norm': f'{b_back:.2f}',
        'pred37': f'{model((torch.tensor([[37.0]]) - mu) / sigma).item():.1f}',
    },
    'extras': {
        'C': C.tolist(), 'F': F.tolist(),
        'lrs': list(lr_curves.keys()),
        'lr_curves': list(lr_curves.values()),      # 与 lrs 一一对应
    },
}
