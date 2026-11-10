"""第 4 课 · 计算图与反向传播
经典案例：一个神经元的梯度，手算 vs PyTorch autograd —— 完全一致；再推广到两层网络（矩阵形式）。

一个神经元 + 平方损失：L = (σ(w·x + b) − y)²
前向：u = w·x → z = u + b → a = σ(z) → e = a − y → L = e²
反向：从 dL/dL = 1 出发，每个节点把收到的梯度乘上自己的「局部导数」再往回传（链式法则）。

运行：python 04_backprop.py（几秒钟）
"""

# %% 1. 前向传播：一步一步算，并记下每个中间结果
import torch

x, w, b, y = 1.5, 0.8, -0.5, 1.0           # 输入、权重、偏置、真实标签
u = w * x                                   # 乘法节点
z = u + b                                   # 加法节点
a = 1 / (1 + torch.exp(torch.tensor(-z)).item())   # sigmoid 节点：把 z 压到 0~1 之间
e = a - y                                   # 误差
L = e ** 2                                  # 平方损失
print(f'u = w·x = {u:.4f}\nz = u + b = {z:.4f}\na = σ(z) = {a:.4f}\ne = a − y = {e:.4f}\nL = e² = {L:.4f}')

# %% 2. 反向传播：从损失出发，链式法则逐个节点往回乘
dL_dL = 1.0
dL_de = dL_dL * 2 * e                       # L = e²      → 局部导数 2e
dL_da = dL_de * 1                           # e = a − y   → 对 a 的局部导数是 1
dL_dz = dL_da * a * (1 - a)                 # a = σ(z)    → σ'(z) = σ(z)·(1 − σ(z))
dL_db = dL_dz * 1                           # z = u + b   → 加法把梯度原样分给两个输入
dL_du = dL_dz * 1
dL_dw = dL_du * x                           # u = w·x     → 乘法：对 w 的导数是 x，对 x 的导数是 w
print(f'∂L/∂e = {dL_de:.4f}\n∂L/∂a = {dL_da:.4f}\n∂L/∂z = {dL_dz:.4f}   ← 乘上了 σ\'(z) = {a * (1 - a):.4f}')
print(f'∂L/∂b = {dL_db:.4f}\n∂L/∂w = {dL_dw:.4f}   ← 再乘上 x = {x}')

# %% 3. 用 autograd 验证：PyTorch 自动做了上面全部的事
wt = torch.tensor(w, requires_grad=True)
bt = torch.tensor(b, requires_grad=True)
Lt = (torch.sigmoid(wt * x + bt) - y) ** 2
Lt.backward()
print(f'autograd：∂L/∂w = {wt.grad.item():.4f}，∂L/∂b = {bt.grad.item():.4f}')
print(f'手算    ：∂L/∂w = {dL_dw:.4f}，∂L/∂b = {dL_db:.4f}  → 一致 ✓')
print('梯度是负的 → 把 w、b 调大一点，输出 a 会更接近目标 1，损失会变小')

# %% 4. autograd 记下的计算图：每个结果都知道自己是怎么来的
print('Lt.grad_fn 是', type(Lt.grad_fn).__name__, '← 最后一步是平方（Pow）')
node, depth = Lt.grad_fn, 0
while node is not None and depth < 6:            # 沿着 next_functions 往回走，就是反向传播的路线
    print('  ' * depth + '↳', type(node).__name__)
    nxt = [f for f, _ in node.next_functions if f is not None]
    node, depth = (nxt[0] if nxt else None), depth + 1

# %% 5. 推广到两层网络：矩阵形式的反向传播
torch.manual_seed(0)
X = torch.randn(8, 3)                     # 8 个样本，每个 3 个特征
Y = torch.randn(8, 1)
W1, b1 = torch.randn(3, 4) * 0.5, torch.zeros(4)
W2, b2 = torch.randn(4, 1) * 0.5, torch.zeros(1)

# 前向
Z1 = X @ W1 + b1                          # [8, 4]
H = torch.sigmoid(Z1)                     # [8, 4]
P = H @ W2 + b2                           # [8, 1]
loss = ((P - Y) ** 2).mean()

# 手写反向：每个梯度的形状都和对应参数一样（这是检查公式对不对的好办法）
dP = 2 * (P - Y) / len(X)                 # [8, 1]  均方误差对 P 的导数
dW2 = H.T @ dP                            # [4, 1]  线性层：dW = 输入ᵀ @ 上游梯度
db2 = dP.sum(0)                           # [1]     偏置：上游梯度按样本求和
dH = dP @ W2.T                            # [8, 4]  传给上一层：上游梯度 @ Wᵀ
dZ1 = dH * H * (1 - H)                    # [8, 4]  逐元素乘 sigmoid 的导数
dW1 = X.T @ dZ1                           # [3, 4]
db1 = dZ1.sum(0)                          # [4]

# 用 autograd 再算一遍对照
params = [t.clone().requires_grad_(True) for t in (W1, b1, W2, b2)]
pW1, pb1, pW2, pb2 = params
loss2 = ((torch.sigmoid(X @ pW1 + pb1) @ pW2 + pb2 - Y) ** 2).mean()
loss2.backward()
diff = max((m - p.grad).abs().max().item() for m, p in zip((dW1, db1, dW2, db2), params))
print(f'损失 {loss.item():.4f}；手写反向传播与 autograd 的最大差异 = {diff:.2e}（只是浮点舍入误差）')

# %% 6. 数值梯度检验：用「挪一点点看损失变多少」验证反向传播
def loss_of(W1_):
    return ((torch.sigmoid(X @ W1_ + b1) @ W2 + b2 - Y) ** 2).mean().item()

eps, worst = 1e-3, 0.0
X, Y, W1 = X.double(), Y.double(), W1.double()    # 用双精度，差分更准
b1, W2, b2, dW1 = b1.double(), W2.double(), b2.double(), dW1.double()
for i in range(3):
    for j in range(4):
        Wp, Wm = W1.clone(), W1.clone()
        Wp[i, j] += eps
        Wm[i, j] -= eps
        num = (loss_of(Wp) - loss_of(Wm)) / (2 * eps)          # 中心差分
        rel = abs(num - dW1[i, j].item()) / max(1e-12, abs(num) + abs(dW1[i, j].item()))
        worst = max(worst, rel)
print(f'W1 全部 12 个梯度：数值差分 vs 反向传播，最大相对误差 {worst:.2e}（小于 1e-6 就说明反向传播写对了）')

# %% 7. 一个常见的坑：梯度会累加
wt = torch.tensor(0.8, requires_grad=True)
for i in range(3):
    Lt = (torch.sigmoid(wt * x + bt.detach()) - y) ** 2
    Lt.backward()                             # 没有清零就再次 backward
    print(f'第 {i + 1} 次 backward 后 w.grad = {wt.grad.item():.4f}')
print('梯度在一直累加！所以训练循环里每一步都要 optimizer.zero_grad()。')

# %% 导出给网站（学习时可跳过）
site_export = {
    'summary': {
        'L': f'{L:.4f}', 'dw': f'{dL_dw:.4f}', 'db': f'{dL_db:.4f}',
        'diff': f'{diff:.1e}', 'rel': f'{worst:.1e}',
    },
}
