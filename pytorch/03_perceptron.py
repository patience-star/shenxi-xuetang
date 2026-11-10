"""第 3 课 · 感知机
经典案例：感知机学逻辑门——「与」「或」学得会，「异或」学不会；叠成两层就能解决。

感知机（Rosenblatt，1958）：输出 = 1 如果 w·x + b > 0，否则 0。
学习规则：预测错了才改，w ← w + η·(真实 − 预测)·x，b ← b + η·(真实 − 预测)。

运行：python 03_perceptron.py（几秒钟）
"""

# %% 1. 数据：两个输入的四种组合，以及三种逻辑门的真值表
import torch

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
gates = {
    '与 AND': torch.tensor([0., 0., 0., 1.]),   # 两个都是 1 才输出 1
    '或 OR':  torch.tensor([0., 1., 1., 1.]),   # 有一个是 1 就输出 1
    '异或 XOR': torch.tensor([0., 1., 1., 0.]),  # 两个不一样才输出 1
}
for name, y in gates.items():
    print(f'{name:8s}', [f'{int(a)}{int(b)}→{int(t)}' for (a, b), t in zip(X.tolist(), y.tolist())])

# %% 2. 手写一个感知机，用感知机学习规则训练
def predict(x, w, b):
    return (x @ w + b > 0).float()                # 阶跃函数：加权和大于 0 输出 1，否则 0

def train_perceptron(X, y, lr=0.1, max_epochs=20):
    w, b = torch.zeros(2), torch.tensor(0.)
    errors_per_epoch = []
    for epoch in range(1, max_epochs + 1):
        errors = 0
        for xi, ti in zip(X, y):                  # 一个样本一个样本地看
            err = ti - predict(xi, w, b)          # 对了是 0；该出 1 却出 0 是 +1；反过来是 −1
            if err != 0:
                w = w + lr * err * xi             # 把分界线往「让这个样本被分对」的方向推一下
                b = b + lr * err
                errors += 1
        errors_per_epoch.append(errors)
        if errors == 0:                            # 一整轮都没错：收敛了
            break
    return w, b, errors_per_epoch

results = {}
for name, y in gates.items():
    w, b, errs = train_perceptron(X, y)
    results[name] = (w, b, errs)
    done = errs[-1] == 0
    print(f'{name:8s} 每轮出错次数 {errs}')
    print(f'{"":8s} {"✓ 第 " + str(len(errs)) + " 轮收敛" if done else "✗ 20 轮后仍然出错"}：w = {[round(v, 2) for v in w.tolist()]}, b = {b.item():.1f}，'
          f'预测 {predict(X, w, b).int().tolist()} / 真实 {y.int().tolist()}')

# %% 3. 学到的分界线：w1·x1 + w2·x2 + b = 0
for name in ['与 AND', '或 OR']:
    w, b, _ = results[name]
    # 把直线写成 x2 = −(w1·x1 + b) / w2 的形式，方便想象它在平面上的位置
    print(f'{name}：{w[0].item():.1f}·x1 + {w[1].item():.1f}·x2 + ({b.item():.1f}) = 0  →  x2 = {-w[0].item() / w[1].item():.2f}·x1 + {-b.item() / w[1].item():.2f}')
print('异或：(0,1)、(1,0) 要在一边，(0,0)、(1,1) 要在另一边——平面上找不到这样的一条直线。')

# %% 4. 两层感知机解决异或：异或 = (x1 或 x2) 且 不是(x1 且 x2)
w_or, b_or = torch.tensor([1., 1.]), torch.tensor(-0.5)      # 或：x1 + x2 − 0.5 > 0
w_nand, b_nand = torch.tensor([-1., -1.]), torch.tensor(1.5)  # 与非：−x1 − x2 + 1.5 > 0
w_and, b_and = torch.tensor([1., 1.]), torch.tensor(-1.5)    # 与：h1 + h2 − 1.5 > 0
h = torch.stack([predict(X, w_or, b_or), predict(X, w_nand, b_nand)], dim=1)   # 第一层：两个感知机
out = predict(h, w_and, b_and)                                               # 第二层：一个感知机
for (x1, x2), (h1, h2), o in zip(X.int().tolist(), h.int().tolist(), out.int().tolist()):
    print(f'输入 ({x1},{x2}) → 隐藏层 (或={h1}, 与非={h2}) → 输出 {o}')
print('两层就能画出「两条线夹出的带子」，异或被解决了。但这些权重是人想出来的——')

# %% 5. 让网络自己学：把阶跃函数换成平滑的 sigmoid，用梯度下降训练两层网络
import torch.nn as nn

torch.manual_seed(0)
net = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1), nn.Sigmoid())
opt = torch.optim.Adam(net.parameters(), lr=0.05)
y = gates['异或 XOR'].unsqueeze(1)
for step in range(1001):
    loss = nn.functional.binary_cross_entropy(net(X), y)   # 二分类交叉熵（第 5 课详讲）
    opt.zero_grad()
    loss.backward()                                         # 反向传播：下一课的主题
    opt.step()
    if step % 250 == 0:
        print(f'第 {step:4d} 步  loss = {loss.item():.4f}')
with torch.no_grad():
    probs = net(X).squeeze(1)
print('学到的输出概率：', [f'{p:.3f}' for p in probs.tolist()], '→ 四舍五入', probs.round().int().tolist())
print('阶跃函数的导数处处为 0，没法用梯度下降；换成平滑的激活函数，多层网络才能被训练。')

# %% 导出给网站（学习时可跳过）
site_export = {
    'summary': {
        'and_epochs': len(results['与 AND'][2]),
        'or_epochs': len(results['或 OR'][2]),
        'xor_errors': results['异或 XOR'][2][-1],
        'xor_probs': '、'.join(f'{p:.2f}' for p in probs.tolist()),
    },
    'extras': {name: {'w': w.tolist(), 'b': b.item(), 'errors': errs} for name, (w, b, errs) in results.items()},
}
