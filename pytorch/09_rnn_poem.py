"""第 9 课 · 循环神经网络（RNN）
经典案例：字符级语言模型——读入半句唐诗，预测下一个字（就像手机输入法的联想），还能自己续写。

语料：《全唐诗》里的 17,608 首五言诗（chinese-poetry 项目，MIT 许可；整理脚本见 tools/prep_poems.py）。
运行：python 09_rnn_poem.py（CPU 上约 5~10 分钟）。需要 data/tang5.txt（网站第 9 课「经典案例」里可以下载）。
"""

# %% 1. 读取语料，建立字表
import collections
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
if not os.path.exists('data/tang5.txt'):
    sys.exit('找不到 data/tang5.txt：请从网站第 9 课页面下载，放到本脚本所在目录的 data 文件夹里')
poems = open('data/tang5.txt', encoding='utf-8').read().split()
print('诗的数量：', len(poems), '  第一首：', poems[0])
counter = collections.Counter(''.join(poems))
SPECIALS = ['<pad>', '<s>', '<unk>']               # 填充、开头标记、生僻字
chars = [c for c, _ in counter.most_common(2500)]
itos = SPECIALS + chars                            # 编号 → 字
stoi = {c: i for i, c in enumerate(itos)}          # 字 → 编号
V = len(itos)
cover = sum(counter[c] for c in chars) / sum(counter.values())
print(f'不同的字共 {len(counter)} 个；保留最常用的 2500 个（覆盖 {cover:.1%} 的文本），其余记作 <unk>')

# %% 2. 把每首诗变成数字序列：输入 = <s> + 前面的字，目标 = 每个位置的「下一个字」
def encode(text):
    return [stoi.get(c, 2) for c in text]

L = 48                                             # 律诗 48 个字符（含标点），绝句 24 个，不足的补 <pad>
X = torch.zeros(len(poems), L, dtype=torch.long)
Y = torch.zeros(len(poems), L, dtype=torch.long)
for i, p in enumerate(poems):
    ids = encode(p)
    X[i, :len(ids)] = torch.tensor([1] + ids[:-1])
    Y[i, :len(ids)] = torch.tensor(ids)
print('输入：', ''.join(itos[i] for i in X[0, :12]), ' ← 以 <s> 开头')
print('目标：', ''.join(itos[i] for i in Y[0, :12]), ' ← 输入整体往左挪一格：每个位置都在预测下一个字')
perm = torch.randperm(len(poems))
val_idx, train_idx = perm[:1000], perm[1000:]      # 留 1000 首做验证

# %% 3. 模型：嵌入 → RNN → 输出层
class CharRNN(nn.Module):
    def __init__(self, V, d=128):
        super().__init__()
        self.emb = nn.Embedding(V, d)              # 每个字对应一个 128 维向量（词嵌入），随训练一起学
        self.rnn = nn.RNN(d, d, batch_first=True)  # h_t = tanh(W_ih·x_t + b_ih + W_hh·h_(t−1) + b_hh)
        self.out = nn.Linear(d, V)
        self.out.weight = self.emb.weight          # 权重共享：输出层复用嵌入矩阵，参数少一半

    def forward(self, x, h=None):
        o, h = self.rnn(self.emb(x), h)            # o：每个时刻的隐状态 [N, T, 128]
        return self.out(o), h, o                   # 每个时刻对 V 个字的得分

model = CharRNN(V)
print(model)
print(f'参数 {sum(p.numel() for p in model.parameters()):,} 个（嵌入矩阵 {V}×128 占了大部分）')

# %% 4. 训练：每个位置都是一道「下一个字是什么」的分类题
opt = torch.optim.Adam(model.parameters(), lr=3e-3)

def val_loss():
    model.eval()
    with torch.no_grad():
        logits = model(X[val_idx])[0]
        loss = F.cross_entropy(logits.reshape(-1, V), Y[val_idx].reshape(-1), ignore_index=0)
    model.train()
    return loss.item()

EPOCHS = 10
history = {'train_loss': [], 'val_loss': []}
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    total, n = 0.0, 0
    for idx in train_idx[torch.randperm(len(train_idx))].split(64):
        logits = model(X[idx])[0]
        loss = F.cross_entropy(logits.reshape(-1, V), Y[idx].reshape(-1), ignore_index=0)   # <pad> 的位置不算损失
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 梯度裁剪：RNN 容易梯度爆炸，总范数超过 1 就按比例缩小
        opt.step()
        total += loss.item()
        n += 1
    history['train_loss'].append(total / n)
    history['val_loss'].append(val_loss())
    print(f'第 {epoch:2d} 轮  训练损失 {total / n:.3f}  验证损失 {history["val_loss"][-1]:.3f}'
          f'（困惑度 {math.exp(history["val_loss"][-1]):.0f}）  已用 {time.time() - t0:.0f} 秒', flush=True)
train_secs = time.time() - t0
print('困惑度 = e^损失，可以理解为「模型平均在多少个字之间犹豫」。随机猜是 2500，学过之后小得多。')

# %% 5. 预测下一个字（输入法联想）
def next_chars(text, k=5):
    with torch.no_grad():
        logits = model(torch.tensor([[1] + encode(text)]))[0]
    p = torch.softmax(logits[0, -1], 0)
    top = p.topk(k)
    return '  '.join(f'{itos[i]} {v:.0%}' for v, i in zip(top.values.tolist(), top.indices.tolist()))

for text in ['床前明月', '白日依山', '春眠不觉', '两个黄鹂鸣', '大漠孤烟', '海内存知']:
    print(f'{text:6s} → {next_chars(text)}')

# %% 6. 让它自己写：每次按概率抽一个字，接到后面继续
def generate(n_lines=4, temperature=0.8):
    ids, h, out = torch.tensor([[1]]), None, ''
    with torch.no_grad():
        while len(out) < n_lines * 6:
            logits, h, _ = model(ids, h)
            pos = len(out) % 6
            if pos == 5:                           # 第 6 个位置固定放标点，保证五言格式
                c = '，' if len(out) % 12 == 5 else '。'
            else:
                z = logits[0, -1] / temperature    # 温度：越低越保守（总挑最常见的字），越高越大胆
                z[:3] = -1e9                       # 不生成 <pad>、<s>、<unk>
                z[stoi['，']] = z[stoi['。']] = -1e9
                c = itos[torch.multinomial(torch.softmax(z, 0), 1).item()]
            out += c
            ids = torch.tensor([[stoi[c]]])
    return out

torch.manual_seed(42)
for t in [0.6, 1.0]:
    for _ in range(2):
        print(f'温度 {t}：{generate(temperature=t)}')

# %% 7. 梯度随时间衰减：最后一个字的预测，对前面每个位置的输入有多敏感？
sel = val_idx[(Y[val_idx] != 0).sum(1) == 48][:100]  # 取 100 首律诗（满 48 个字）
emb = model.emb(X[sel]).detach().requires_grad_(True)
o, _ = model.rnn(emb)
loss = F.cross_entropy(model.out(o[:, -1]), Y[sel, -1])
loss.backward()
grad_t = emb.grad.norm(dim=2).mean(0)              # 每个位置的梯度大小 [48]
for t in [47, 42, 36, 24, 12, 0]:
    print(f'第 {t + 1:2d} 个位置的梯度：{grad_t[t].item():.2e}')
print('离预测位置越远，梯度越小（大约按指数衰减）：普通 RNN 很难记住很久以前的信息。')

# %% 导出给网站（学习时可跳过）
gold = torch.tensor([[1] + encode('床前明月光，疑是地上霜。举头望明月')])
site_export = {
    'model': model,
    'golden': (gold, lambda x: (model(x)[2][0], model(x)[0][0, -1])),   # 每个时刻的隐状态 + 最后一个时刻的得分
    'history': history,
    'summary': {
        'n_poems': f'{len(poems):,}', 'vocab': V - 3, 'cover': f'{cover:.1%}', 'epochs': EPOCHS,
        'val_loss': f'{history["val_loss"][-1]:.3f}', 'ppl': f'{math.exp(history["val_loss"][-1]):.0f}', 'minutes': f'{train_secs / 60:.0f}',
        'g_last': f'{grad_t[46].item():.1e}', 'g_first': f'{grad_t[0].item():.1e}',
    },
    'extras': {'vocab': ''.join(chars), 'specials': SPECIALS, 'grad_t': grad_t.tolist()},
}
