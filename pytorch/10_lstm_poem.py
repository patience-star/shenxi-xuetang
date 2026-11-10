"""第 10 课 · LSTM 与 GRU
经典案例：用 LSTM 写藏头诗；并在同一份唐诗语料上对比普通 RNN、GRU、LSTM。

运行：python 10_lstm_poem.py（CPU 上约 10~20 分钟）。需要 data/tang5.txt（网站第 9 课「经典案例」里可以下载）。
"""

# %% 1. 读取语料、建立字表（和第 9 课完全相同）
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
counter = collections.Counter(''.join(poems))
SPECIALS = ['<pad>', '<s>', '<unk>']
chars = [c for c, _ in counter.most_common(2500)]
itos = SPECIALS + chars
stoi = {c: i for i, c in enumerate(itos)}
V = len(itos)

def encode(text):
    return [stoi.get(c, 2) for c in text]

L = 48
X = torch.zeros(len(poems), L, dtype=torch.long)
Y = torch.zeros(len(poems), L, dtype=torch.long)
for i, p in enumerate(poems):
    ids = encode(p)
    X[i, :len(ids)] = torch.tensor([1] + ids[:-1])
    Y[i, :len(ids)] = torch.tensor(ids)
perm = torch.randperm(len(poems))
val_idx, train_idx = perm[:1000], perm[1000:]
print(f'{len(poems)} 首诗，字表 {V} 个（含 3 个特殊记号）')

# %% 2. 同样的结构，只换「循环单元」：RNN / GRU / LSTM
class CharModel(nn.Module):
    def __init__(self, V, cell='lstm', d=128):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.rnn = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}[cell](d, d, batch_first=True)
        self.out = nn.Linear(d, V)
        self.out.weight = self.emb.weight           # 和第 9 课一样：输出层与嵌入共享权重

    def forward(self, x, state=None):
        o, state = self.rnn(self.emb(x), state)     # LSTM 的 state 是 (h, c) 两个张量
        return self.out(o), state, o

for cell in ['rnn', 'gru', 'lstm']:
    n_rec = sum(p.numel() for p in CharModel(V, cell).rnn.parameters())
    print(f'{cell.upper():4s} 循环层参数 {n_rec:,}')
print('GRU 有 3 组权重（重置门、更新门、候选状态），LSTM 有 4 组（输入门、遗忘门、候选值、输出门），所以分别是 RNN 的 3 倍和 4 倍。')

# %% 3. 对比训练：同样的数据、轮数、学习率，只换循环单元
def run_epochs(model, opt, epochs, history):
    for _ in range(epochs):
        for idx in train_idx[torch.randperm(len(train_idx))].split(64):
            logits = model(X[idx])[0]
            loss = F.cross_entropy(logits.reshape(-1, V), Y[idx].reshape(-1), ignore_index=0)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.cross_entropy(model(X[val_idx])[0].reshape(-1, V), Y[val_idx].reshape(-1), ignore_index=0).item()
        model.train()
        history.append(vl)
        print(f'  第 {len(history):2d} 轮  验证损失 {vl:.3f}（困惑度 {math.exp(vl):.0f}）', flush=True)
    return history

t0 = time.time()
curves, models = {}, {}
for cell in ['rnn', 'gru', 'lstm']:
    print(cell.upper())
    torch.manual_seed(0)
    models[cell] = CharModel(V, cell)
    opts = torch.optim.Adam(models[cell].parameters(), lr=3e-3)
    curves[cell] = run_epochs(models[cell], opts, 6, [])
    if cell == 'lstm':
        lstm_opt = opts
print(f'用时 {time.time() - t0:.0f} 秒。同样训练 6 轮：RNN {curves["rnn"][-1]:.3f}，GRU {curves["gru"][-1]:.3f}，LSTM {curves["lstm"][-1]:.3f}')

# %% 4. 继续训练 LSTM：再练 8 轮，学习率减半
model = models['lstm']
for g in lstm_opt.param_groups:
    g['lr'] = 1.5e-3
history = run_epochs(model, lstm_opt, 8, list(curves['lstm']))
total_minutes = (time.time() - t0) / 60

# %% 5. 写藏头诗：每句的第一个字由我们指定，其余的字让 LSTM 按概率逐个生成
def acrostic(heads, temperature=0.8):
    model.eval()
    state, out = None, ''
    ids = torch.tensor([[1]])                        # 从 <s> 开始
    with torch.no_grad():
        for k, head in enumerate(heads):
            for pos in range(6):                     # 每句 5 个字 + 1 个标点
                logits, state, _ = model(ids, state)
                if pos == 0:
                    c = head                         # 句首：用指定的字
                elif pos == 5:
                    c = '，' if k % 2 == 0 else '。'
                else:
                    z = logits[0, -1] / temperature
                    z[:3] = -1e9                     # 不生成特殊记号
                    z[stoi['，']] = z[stoi['。']] = -1e9
                    c = itos[torch.multinomial(torch.softmax(z, 0), 1).item()]
                out += c
                ids = torch.tensor([[stoi.get(c, 2)]])
    return out

torch.manual_seed(7)
for heads in ['深度学习', '春夏秋冬', '山高水长', '明月清风']:
    print(f'【{heads}】{acrostic(heads)}')

# %% 6. 打开 LSTM 看门：读「床前明月光」时，三个门的平均开度
W_ih, W_hh = model.rnn.weight_ih_l0, model.rnn.weight_hh_l0
b = model.rnn.bias_ih_l0 + model.rnn.bias_hh_l0
h = torch.zeros(128)
c = torch.zeros(128)
with torch.no_grad():
    for ch in '床前明月光，疑是地上霜。':
        z = W_ih @ model.emb.weight[stoi[ch]] + W_hh @ h + b
        i, f, g, o = z.chunk(4)                      # PyTorch 的 LSTM 按「输入门、遗忘门、候选值、输出门」排列
        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
        c = f * c + i * g                            # 细胞状态：留下一部分旧记忆，再写入一部分新内容
        h = o * torch.tanh(c)                        # 隐状态：从细胞状态里读出一部分
        print(f'{ch}  遗忘门 {f.mean():.2f}  输入门 {i.mean():.2f}  输出门 {o.mean():.2f}  |细胞状态| {c.abs().mean():.2f}')

# %% 7. 梯度随时间衰减：训练后的 RNN vs LSTM
def grad_over_time(m):
    sel = val_idx[(Y[val_idx] != 0).sum(1) == 48][:100]
    emb = m.emb(X[sel]).detach().requires_grad_(True)
    o, _ = m.rnn(emb)
    F.cross_entropy(m.out(o[:, -1]), Y[sel, -1]).backward()
    return emb.grad.norm(dim=2).mean(0)

g_rnn, g_lstm = grad_over_time(models['rnn']), grad_over_time(model)
for t in [46, 36, 24, 12, 0]:
    print(f'第 {t + 1:2d} 个位置：RNN {g_rnn[t].item():.1e}   LSTM {g_lstm[t].item():.1e}')
print('每个位置上 LSTM 的梯度都比 RNN 大好几倍、衰减也慢一些：细胞状态这条「传送带」让梯度更容易传得远。')
print('这个任务里下一个字主要取决于前几个字，所以差距不算悬殊；需要长距离记忆的任务上差别会大得多。')

# %% 导出给网站（学习时可跳过）
gold = torch.tensor([[1] + encode('床前明月光，疑是地上霜。举头望明月')])
site_export = {
    'model': model,
    'golden': (gold, lambda x: (model(x)[2][0], model(x)[0][0, -1])),
    'history': {'val_loss': history, 'rnn': curves['rnn'], 'gru': curves['gru'], 'lstm': curves['lstm']},
    'summary': {
        'rnn6': f'{curves["rnn"][-1]:.3f}', 'gru6': f'{curves["gru"][-1]:.3f}', 'lstm6': f'{curves["lstm"][-1]:.3f}',
        'final': f'{history[-1]:.3f}', 'ppl': f'{math.exp(history[-1]):.0f}', 'epochs': len(history), 'minutes': f'{total_minutes:.0f}',
        'g_rnn0': f'{g_rnn[0].item():.1e}', 'g_lstm0': f'{g_lstm[0].item():.1e}',
    },
    'extras': {'vocab': ''.join(chars), 'specials': SPECIALS, 'grad_rnn': g_rnn.tolist(), 'grad_lstm': g_lstm.tolist()},
}
