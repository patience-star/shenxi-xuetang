"""第 11 课 · 注意力机制与 Transformer
经典案例：用一个迷你 Transformer 做「日期翻译」——把各种写法的日期（中文、英文、中文数字、带星期……）
翻译成统一的 YYYY-MM-DD。它是机器翻译的缩小版：编码器读懂输入，解码器一个字一个字地写出输出，
注意力决定每写一个字时该看输入的哪里。

运行：python 11_transformer_dates.py（CPU 上约 6~12 分钟，数据由程序随机生成，不需要下载）
"""

# %% 1. 生成数据：同一个日期，十种写法
import datetime
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(0)
torch.manual_seed(0)
CN = '〇一二三四五六七八九'
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
WEEK = '一二三四五六日'

def cn_num(n):                                     # 1~31 写成中文：九、十一、二十三
    return (CN[n // 10] if n >= 20 else '') + ('十' if n >= 10 else '') + (CN[n % 10] if n % 10 else '')

FORMATS = {
    '中文': lambda d: f'{d.year}年{d.month}月{d.day}日',
    '中文数字': lambda d: ''.join(CN[int(c)] for c in str(d.year)) + f'年{cn_num(d.month)}月{cn_num(d.day)}日',
    '带星期': lambda d: f'{d.year}年{d.month}月{d.day}日 星期{WEEK[d.weekday()]}',
    '斜杠（年/月/日）': lambda d: f'{d.year}/{d.month}/{d.day}',
    '美式（月/日/年）': lambda d: f'{d.month:02d}/{d.day:02d}/{d.year}',
    '欧式（日.月.年）': lambda d: f'{d.day:02d}.{d.month:02d}.{d.year}',
    '紧凑数字': lambda d: f'{d.year}{d.month:02d}{d.day:02d}',
    '英文全称': lambda d: f'{MONTHS[d.month - 1]} {d.day}, {d.year}',
    '英文缩写': lambda d: f'{MONTHS[d.month - 1][:3]} {d.day}, {d.year}',
    '日 月 年': lambda d: f'{d.day} {MONTHS[d.month - 1][:3]} {d.year}',
}
start, span = datetime.date(1950, 1, 1), (datetime.date(2049, 12, 31) - datetime.date(1950, 1, 1)).days

def sample():
    d = start + datetime.timedelta(days=random.randrange(span + 1))
    fmt = random.choice(list(FORMATS))
    return FORMATS[fmt](d), d.isoformat(), fmt

d = datetime.date(2026, 11, 9)
for name, f in FORMATS.items():
    print(f'{name:10s} {f(d):24s} → {d.isoformat()}')
pairs = [sample() for _ in range(42000)]
train_pairs, test_pairs = pairs[:40000], pairs[40000:]

# %% 2. 字表：输入和输出共用一套字符编号
chars = sorted(set(''.join(s + t for s, t, _ in pairs)))
itos = ['<pad>', '<s>'] + chars
stoi = {c: i for i, c in enumerate(itos)}
V = len(itos)
print(f'字表 {V} 个字符：', ''.join(chars))

def to_tensor(strings, length):
    t = torch.zeros(len(strings), length, dtype=torch.long)
    for i, s in enumerate(strings):
        t[i, :len(s)] = torch.tensor([stoi[c] for c in s])
    return t

SRC_LEN = max(len(s) for s, _, _ in pairs)
src_all = to_tensor([s for s, _, _ in pairs], SRC_LEN)
tgt_all = to_tensor([t for _, t, _ in pairs], 10)
tgt_in_all = torch.cat([torch.ones(len(pairs), 1, dtype=torch.long), tgt_all[:, :-1]], 1)   # 解码器输入 = <s> + 目标去掉最后一位
print('最长输入', SRC_LEN, '个字符；输出固定 10 个字符')

# %% 3. 多头注意力：每个位置拿自己的「查询」去和所有位置的「键」比相似度，再按相似度加权求和「值」
class MultiHeadAttention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, x_q, x_kv, mask=None):
        N, Tq, d = x_q.shape
        Tk, h, dh = x_kv.shape[1], self.h, d // self.h
        Q = self.q(x_q).view(N, Tq, h, dh).transpose(1, 2)        # [N, 头, Tq, dh]
        K = self.k(x_kv).view(N, Tk, h, dh).transpose(1, 2)
        Vv = self.v(x_kv).view(N, Tk, h, dh).transpose(1, 2)
        scores = Q @ K.transpose(-1, -2) / math.sqrt(dh)          # 相似度，除以 √dh 防止 softmax 过于尖锐
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))      # 被屏蔽的位置：权重为 0
        att = scores.softmax(-1)                                  # 每一行加起来等于 1
        self.att = att.detach()                                   # 记下来，后面画注意力图
        out = (att @ Vv).transpose(1, 2).reshape(N, Tq, d)        # 加权求和后把各个头拼回去
        return self.o(out)

# %% 4. 编码器层、解码器层，以及整个 Transformer（Pre-LN：先 LayerNorm 再进子层，小模型更好训练）
class EncoderLayer(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.ln1, self.att, self.ln2 = nn.LayerNorm(d), MultiHeadAttention(d, heads), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))

    def forward(self, x, pad_mask):
        y = self.ln1(x)
        x = x + self.att(y, y, pad_mask)          # 自注意力 + 残差连接（第 8 课）
        return x + self.ff(self.ln2(x))           # 逐位置的前馈网络 + 残差连接

class DecoderLayer(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.ln1, self.self_att = nn.LayerNorm(d), MultiHeadAttention(d, heads)
        self.ln2, self.cross_att = nn.LayerNorm(d), MultiHeadAttention(d, heads)
        self.ln3 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))

    def forward(self, x, memory, causal_mask, pad_mask):
        y = self.ln1(x)
        x = x + self.self_att(y, y, causal_mask)                 # 带掩码的自注意力：不能偷看后面还没写的字
        x = x + self.cross_att(self.ln2(x), memory, pad_mask)    # 交叉注意力：去输入里找需要的信息
        return x + self.ff(self.ln3(x))

class Transformer(nn.Module):
    def __init__(self, V, d=64, heads=4, ff=128, layers=2):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(V, d)
        self.enc = nn.ModuleList(EncoderLayer(d, heads, ff) for _ in range(layers))
        self.dec = nn.ModuleList(DecoderLayer(d, heads, ff) for _ in range(layers))
        self.enc_ln, self.dec_ln = nn.LayerNorm(d), nn.LayerNorm(d)
        self.out = nn.Linear(d, V)
        pos = torch.arange(64).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe = torch.zeros(64, d)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)   # 正弦位置编码：告诉模型「第几个字」
        self.register_buffer('pe', pe, persistent=False)

    def embed(self, ids):
        return self.emb(ids) * math.sqrt(self.d) + self.pe[:ids.shape[1]]

    def encode(self, src):
        pad = (src == 0)[:, None, None, :]                       # [N, 1, 1, Ts]：<pad> 不参与注意力
        x = self.embed(src)
        for layer in self.enc:
            x = layer(x, pad)
        return self.enc_ln(x), pad

    def decode(self, tgt_in, memory, pad):
        T = tgt_in.shape[1]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)   # 上三角为 True：第 i 个位置看不到 i 之后
        x = self.embed(tgt_in)
        for layer in self.dec:
            x = layer(x, memory, causal, pad)
        return self.out(self.dec_ln(x))

    def forward(self, src, tgt_in):
        memory, pad = self.encode(src)
        return self.decode(tgt_in, memory, pad)

model = Transformer(V)
n_params = sum(p.numel() for p in model.parameters())
print(f'参数 {n_params:,}：2 层编码器 + 2 层解码器，模型维度 64，4 个注意力头')

# %% 5. 训练：教师强制（解码器输入正确答案的前缀，预测下一个字符）
@torch.no_grad()
def translate(src):
    """贪心解码：从 <s> 开始，每次取概率最大的字符接到后面，共 10 步。"""
    model.eval()
    memory, pad = model.encode(src)
    out = torch.ones(len(src), 1, dtype=torch.long)
    for _ in range(10):
        nxt = model.decode(out, memory, pad)[:, -1].argmax(-1, keepdim=True)
        out = torch.cat([out, nxt], 1)
    model.train()
    return out[:, 1:]

def exact_match(n=2000):
    pred = translate(src_all[40000:40000 + n])
    return (pred == tgt_all[40000:40000 + n]).all(1).float().mean().item()

EPOCHS = 30
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)   # 学习率按余弦曲线从 1e-3 降到接近 0（第 6 课）
history = {'train_loss': [], 'exact': []}
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    total, n = 0.0, 0
    for idx in torch.randperm(40000).split(128):
        logits = model(src_all[idx], tgt_in_all[idx])
        loss = F.cross_entropy(logits.reshape(-1, V), tgt_all[idx].reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()
        n += 1
    sched.step()
    history['train_loss'].append(total / n)
    history['exact'].append(exact_match())
    print(f'第 {epoch:2d} 轮  训练损失 {total / n:.4f}  测试集整句完全正确 {history["exact"][-1]:.2%}  已用 {time.time() - t0:.0f} 秒', flush=True)
train_secs = time.time() - t0

# %% 6. 试试翻译，并按写法统计正确率
tests = ['2026年11月9日', '二〇二六年十一月九日', 'November 9, 2026', '9 Nov 2026', '11/09/2026', '09.11.2026', '1999年12月31日 星期五', '20080808']
pred = translate(to_tensor(tests, SRC_LEN))
for s, p in zip(tests, pred):
    print(f'{s:24s} → {"".join(itos[i] for i in p)}')
by_fmt = {}
pred_all = translate(src_all[40000:])
for k, (_, _, fmt) in enumerate(test_pairs):
    by_fmt.setdefault(fmt, []).append(bool((pred_all[k] == tgt_all[40000 + k]).all()))
for fmt, r in by_fmt.items():
    print(f'{fmt:10s} {sum(r) / len(r):.1%}（{len(r)} 条）')

# %% 7. 看注意力：写每个输出字符时，交叉注意力看向输入的哪里
src = to_tensor(['November 9, 2026'], SRC_LEN)[:, :16]
out = translate(src)
tgt_in = torch.cat([torch.ones(1, 1, dtype=torch.long), out[:, :-1]], 1)
model.eval()
with torch.no_grad():
    model(src, tgt_in)
att = model.dec[-1].cross_att.att[0].mean(0)          # 最后一层、4 个头平均：[输出 10, 输入 16]
for i, row in enumerate(att):
    j = row.argmax().item()
    print(f'写「{itos[out[0, i]]}」时最关注输入的第 {j + 1} 个字符「{"November 9, 2026"[j]}」（权重 {row[j]:.2f}）')

# %% 导出给网站（学习时可跳过）
gs = to_tensor(['二〇二六年十一月九日'], 10)
gt = torch.cat([torch.ones(1, 1, dtype=torch.long), to_tensor(['2026-11-09'], 10)[:, :-1]], 1)
site_export = {
    'model': model,
    'golden': ((gs, gt), lambda s, t: model(s, t)[0]),
    'history': history,
    'summary': {
        'params': f'{n_params:,}', 'epochs': EPOCHS, 'exact': f'{history["exact"][-1]:.2%}', 'secs': f'{train_secs:.0f}', 'vocab': V,
        'worst_fmt': min(by_fmt, key=lambda f: sum(by_fmt[f]) / len(by_fmt[f])),
    },
    'extras': {'itos': itos, 'examples': tests + ['2008年8月8日', 'Jan 1, 2000', '二〇〇八年八月八日', '31.12.1999']},
}
