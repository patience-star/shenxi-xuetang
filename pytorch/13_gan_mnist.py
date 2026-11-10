"""第 13 课 · 生成对抗网络（GAN）
经典案例：DCGAN 生成手写数字。生成器从随机噪声「画」出数字，判别器努力分辨真假，两者互相博弈、共同进步。

运行：python 13_gan_mnist.py（CPU 上约 10~20 分钟）
"""

# %% 1. 数据：像素缩放到 −1 ~ 1（和生成器最后的 tanh 输出范围一致）
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

torch.manual_seed(0)
X = datasets.MNIST('data', train=True, download=True).data.float().div(127.5).sub(1).unsqueeze(1)   # [60000, 1, 28, 28]
print('真实图片', tuple(X.shape), '像素范围', X.min().item(), '~', X.max().item())
Z = 32                                     # 噪声向量的维度

# %% 2. 生成器：噪声 → 7×7 的小特征图 → 转置卷积放大到 14×14 → 28×28
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(Z, 64 * 7 * 7)
        self.bn0 = nn.BatchNorm1d(64 * 7 * 7)
        self.up1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)   # 转置卷积：边长翻倍 7 → 14
        self.bn1 = nn.BatchNorm2d(32)
        self.up2 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)    # 14 → 28

    def forward(self, z):
        x = torch.relu(self.bn0(self.fc(z))).view(-1, 64, 7, 7)
        x = torch.relu(self.bn1(self.up1(x)))
        return torch.tanh(self.up2(x))      # 输出 −1 ~ 1

# %% 3. 判别器：图片 → 「是真图」的得分（logit）
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.LeakyReLU(0.2),                       # 28 → 14
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),  # 14 → 7
            nn.Flatten(), nn.Linear(64 * 7 * 7, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

G, D = Generator(), Discriminator()
print(f'生成器参数 {sum(p.numel() for p in G.parameters()):,}，判别器参数 {sum(p.numel() for p in D.parameters()):,}')

# %% 4. 对抗训练：先练判别器（分清真假），再练生成器（骗过判别器）
opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))   # DCGAN 论文推荐的设置
opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()
fixed_z = torch.randn(16, Z)               # 固定的 16 个噪声：每轮都用它们生成，看同一批「种子」怎么越画越好
EPOCHS, CHECKPOINTS = 12, [1, 2, 3, 5, 8, 12]
history = {'d_loss': [], 'g_loss': [], 'd_real': [], 'd_fake': []}
snapshots = []
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    s = [0.0] * 4
    n = 0
    for idx in torch.randperm(60000).split(128):
        real = X[idx]
        fake = G(torch.randn(len(idx), Z))
        # ① 判别器：真图判 1、假图判 0（假图 detach，这一步不更新生成器）
        d_real, d_fake = D(real), D(fake.detach())
        loss_d = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()
        # ② 生成器：让判别器把假图判成 1
        d_fake = D(fake)
        loss_g = bce(d_fake, torch.ones_like(d_fake))
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()
        s[0] += loss_d.item(); s[1] += loss_g.item()
        s[2] += torch.sigmoid(d_real).mean().item(); s[3] += torch.sigmoid(d_fake).mean().item()
        n += 1
    for k, key in enumerate(history):
        history[key].append(s[k] / n)
    if epoch in CHECKPOINTS:
        G.eval()
        with torch.no_grad():
            snapshots.append(((G(fixed_z) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).squeeze(1))
        G.train()
    print(f'第 {epoch:2d} 轮  判别器损失 {history["d_loss"][-1]:.3f}  生成器损失 {history["g_loss"][-1]:.3f}  '
          f'D(真图) {history["d_real"][-1]:.2f}  D(假图) {history["d_fake"][-1]:.2f}  已用 {(time.time() - t0) / 60:.1f} 分钟', flush=True)
train_min = (time.time() - t0) / 60
print('D(真图)、D(假图) 是判别器认为「是真的」的概率。理想的平衡点是两者都接近 0.5：判别器真假难辨。')

# %% 5. 用训练好的生成器画数字
G.eval()
with torch.no_grad():
    samples = G(torch.randn(64, Z))
print('生成了 64 张图，形状', tuple(samples.shape))
for r in ((samples[0, 0] + 1) / 2)[4:24]:     # 用字符画看一张
    print(''.join('##' if v > 0.5 else '++' if v > 0.2 else ' .' for v in r))

# %% 6. 隐空间插值：两个噪声之间平滑过渡
with torch.no_grad():
    z1, z2 = torch.randn(Z), torch.randn(Z)
    steps = torch.stack([z1 + (z2 - z1) * t for t in torch.linspace(0, 1, 8)])
    inter = G(steps)
print('从 z1 到 z2 均匀插值 8 步，生成的图片会从一个数字平滑地变成另一个：', tuple(inter.shape))

# %% 导出给网站（学习时可跳过）
torch.manual_seed(3)
site_export = {
    'model': G,
    'golden': (torch.randn(4, Z), lambda z: G(z)),
    'history': history,
    'summary': {'epochs': EPOCHS, 'minutes': f'{train_min:.0f}', 'd_real': f'{history["d_real"][-1]:.2f}', 'd_fake': f'{history["d_fake"][-1]:.2f}',
                'g_params': f'{sum(p.numel() for p in G.parameters()):,}', 'd_params': f'{sum(p.numel() for p in D.parameters()):,}'},
    'extras': {'snapshots': torch.stack(snapshots), 'snap_epochs': CHECKPOINTS, 'z_dim': Z},
}
