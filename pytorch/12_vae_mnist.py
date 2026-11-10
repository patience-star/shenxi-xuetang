"""第 12 课 · 自编码器与 VAE
经典案例：把手写数字压缩到只有 2 个数的「隐空间」里，再从隐空间生成新的数字。
同时训练一个普通自编码器（AE）做对比：为什么 AE 的隐空间不能拿来随便采样，而 VAE 可以。

运行：python 12_vae_mnist.py（CPU 上约 2~4 分钟）
"""

# %% 1. 数据：像素保持 0~1，用作「重建」的目标
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

torch.manual_seed(0)
X_train = datasets.MNIST('data', train=True, download=True).data.float().div(255).flatten(1)   # [60000, 784]
test = datasets.MNIST('data', train=False, download=True)
X_test, y_test = test.data.float().div(255).flatten(1), test.targets
print('训练集', tuple(X_train.shape), '——这次不需要标签：模型只要学会「压缩再还原」')

# %% 2. 编码器和解码器：784 → 128 → 2 → 128 → 784
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 128)
        self.mu = nn.Linear(128, 2)          # 隐空间只有 2 维：方便画在平面上
        self.logvar = nn.Linear(128, 2)      # VAE 额外输出「不确定度」（方差的对数）；普通 AE 不用它

    def forward(self, x):
        h = torch.relu(self.fc(x))
        return self.mu(h), self.logvar(h)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 784)

    def forward(self, z):
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(z))))   # 输出 784 个 0~1 的像素

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc, self.dec = Encoder(), Decoder()

    def forward(self, x):
        z, _ = self.enc(x)                   # 普通自编码器：编码成一个确定的点
        return self.dec(z), z, None

class VAE(AE):
    def forward(self, x):
        mu, logvar = self.enc(x)             # VAE：编码成一个「分布」（均值 mu、方差 exp(logvar)）
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps   # 重参数化技巧：从分布里抽样，但梯度还能传回 mu 和 logvar
        return self.dec(z), mu, logvar

# %% 3. 损失：重建误差（+ VAE 的 KL 散度）
def losses(model, x):
    recon, mu, logvar = model(x)
    rec = F.binary_cross_entropy(recon, x, reduction='sum') / len(x)          # 每张图的重建误差（784 个像素求和）
    if logvar is None:
        return rec, rec, torch.tensor(0.0)
    kl = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp()) / len(x)      # 让每张图的分布靠近标准正态 N(0, 1)
    return rec + kl, rec, kl

def train(model, epochs=15):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist = {'rec': [], 'kl': []}
    for epoch in range(1, epochs + 1):
        tot = [0.0, 0.0]
        for idx in torch.randperm(60000).split(128):
            loss, rec, kl = losses(model, X_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot[0] += rec.item() * len(idx)
            tot[1] += kl.item() * len(idx)
        hist['rec'].append(tot[0] / 60000)
        hist['kl'].append(tot[1] / 60000)
        if epoch % 3 == 0 or epoch == 1:
            print(f'  第 {epoch:2d} 轮  重建误差 {hist["rec"][-1]:.1f}  KL {hist["kl"][-1]:.2f}', flush=True)
    return hist

t0 = time.time()
torch.manual_seed(0)
ae = AE()
print('普通自编码器 AE：')
hist_ae = train(ae)
torch.manual_seed(0)
vae = VAE()
print('变分自编码器 VAE：')
hist_vae = train(vae)
print(f'用时 {time.time() - t0:.0f} 秒。VAE 的重建误差略高一点——它要同时满足 KL 这个「规矩」。')

# %% 4. 隐空间长什么样：把测试集编码成二维坐标
with torch.no_grad():
    z_ae, _ = ae.enc(X_test)
    z_vae, _ = vae.enc(X_test)                # 用均值 mu 作为每张图的坐标
for name, z in [('AE ', z_ae), ('VAE', z_vae)]:
    print(f'{name} 隐空间坐标范围：x {z[:, 0].min():.1f} ~ {z[:, 0].max():.1f}，y {z[:, 1].min():.1f} ~ {z[:, 1].max():.1f}，'
          f'均值 ({z[:, 0].mean():.2f}, {z[:, 1].mean():.2f})，标准差 ({z[:, 0].std():.2f}, {z[:, 1].std():.2f})')
print('AE 的坐标可以散得很开、中间留着大片空白；VAE 被 KL 项拉向原点附近，挤成一团紧凑、连续的分布。')

# %% 5. 从隐空间随机采样生成新数字：VAE 可以，AE 往往不行
torch.manual_seed(1)
z = torch.randn(8, 2)                         # 从标准正态分布里随便抽 8 个点
with torch.no_grad():
    for name, model in [('AE ', ae), ('VAE', vae)]:
        imgs = model.dec(z)
        sharp = ((imgs > 0.2) & (imgs < 0.8)).float().mean().item()    # 灰色（模糊）像素的比例
        print(f'{name} 随机采样生成的 8 张图里，模糊的灰色像素占 {sharp:.1%}')
print('AE 的隐空间里，很多点附近从来没有训练样本，解码出来是四不像；VAE 的隐空间被「填满」了，随便取一点都像个数字。')

# %% 导出给网站（学习时可跳过）
torch.manual_seed(2)
idx = torch.randperm(10000)[:2000]
gz, gx = torch.randn(6, 2), X_test[:4]
site_export = {
    'model': {'vae': vae, 'ae_dec': ae.dec},
    'golden': ((gz, gx), lambda z, x: (vae.dec(z), vae.enc(x)[0], ae.dec(z))),
    'history': {'ae_rec': hist_ae['rec'], 'vae_rec': hist_vae['rec'], 'vae_kl': hist_vae['kl']},
    'summary': {'ae_rec': f'{hist_ae["rec"][-1]:.1f}', 'vae_rec': f'{hist_vae["rec"][-1]:.1f}', 'vae_kl': f'{hist_vae["kl"][-1]:.2f}'},
    'extras': {'z_vae': z_vae[idx], 'z_ae': z_ae[idx], 'y': y_test[idx].tolist()},
}
