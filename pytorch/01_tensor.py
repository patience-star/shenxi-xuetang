"""第 1 课 · 张量与 PyTorch 入门
经典案例：用张量装一批手写数字（MNIST），看清每一维的含义。

运行：python 01_tensor.py      （首次会自动下载 MNIST，约 11 MB，存到 ./data）
建议在 VS Code / PyCharm 里按 "# %%" 分段运行，边看输出边读注释。
"""

# %% 1. 创建张量：深度学习里的一切都是张量
import torch

torch.manual_seed(0)                      # 固定随机种子：每次运行结果一样，方便对照

# 张量（tensor）就是多维数组。下面从 0 维到 3 维各建一个
scalar = torch.tensor(3.14)               # 0 维：一个数，比如今天的气温
vector = torch.tensor([1.70, 65.0, 28])   # 1 维：一个人的 [身高, 体重, 年龄]
matrix = torch.zeros(3, 4)                # 2 维：3 行 4 列，全是 0
cube = torch.arange(24).reshape(2, 3, 4)  # 3 维：0~23 排成 2 块 × 3 行 × 4 列

for name, t in [('scalar', scalar), ('vector', vector), ('matrix', matrix), ('cube', cube)]:
    # ndim：有几个维度；shape：每一维多长；numel：元素总数；dtype：数据类型
    print(f'{name:7s} ndim={t.ndim}  shape={tuple(t.shape)}  numel={t.numel():2d}  dtype={t.dtype}')
print(cube)

# %% 2. 索引与切片：和 Python 列表、NumPy 的写法一样
x = torch.arange(12).reshape(3, 4)        # 3 行 4 列
print(x)
print('第 0 行        x[0]      =', x[0])
print('第 1 列        x[:, 1]   =', x[:, 1])
print('右下角 2×2     x[1:, 2:] =\n', x[1:, 2:])
print('大于 6 的元素  x[x > 6]  =', x[x > 6])   # 布尔掩码：挑出满足条件的元素

# %% 3. 变形：reshape / view / permute / unsqueeze / squeeze
a = torch.arange(6)                        # [0, 1, 2, 3, 4, 5]
print('reshape(2, 3)：元素顺序不变，按行重新排\n', a.reshape(2, 3))
print('reshape(3, -1) 的形状：', tuple(a.reshape(3, -1).shape), ' ← -1 表示这一维让 PyTorch 自己算')
b = a.reshape(2, 3)
print('转置 b.T（等价于 b.permute(1, 0)）：行列互换，元素的排列顺序真的变了\n', b.T)
c = torch.rand(3, 28, 28)                  # 一张 3 通道的图片：[通道, 高, 宽]
print('unsqueeze(0) 在最前面加一个批量维：', tuple(c.unsqueeze(0).shape))
print('squeeze() 去掉所有长度为 1 的维：  ', tuple(c.unsqueeze(0).squeeze().shape))
print('permute(1, 2, 0)：CHW → HWC（matplotlib 画图要这种顺序）', tuple(c.permute(1, 2, 0).shape))

# %% 4. 广播：形状不同的张量也能直接运算
scores = torch.tensor([[1., 2., 3.],
                       [4., 5., 6.]])      # [2, 3]：2 个样本、各 3 个类别的得分
bias = torch.tensor([10., 20., 30.])      # [3]：每个类别一个偏置
print('[2, 3] + [3] →', tuple((scores + bias).shape), '：bias 被「复制」到每一行（并不真的复制内存）')
print(scores + bias)

col = torch.tensor([[1.], [2.], [3.]])    # [3, 1]
row = torch.tensor([[10., 20., 30., 40.]])  # [1, 4]
print('[3, 1] + [1, 4] →', tuple((col + row).shape), '：两边各自沿长度为 1 的维扩展')
print(col + row)

# 常见坑：[N] 和 [N, 1] 相减，会广播成 [N, N]，而不是想要的 [N]
pred = torch.zeros(4)                      # 模型输出 [4]
target = torch.zeros(4, 1)                 # 标签 [4, 1]
print('坑：[4] - [4, 1] →', tuple((pred - target).shape))

# %% 5. 沿某个维度计算：sum / mean / argmax，以及矩阵乘法
m = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])
print('全部求和               ', m.sum())
print('sum(dim=0)：压掉第 0 维', m.sum(dim=0), '← 按列加，剩下 3 个数')
print('sum(dim=1)：压掉第 1 维', m.sum(dim=1), '← 按行加，剩下 2 个数')
print('keepdim=True 保留被压掉的维：', tuple(m.sum(dim=1, keepdim=True).shape))
print('argmax(dim=1)：每行最大值的位置', m.argmax(dim=1), '← 分类时就这样从得分里挑出预测类别')
W = torch.rand(3, 5)
print('矩阵乘法 [2, 3] @ [3, 5] →', tuple((m @ W).shape), '：前者的列数要等于后者的行数')
print('逐元素乘法 m * m →', tuple((m * m).shape))

# %% 6. 案例：读取一批手写数字（MNIST）
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ToTensor() 做两件事：① 28×28 的灰度图 → 形状 [1, 28, 28] 的张量；② 像素从 0~255 缩放到 0~1
train_set = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
print('训练集样本数：', len(train_set))

loader = DataLoader(train_set, batch_size=64, shuffle=True)   # 每次取 64 张，每轮打乱顺序
images, labels = next(iter(loader))       # 取出第一批
print('images:', tuple(images.shape), images.dtype, ' ← [批量 N, 通道 C, 高 H, 宽 W]')
print('labels:', tuple(labels.shape), labels.dtype, ' ← 每张图一个类别编号 0~9（整数 long）')
print('像素范围：', images.min().item(), '~', images.max().item())
print('前 10 个标签：', labels[:10].tolist())

# %% 7. 一张图片就是一个数字矩阵：用字符把它「画」出来
img = images[0, 0]                        # 第 0 张图的第 0 个通道 → 形状 [28, 28]
print('这张图的标签是', labels[0].item(), '，形状', tuple(img.shape), '，左上角 3×3 的像素值：')
print(img[:3, :3])
for r in img[3:25]:                       # 打印中间 22 行：亮的像素画成 #，暗的画成 .
    print(''.join('##' if v > 0.5 else '++' if v > 0.2 else ' .' for v in r))

# %% 8. 送进网络之前的预处理：标准化 + 展平
mean, std = images.mean().item(), images.std().item()
print(f'这一批的像素均值 {mean:.4f}，标准差 {std:.4f}')
# 整个 MNIST 训练集的均值、标准差约为 0.1307 和 0.3081，后面几课都用这两个数做标准化
x = (images - 0.1307) / 0.3081            # 广播：[64, 1, 28, 28] 减去一个数、除以一个数
print(f'标准化后：均值 {x.mean().item():.4f}，标准差 {x.std().item():.4f}  ← 接近 0 和 1，训练更稳定')
flat = x.flatten(1)                        # 从第 1 维开始展平：[64, 1, 28, 28] → [64, 784]
print('展平后给多层感知机用：', tuple(flat.shape))
print('等价写法 x.view(64, -1)：', tuple(x.view(64, -1).shape))

# %% 9. 与 NumPy 互转、放到 GPU
import numpy as np

arr = images[0, 0].numpy()                # 张量 → NumPy（在 CPU 上共享同一块内存，不复制）
print(type(arr).__name__, arr.shape, arr.dtype)
back = torch.from_numpy(np.ones((2, 2), dtype=np.float32))   # NumPy → 张量
print(back)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('可用设备：', device, ' ← 有 NVIDIA 显卡并装了 CUDA 版 PyTorch 才会是 cuda')
images = images.to(device)                # 模型和数据必须放在同一个设备上
print('images 现在在：', images.device)

# %% 导出给网站（学习时可跳过）
# 把这一批里的前 16 张图片（0~255 的整数）和标签交给网页，画「像素就是数字」的互动图
site_export = {
    'summary': {
        'batch_shape': str(list(images.shape)),
        'n_train': f'{len(train_set):,}',
        'mean': f'{mean:.4f}',
        'std': f'{std:.4f}',
    },
    'extras': {
        'digits': (images[:16, 0].cpu() * 255).round().to(torch.uint8),   # [16, 28, 28]
        'labels': labels[:16].tolist(),
    },
}
