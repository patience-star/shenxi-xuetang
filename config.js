/* =====================================================================
 * 站点配置 —— 日常调整只需改这个文件，保存后刷新浏览器即可生效（无需构建）。
 * 详细说明见 README.md「常见修改怎么做」。
 * ===================================================================== */
window.SITE = {
  name: '深析学堂',
  slogan: '从感知机到 Transformer，看得见、摸得着的深度学习',

  // 顶部导航：[显示文字, 路由]
  nav: [
    ['首页', '#/'],
    ['课程', '#/learn'],
    ['经典案例', '#/cases'],
    ['排错手册', '#/handbook'],
  ],

  // 课程：数组顺序 = 学习顺序；文件位于 content/lessons/<id>.js
  // 注释掉一行即可隐藏该课；新增：复制 content/lessons/_template.js，再把 id 加到这里
  lessons: [
    'tensor',       // 张量与 PyTorch 入门
    'gradient',     // 导数、梯度与梯度下降
    'perceptron',   // 感知机
    'backprop',     // 计算图与反向传播
    'mlp',          // 多层感知机
    'training',     // 训练技巧：优化器、初始化与正则化
    'cnn',          // 卷积神经网络与 LeNet
    'resnet',       // 经典 CNN 与残差网络
    'rnn',          // 循环神经网络
    'lstm',         // LSTM 与 GRU
    'transformer',  // 注意力机制与 Transformer
    'vae',          // 自编码器与 VAE
    'gan',          // 生成对抗网络
  ],

  quizPass: 0.6,          // 闯关练习及格线（正确率），达到即视为本课完成
  storageKey: 'sxdl.v1',  // 浏览器本地进度的存储 key（进度结构有大改动时把 v1 改成 v2）
};
