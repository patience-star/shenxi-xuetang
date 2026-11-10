/* 第 6 课：训练技巧 */
Content.addLesson({
  id: 'training',
  title: '训练技巧：优化器、初始化与正则化',
  icon: '🛠️',
  stage: '神经网络基础',
  summary: '同一个网络，有人训练出 98%，有人卡在 10%。差别在优化器、初始化、BatchNorm 和正则化——这一课把它们一个个拆开做对照实验。',
  minutes: 55,
  goals: [
    '知道训练不顺的四类症状，以及每类该从哪里下手',
    '理解 Momentum、RMSProp、Adam 各自解决了 SGD 的什么问题',
    '理解初始化为什么重要，Xavier / He 初始化在保持什么',
    '理解 BatchNorm 做了什么、训练和测试时有何不同',
    '会诊断过拟合，掌握数据增强、权重衰减、Dropout、早停',
  ],

  videos: [
    { bvid: 'BV1kX4y1g7jp', p: 2, title: '11 模型选择 + 过拟合和欠拟合（第 2 集）', up: '跟李沐学AI', minutes: 16, note: '模型容量、训练误差与泛化误差' },
    { bvid: 'BV1u64y1i75a', p: 2, title: '14 数值稳定性 + 模型初始化和激活函数（第 2 集）', up: '跟李沐学AI', minutes: 24, note: '让每层输出和梯度的方差保持稳定' },
    { bvid: 'BV1X44y1r77r', p: 1, title: '28 批量归一化（第 1 集）', up: '跟李沐学AI', minutes: 17, note: 'BatchNorm 的做法与作用' },
    { bvid: 'BV1Y5411c7aY', p: 1, title: '13 丢弃法（第 1 集）', up: '跟李沐学AI', minutes: 11, note: 'Dropout 的动机与实现' },
  ],
  searchKeyword: '深度学习 训练技巧 优化器 正则化',

  knowledge: [
    md`
      ## 1. 训练不顺？先看症状

      | 症状 | 常见原因 | 本课对应 |
      |---|---|---|
      | 损失变成 NaN、越训越大 | 学习率太大 | 第 2 课 + 本课第 2 节 |
      | 损失下降得很慢 | 优化器 / 学习率不合适、没有标准化 | 第 2、3 节 |
      | 深层网络完全不动（准确率≈瞎猜） | 初始化不当、梯度消失 | 第 4、5 节 |
      | 训练集很好、测试集差很多 | 过拟合 | 第 6 节 |
      | 训练集也不好 | 欠拟合：模型太小、训练不够 | 第 6 节 |

      下面每一种技巧都配了一个可以动手的实验；本课案例则在真实的 MNIST 上把它们逐个对照验证。
    `,
    md`
      ## 2. 优化器：比「沿着梯度走」更聪明

      普通 SGD 每一步只看当前的梯度。遇到「狭长山谷」（一个方向很陡、另一个方向很平），它的步长只能迁就陡的方向，于是在陡的方向来回横跳、在平的方向慢慢挪。

      - **Momentum（动量）**：像一个滚下山的球，带着「惯性」。把历史梯度累积成速度 v ← 0.9·v + g，参数沿 v 走。来回横跳的分量相互抵消，一致的前进方向越滚越快。
      - **RMSProp**：给**每个参数**单独配步长：用梯度平方的移动平均 s 去除，梯度一直很大的方向步子变小、一直很小的方向步子变大。
      - **Adam = Momentum + RMSProp**：既有惯性，又自动调节每个参数的步长，再加上「偏差修正」让开头几步不至于太小。对学习率不敏感，**默认首选，学习率 1e-3**。

      实践中：Adam（或 AdamW）几乎总能快速得到不错的结果；训练大型卷积网络时，精调过的 SGD + Momentum 有时能达到略高的最终精度。
    `,
    { widget: 'optimizer-race', intro: '四种优化器同时从同一点出发。依次试试「狭长山谷」「鞍点」「香蕉谷」三种地形，再调学习率看谁先崩。' },
    { check: {
      q: '在狭长山谷里，Momentum 比 SGD 快的根本原因是？',
      options: ['它的学习率更大', '来回震荡的梯度分量在累积时相互抵消，一致的方向不断累加', '它计算了二阶导数', '它每步用了更多数据'],
      answer: 1,
      explain: '速度 v 是历史梯度的指数加权和：横跳方向正负交替、互相抵消；沿山谷的方向梯度符号一致、不断累加，于是「越滚越快」。',
    } },
    md`
      ## 3. 学习率调度：先大步走，再小步挪

      训练初期离谷底很远，学习率大一点走得快；后期在谷底附近，大学习率会来回震荡，调小才能精确落到底。常见做法：

      - **阶梯衰减**：每隔若干轮乘以 0.1；
      - **余弦退火**：学习率沿余弦曲线从大平滑降到接近 0；
      - **预热（warm-up）**：开头几百步从很小的学习率线性升上去，避免随机初始化时的大梯度把网络「打坏」——训练 Transformer 时几乎必用。

      PyTorch 里用 \`torch.optim.lr_scheduler\`（如 \`CosineAnnealingLR\`、\`OneCycleLR\`），第 8 课训练 ResNet 时会用到。

      ## 4. 初始化：让信号顺利穿过很多层

      每一层都把上一层的输出乘以一个权重矩阵。如果权重太小，信号每过一层就缩小一点，十层后几乎为 0；太大，则每层放大，要么爆炸、要么把 tanh / sigmoid 推到饱和区。**两种情况下梯度都会消失或爆炸。**

      好的初始化要让每层输出的方差大致不变：一层有 n 个输入，每个输出是 n 项之和，所以权重方差应该和 1/n 成比例：

      - **Xavier（Glorot）初始化**：方差 ≈ 1/n（或 2/(输入数 + 输出数)），适合 tanh；
      - **He（Kaiming）初始化**：方差 = 2/n，适合 ReLU（ReLU 把一半输入置 0，所以要多乘一个 2 补回来）。
    `,
    { widget: 'init-act', intro: '依次选「太小」「太大」「Xavier」「He」，再切换激活函数。重点看最后几层的直方图有没有塌成一根柱子，以及最下面的梯度条有没有缩到几乎看不见。' },
    md`
      ## 5. BatchNorm：给每一层的输入做标准化

      第 2 课说过「输入要标准化」，因为各维量级一致时训练最顺。可网络中间每一层的输入，是上一层的输出，它的分布会随着训练不断漂移。**批归一化（Batch Normalization，2015）** 的想法很直接：在每层里也做一次标准化。

      <div class="formula">x̂ = (x − 这一批的均值) / √(这一批的方差 + ε)        y = γ·x̂ + β</div>

      - 用**当前这一批样本**的均值和方差把每个特征标准化到均值 0、方差 1；
      - 再用两个**可学习**的参数 γ、β 缩放和平移，网络需要的话还能「还原」；
      - **测试时**没有「一批」可以统计（可能只来一张图），于是改用训练过程中累积的**滑动平均**均值和方差——所以评估前一定要 \`model.eval()\`。

      效果：允许更大的学习率、对初始化不再敏感、训练快得多，还带一点正则化作用。下面是本课案例的真实实验：4 个隐藏层的 sigmoid 网络，用同样的学习率训练 1 个 epoch。
    `,
    { widget: 'py-chart', script: '06_training', title: '有无 BatchNorm：深层 sigmoid 网络的损失曲线', series: [{ path: 'history.plain', name: '没有 BatchNorm' }, { path: 'history.bn', name: '有 BatchNorm' }], xLabel: '训练步数（每 20 步记一次）', xStep: 20, x0: 20, yLabel: '训练损失' },
    md`
      ## 6. 过拟合与正则化

      **过拟合**：模型把训练数据的细节甚至噪声都「背」了下来，在训练集上近乎完美，换一批新数据就露馅。诊断方法是把数据分成三份：

      - **训练集**：用来更新参数；
      - **验证集**：用来挑超参数、决定何时停止（不参与训练）；
      - **测试集**：最后只用一次，报告真实水平。

      训练损失持续下降、验证损失却开始上升，就是过拟合的典型信号。对付它的办法（从最有效到最便宜）：

      1. **更多数据**——永远是最好的正则化；
      2. **数据增强**：对训练图片做平移、旋转、翻转、裁剪……「免费」制造新样本，同时告诉模型「这些变化不改变答案」；
      3. **权重衰减（L2 正则）**：在损失里加上 λ·Σw²，惩罚过大的权重，让函数更平滑。PyTorch 里就是优化器的 \`weight_decay\` 参数；
      4. **Dropout**：训练时随机关掉一部分神经元（见下面的组件）；
      5. **早停**：验证损失不再下降时停止训练，保留最好的那一轮；
      6. **换小一点的模型**——但太小又会欠拟合。
    `,
    { widget: 'overfit', intro: '默认是「12 个样本 + 大模型」。先点开始，看橙色曲线怎样为了穿过每个训练点而扭来扭去、验证损失怎样回升；再把样本加到 100，或者加权重衰减，或者换成小模型。' },
    { widget: 'dropout' },
    { check: {
      q: '模型里用了 Dropout，评估时忘了调用 `model.eval()`，会发生什么？',
      options: ['没有影响', '评估时仍会随机丢弃神经元，结果偏低且每次不同', '模型参数会被清空', '程序报错'],
      answer: 1,
      explain: 'Dropout 和 BatchNorm 在训练 / 评估模式下行为不同。忘了 `eval()`，Dropout 会继续随机关神经元，BN 会用当前批的统计量，评估结果既偏低又不稳定。',
    } },
    { scenario: {
      title: '调参情景模拟：你的模型出问题了',
      intro: '你在训练一个图片分类网络。每一轮选一个最合理的下一步。',
      steps: [
        { who: '🧑‍💻', text: '第一次跑：训练损失第 1 轮就变成了 `nan`。', options: [
          { label: '把学习率缩小 10 倍再试', ok: true, feedback: '对。NaN 最常见的原因是学习率太大导致发散。其次检查输入里有没有 NaN、有没有除以 0、log(0)。' },
          { label: '加 Dropout', ok: false, feedback: 'Dropout 是对付过拟合的，和数值发散无关。' },
          { label: '多训练几轮看看', ok: false, feedback: '变成 NaN 之后参数已经坏了，再训练也回不来。' },
        ] },
        { who: '🧑‍💻', text: '学习率改小后能训练了：训练准确率 99.5%，验证准确率只有 82%。', options: [
          { label: '把网络加深加宽', ok: false, feedback: '训练集已经 99.5% 了，问题不是能力不够，而是过拟合。加大模型只会更严重。' },
          { label: '加数据增强、权重衰减或 Dropout，并用验证集做早停', ok: true, feedback: '对。训练好、验证差 = 过拟合。优先考虑更多 / 更多样的数据（增强），再加正则化。' },
          { label: '把学习率调大', ok: false, feedback: '学习率主要影响能不能学、学多快，对过拟合帮助不大。' },
        ] },
        { who: '🧑‍💻', text: '又换了一个 20 层的深网络（sigmoid 激活），结果准确率一直停在 10%，损失几乎不动。', options: [
          { label: '换 ReLU + He 初始化，并加 BatchNorm', ok: true, feedback: '对。深层 sigmoid 网络梯度消失、几乎学不动。ReLU、合适的初始化和 BN 都能让梯度顺利传到前面的层。第 8 课的残差连接是更彻底的解法。' },
          { label: '加大权重衰减', ok: false, feedback: '模型连训练集都学不会，是欠拟合 / 梯度问题，再加正则化只会更糟。' },
          { label: '把批量从 64 改成 63', ok: false, feedback: '无关紧要的改动不能解决梯度消失。' },
        ] },
        { who: '🧑‍💻', text: '最后，训练集和验证集准确率都只有 70%，而且很接近。', options: [
          { label: '欠拟合：增大模型、训练更久、减少正则化', ok: true, feedback: '对。两者都低且接近，说明模型能力不够或者没训练充分。先让它在训练集上表现好，再考虑泛化。' },
          { label: '过拟合：加 Dropout', ok: false, feedback: '训练集本身就只有 70%，不存在「背答案」的问题。' },
        ] },
      ],
      end: '🎉 总结一个调参顺序：**先确保能训练（损失下降、无 NaN）→ 再让训练集表现好（模型够大、训练够久）→ 最后缩小训练与验证的差距（数据增强、正则化、早停）。** 每次只改一个因素，并记录结果。',
    } },
    md`
      ## 小结

      1. Adam（lr=1e-3）是好的起点；学习率调度能再提一截。
      2. 初始化保持各层方差稳定：tanh 用 Xavier，ReLU 用 He。
      3. BatchNorm 标准化每层的输入，训练更快更稳；评估前 \`model.eval()\`。
      4. 过拟合看「训练 vs 验证」的差距；数据（增强）第一，正则化第二，早停兜底。
    `,
  ],

  case: {
    title: 'MNIST 上的四组对照实验',
    icon: '🔬',
    summary: '每组只改一个因素：优化器、正则化、BatchNorm、初始化。用真实曲线看清每种技巧到底起什么作用。',
    script: '06_training',
    blocks: [
      md`
        **任务**：同一份 MNIST 数据、同样的随机种子，每组实验只改变一个因素，其余完全相同。这是做深度学习实验的基本功——**控制变量**，否则你永远不知道效果是哪个改动带来的。

        **本次运行的结论**：

        - 优化器（训练 2 轮）：SGD {{sgd_acc}}，SGD + Momentum {{mom_acc}}，Adam {{adam_acc}}；
        - 只用 1000 张图训练大网络：无正则化测试准确率 {{none_test}}，Dropout {{drop_test}}，权重衰减 {{wd_test}}，**数据增强 {{aug_test}}**，换成 1 万张数据 {{more_test}}；
        - 4 层 sigmoid 网络训练 1 轮：没有 BatchNorm {{plain_acc}}（相当于瞎猜），有 BatchNorm {{bn_acc}}。
      `,
      md`## 实验一：优化器`,
      { widget: 'py-chart', title: '同样的网络和初始值，只换优化器', series: [{ path: 'history.opt.0', name: 'SGD（lr=0.01）' }, { path: 'history.opt.1', name: 'SGD + Momentum（lr=0.01）' }, { path: 'history.opt.2', name: 'Adam（lr=0.001）' }], xLabel: '训练步数（每 20 步记一次平均损失）', xStep: 20, x0: 20, yLabel: '训练损失' },
      md`## 实验二：过拟合与正则化（1000 张训练图 + 66 万参数的大网络）`,
      { widget: 'py-chart', title: '测试准确率', series: [{ path: 'history.reg_test.0', name: '无正则化' }, { path: 'history.reg_test.1', name: 'Dropout 0.5' }, { path: 'history.reg_test.2', name: '权重衰减 0.01' }, { path: 'history.reg_test.3', name: '数据增强' }, { path: 'history.reg_test.4', name: '更多数据（1 万张）' }], format: 'pct', digits: 1, xLabel: '训练轮数', xStep: 3, x0: 3 },
      { widget: 'py-chart', title: '测试损失（越低越好）', series: [{ path: 'history.reg_loss.0', name: '无正则化' }, { path: 'history.reg_loss.1', name: 'Dropout 0.5' }, { path: 'history.reg_loss.2', name: '权重衰减 0.01' }, { path: 'history.reg_loss.3', name: '数据增强' }, { path: 'history.reg_loss.4', name: '更多数据（1 万张）' }], xLabel: '训练轮数', xStep: 3, x0: 3, note: '无正则化时测试损失先降后升（最低 {{none_loss_min}}，最后 {{none_loss}}）：准确率不怎么变，但模型对答错的题越来越「自信」。' },
      md`## 实验三：BatchNorm`,
      { widget: 'py-chart', title: '4 个隐藏层的 sigmoid 网络（SGD，lr=0.1）', series: [{ path: 'history.plain', name: '没有 BatchNorm' }, { path: 'history.bn', name: '有 BatchNorm' }], xLabel: '训练步数（每 20 步记一次）', xStep: 20, x0: 20, yLabel: '训练损失' },
      md`## 实验四：初始化（10 层网络，每层输出的标准差）`,
      { widget: 'py-chart', title: '每一层输出的标准差', series: [{ path: 'history.init.0', name: 'tanh + 太小' }, { path: 'history.init.1', name: 'tanh + 太大' }, { path: 'history.init.2', name: 'tanh + Xavier' }, { path: 'history.init.3', name: 'ReLU + Xavier' }, { path: 'history.init.4', name: 'ReLU + He' }], xLabel: '第几层', yLabel: '输出的标准差' },
      { code: true },
      md`
        ## 结果解读

        - **优化器**：同样 938 步，SGD 的损失还在 0.37 左右，Momentum 和 Adam 已经降到 0.11~0.13。Momentum 只是在 SGD 上加了一行「累积速度」，效果立竿见影。
        - **正则化**：五组的训练准确率都接近 100%，差别全在测试集上。Dropout 和权重衰减主要把测试损失压了下来（无正则化 {{none_loss}} → Dropout {{drop_loss}}、权重衰减 {{wd_loss}}），准确率提升有限；**数据增强**把测试准确率从 {{none_test}} 提到 {{aug_test}}，测试损失降到 {{aug_loss}}——它等于告诉模型「数字平移一点还是同一个数字」，这是真正的新知识。真有 1 万张数据时达到 {{more_test}}。
        - **BatchNorm**：没有 BN 的深层 sigmoid 网络损失停在 2.3（= ln 10，相当于对 10 个类别平均瞎猜），一个 epoch 什么都没学到；加了 BN 立刻正常下降。
        - **初始化**：太小时信号第 5 层就衰减到 0；太大时 tanh 全部饱和在 ±1（标准差≈0.97，几乎都是 ±1）；ReLU 配 Xavier 逐层衰减，配 He 则保持稳定。

        ## 思考方向

        - 实验二里把「数据增强」和「Dropout」同时打开，会更好吗？正则化手段通常可以叠加，但要配合调整训练轮数。
        - 实验一换成学习率 0.1 的 SGD 会怎样？学习率和优化器要一起调。
        - BatchNorm 放在激活函数之前还是之后？原论文放在之前，实践中两种都有人用——可以改代码对比。
        - 第 8 课的 ResNet 会把 BatchNorm、He 初始化、SGD + Momentum + 学习率调度全部用上。
      `,
    ],
    troubleshoot: [
      { sign: '训练准确率接近 100%，测试准确率低 10 个百分点以上', cause: '过拟合：训练数据太少或模型太大', fix: '优先做数据增强、收集更多数据；再加权重衰减 / Dropout；用验证集早停' },
      { sign: '验证损失先降后升，但验证准确率变化不大', cause: '模型变得过度自信（对错误答案给出很高的概率）', fix: '早停；加权重衰减、Dropout 或标签平滑（label smoothing）' },
      { sign: '深层网络损失停在 ln(类别数) 附近不动', cause: '梯度消失（sigmoid / tanh + 不合适的初始化）', fix: '换 ReLU + He 初始化；加 BatchNorm；用残差连接' },
      { sign: '加了 BatchNorm 后，评估时结果很差或不稳定', cause: '评估时没调用 `model.eval()`，或批量太小（如 1）导致统计量不准', fix: '评估前 `model.eval()`；训练批量不要太小（一般 ≥ 16）' },
      { sign: 'Adam 训练很快，但最终精度不如别人的 SGD', cause: '自适应优化器有时泛化略差；或缺少学习率调度、权重衰减方式不对', fix: '试试 AdamW（解耦的权重衰减）+ 余弦学习率；或 SGD + Momentum + 学习率调度' },
      { sign: '损失曲线锯齿很大，降得也慢', cause: '批量太小导致梯度噪声大，或学习率偏大', fix: '增大批量或减小学习率；看平滑后的曲线（每 N 步取平均）判断趋势' },
    ],
  },

  quiz: [
    { q: 'Adam 优化器可以看作哪两种方法的结合？', options: ['SGD 和牛顿法', 'Momentum 和 RMSProp', 'Dropout 和 BatchNorm', '权重衰减和早停'], answer: 1, explain: 'Adam 用梯度的一阶矩（Momentum）决定方向，用二阶矩（RMSProp）给每个参数自适应步长，再加偏差修正。' },
    { q: '一层有 256 个输入、使用 ReLU 激活，按 He 初始化，权重的标准差应约为多少？', options: ['1/256', '√(2/256) ≈ 0.088', '2/256', '1'], answer: 1, explain: 'He 初始化方差 = 2/n = 2/256，标准差 = √(2/256) ≈ 0.088。' },
    { q: '计算填空：BatchNorm 对一批数据 [1, 3, 5, 7] 做标准化（忽略 ε，方差用总体方差），数值 7 标准化后约等于多少？（保留两位小数）', answer: 1.34, tol: 0.01, explain: '均值 4，方差 = (9+1+1+9)/4 = 5，标准差 √5 ≈ 2.236。(7 − 4)/2.236 ≈ 1.34。之后还会再乘 γ、加 β。' },
    { q: '关于 BatchNorm 在测试时的行为，正确的是？', options: ['用当前这一批测试数据的均值和方差', '用训练过程中累积的滑动平均均值和方差', '不做任何标准化', '每次随机选择'], answer: 1, explain: '测试时可能只有一张图，无法统计；所以用训练中记录的 running_mean / running_var。这正是要调用 `model.eval()` 的原因之一。' },
    { q: '下面哪种现象最能说明模型过拟合了？', options: ['训练损失和验证损失都很高', '训练损失持续下降，验证损失开始上升', '训练损失为 NaN', '训练准确率和验证准确率都是 99%'], answer: 1, explain: '训练变好、验证变差，说明模型在记忆训练集的特有细节，泛化能力下降。' },
    { q: 'Dropout 在训练时以 p = 0.5 丢弃神经元，保留的神经元输出要乘以多少（倒置 Dropout）？', options: ['0.5', '1', '2', '不乘'], answer: 2, explain: '乘以 1/(1−p) = 2，保证下一层收到的信号总量期望和不丢弃时一致，这样测试时就可以什么都不做。' },
    { q: '以下哪些属于正则化手段（多选）', options: ['数据增强', '权重衰减（L2）', 'Dropout', '提高学习率'], answer: [0, 1, 2], explain: '早停、标签平滑也是常见的正则化手段。提高学习率与正则化无关。' },
    { q: '读代码：这段评估代码的问题是？\n\n```python\nmodel.train()\ncorrect = 0\nfor x, y in test_loader:\n    pred = model(x).argmax(1)\n    correct += (pred == y).sum().item()\n```', options: ['没有问题', '应该用 `model.eval()` 和 `torch.no_grad()`：否则 Dropout / BN 仍是训练行为，还会白白记录计算图', 'argmax 的维度应该是 0', '`.item()` 应该去掉'], answer: 1, explain: '评估前切到 eval 模式（关掉 Dropout、BN 用滑动平均），并在 `torch.no_grad()` 下运行以节省时间和内存。' },
  ],
});
