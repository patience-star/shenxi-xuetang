/* 互动组件：经典卷积网络的演进（LeNet → AlexNet → VGG → GoogLeNet → ResNet）
 * 点时间线上的模型看它的规模和关键创新；下面的柱状图是 ImageNet 大规模视觉识别挑战赛（ILSVRC）的 top-5 错误率。 */
Widgets['cnn-arch'] = {
  title: '经典 CNN 进化史',
  render(el) {
    const M = [
      { name: 'LeNet-5', year: 1998, layers: '5 层（2 卷积 + 3 全连接）', params: '约 6 万', err: null, data: '手写数字（MNIST）',
        ideas: ['确立了「卷积 → 池化 → 卷积 → 池化 → 全连接」的基本结构', '局部连接 + 权重共享：参数少、对平移不敏感', '曾用于美国银行的支票数字识别'] },
      { name: 'AlexNet', year: 2012, layers: '8 层（5 卷积 + 3 全连接）', params: '约 6000 万', err: 15.3, data: 'ImageNet（120 万张、1000 类）',
        ideas: ['用 ReLU 代替 sigmoid / tanh，训练快得多', 'Dropout 抑制全连接层的过拟合；大量数据增强', '两块 GPU 并行训练', '错误率比第二名（26.2%）低了约 11 个百分点，引爆了深度学习浪潮'] },
      { name: 'VGG-16', year: 2014, layers: '16 层（13 卷积 + 3 全连接）', params: '约 1.38 亿', err: 7.3, data: 'ImageNet',
        ideas: ['全部使用 3×3 小卷积核：两个 3×3 叠起来的感受野等于一个 5×5，参数更少、非线性更多', '结构极其规整：卷积块 + 池化，层数加深、通道翻倍', '参数量大（主要在全连接层），但至今仍常被用作特征提取器'] },
      { name: 'GoogLeNet', year: 2014, layers: '22 层', params: '约 500~700 万', err: 6.67, data: 'ImageNet',
        ideas: ['Inception 模块：1×1、3×3、5×5 卷积和池化并行，再拼接起来，让网络自己选尺度', '大量使用 1×1 卷积压缩通道数，降低计算量', '用全局平均池化代替巨大的全连接层：参数只有 AlexNet 的约十分之一'] },
      { name: 'ResNet', year: 2015, layers: '最深 152 层', params: 'ResNet-50 约 2560 万；ResNet-152 约 6000 万', err: 3.57, data: 'ImageNet',
        ideas: ['残差连接：每个块学习「在输入上加点什么」，上百层的网络也能稳定训练', 'BatchNorm 成为标配；去掉了大全连接层', '错误率首次低于人类的估计水平（约 5.1%）；残差思想后来被 Transformer 等几乎所有深度模型继承'] },
    ];
    el.innerHTML = `
      <div class="seg ca-seg">${M.map((m, i) => `<button type="button" data-i="${i}"${i === 1 ? ' class="on"' : ''}>${m.year} ${m.name}</button>`).join('')}</div>
      <div class="ca-card card" style="margin-top:12px;box-shadow:none"></div>
      <h4>ImageNet top-5 错误率（越低越好）</h4>
      <div class="ca-chart"></div>
      <p class="small muted">top-5 错误率：模型给出的 5 个最可能的类别里都没有正确答案的比例。人类水平约 5.1% 来自 Karpathy 的估计。数字来自各年 ILSVRC 比赛与对应论文。</p>`;
    const show = (i) => {
      const m = M[i];
      el.querySelectorAll('.ca-seg button').forEach((b, k) => b.classList.toggle('on', k === i));
      el.querySelector('.ca-card').innerHTML = `<h3 style="margin-top:0">${m.name}（${m.year}）</h3>
        <div class="kpis"><div class="kpi"><div class="label">深度</div><div class="value" style="font-size:1rem">${m.layers}</div></div>
          <div class="kpi"><div class="label">参数量</div><div class="value" style="font-size:1rem">${m.params}</div></div>
          <div class="kpi"><div class="label">数据集</div><div class="value" style="font-size:1rem">${m.data}</div></div>
          <div class="kpi"><div class="label">top-5 错误率</div><div class="value" style="font-size:1rem">${m.err ? m.err + '%' : '—'}</div></div></div>
        <ul>${m.ideas.map((t) => `<li>${t}</li>`).join('')}</ul>`;
    };
    el.querySelector('.ca-seg').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) show(+b.dataset.i); });
    show(1);
    Charts.columns(el.querySelector('.ca-chart'), [...M.filter((m) => m.err).map((m) => ({ label: `${m.name}（${m.year}）`, value: m.err / 100 })), { label: '人类（估计）', value: 0.051, color: 'var(--deemph)' }],
      { format: (v) => Charts.fmt.pct(v, 1), height: 240, yLabel: '错误率' });
  },
};
