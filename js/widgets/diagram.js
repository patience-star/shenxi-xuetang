/* 互动组件：结构示意图（SVG，颜色跟随主题，标注全中文）
 * 用法：{ widget: 'diagram', kind: 'neuron' }。kind 见下面 KINDS 的键。
 * 新增一种图：在 KINDS 里加一个函数，返回 { w, h, svg: '…' }，用 box / circ / arrow / txt 这几个小工具拼出来。 */
(function () {
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
  const txt = (x, y, s, cls = '', anchor = 'middle') => `<text x="${x}" y="${y}" text-anchor="${anchor}" class="${cls}">${esc(s)}</text>`;
  /** 圆角框：标题 + 可选小字 */
  const box = (x, y, w, h, label, sub = '', cls = '') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="9" class="box ${cls}"/>`
    + txt(x + w / 2, y + h / 2 + (sub ? -3 : 5), label) + (sub ? txt(x + w / 2, y + h / 2 + 14, sub, 'sub') : '');
  const circ = (x, y, r, label, cls = 'node', sub = '') => `<circle cx="${x}" cy="${y}" r="${r}" class="${cls}"/>` + txt(x, y + 5, label) + (sub ? txt(x, y + r + 15, sub, 'sub') : '');
  /** 箭头：从 (x1,y1) 指向 (x2,y2)；label 写在中点上方 */
  function arrow(x1, y1, x2, y2, cls = '', label = '', dy = -6) {
    const a = Math.atan2(y2 - y1, x2 - x1), L = 8;
    const p = (ang) => `${x2 - L * Math.cos(a - ang)},${y2 - L * Math.sin(a - ang)}`;
    return `<line x1="${x1}" y1="${y1}" x2="${x2 - 4 * Math.cos(a)}" y2="${y2 - 4 * Math.sin(a)}" class="arrow ${cls}"/><polygon points="${x2},${y2} ${p(0.42)} ${p(-0.42)}" class="head ${cls}"/>`
      + (label ? txt((x1 + x2) / 2, (y1 + y2) / 2 + dy, label, 'sub') : '');
  }
  /** 折线箭头：经过若干点 */
  function path(pts, cls = '') {
    const d = pts.map((p, i) => `${i ? 'L' : 'M'}${p[0]},${p[1]}`).join('');
    const [x1, y1] = pts[pts.length - 2], [x2, y2] = pts[pts.length - 1];
    return `<path d="${d}" class="arrow ${cls}"/>` + arrow(x1, y1, x2, y2, cls).replace(/<line[^>]*\/>/, '');
  }

  const KINDS = {
    /** 一个神经元（感知机）：输入 × 权重 → 求和 + 偏置 → 激活函数 → 输出 */
    neuron: () => {
      let s = '';
      [['x₁', 50], ['x₂', 115], ['x₃', 180]].forEach(([l, y], i) => {
        s += circ(60, y, 22, l, 'node');
        s += arrow(84, y, 268, 115 + (i - 1) * 12, 'hl', `w${'₁₂₃'[i]}`, i === 1 ? -8 : i === 0 ? -4 : 16);
      });
      s += circ(300, 115, 32, 'Σ', 'node') + txt(300, 172, '加权求和', 'sub') + txt(300, 188, 'z = w·x + b', 'sub');
      s += arrow(300, 40, 300, 81, '', '') + txt(300, 30, '偏置 b', 'sub');
      s += arrow(334, 115, 420, 115);
      s += box(420, 85, 120, 60, '激活函数', '阶跃 / sigmoid / ReLU', 'alt');
      s += arrow(540, 115, 620, 115);
      s += circ(645, 115, 22, 'y', 'node', '输出');
      s += txt(60, 225, '输入', 'sub');
      return { w: 700, h: 240, svg: s };
    },
    /** 训练循环：前向 → 损失 → 反向 → 更新，周而复始 */
    'training-loop': () => {
      let s = '';
      s += box(20, 95, 130, 60, '一批数据', 'x 与真实标签 y');
      s += arrow(150, 125, 205, 125);
      s += box(205, 95, 150, 60, '模型（前向）', '用参数算出预测 ŷ', 'hl');
      s += arrow(355, 125, 410, 125);
      s += box(410, 95, 150, 60, '损失函数', '比较 ŷ 和 y → 一个数', 'alt');
      s += path([[485, 155], [485, 215], [430, 215]], 'alt');
      s += box(280, 190, 150, 52, '反向传播', '求每个参数的梯度', 'alt');
      s += path([[280, 216], [180, 216]], 'alt');
      s += box(30, 190, 150, 52, '优化器', '参数 −= 学习率 × 梯度', 'ok');
      s += path([[105, 190], [105, 170], [280, 170], [280, 157]], 'hl');
      s += txt(395, 40, '重复成千上万次：每次看一批数据、改一点点参数', 'sub');
      s += txt(395, 62, '① 前向  ② 算损失  ③ 反向  ④ 更新', '');
      return { w: 590, h: 260, svg: s };
    },
    /** 多层感知机：784 → 128 → 64 → 10 */
    mlp: () => {
      let s = '';
      const cols = [[60, '784', '输入：展平的像素'], [230, '128', '隐藏层 1（ReLU）'], [400, '64', '隐藏层 2（ReLU）'], [570, '10', '输出：10 个得分']];
      const ys = [50, 85, 120, 175, 210];
      cols.forEach(([x, n, label], c) => {
        ys.forEach((y, i) => { if (i === 3) s += txt(x, 152, '⋮', ''); else s += `<circle cx="${x}" cy="${i === 4 ? 205 : y}" r="13" class="node"/>`; });
        s += txt(x, 245, `${n} 个`, '') + txt(x, 263, label, 'sub');
        if (c < 3) {
          const nx = cols[c + 1][0];
          [50, 85, 120, 205].forEach((a) => [50, 85, 120, 205].forEach((b) => { s += `<line x1="${x + 13}" y1="${a}" x2="${nx - 13}" y2="${b}" class="arrow" style="stroke-width:.8;opacity:.6"/>`; }));
          s += txt((x + nx) / 2, 26, ['全连接 784×128', '全连接 128×64', '全连接 64×10'][c], 'sub');
        }
      });
      s += txt(680, 120, '→ softmax', 'sub') + txt(680, 138, '→ 10 个概率', 'sub');
      return { w: 740, h: 280, svg: s };
    },
    /** LeNet-5：卷积 → 池化 → 卷积 → 池化 → 全连接 × 3 */
    lenet: () => {
      let s = '';
      const B = [['输入', '1×28×28', ''], ['卷积 5×5', '6×28×28', 'hl'], ['池化 2×2', '6×14×14', 'alt'], ['卷积 5×5', '16×10×10', 'hl'], ['池化 2×2', '16×5×5', 'alt'], ['展平', '400', ''], ['全连接', '120', 'ok'], ['全连接', '84', 'ok'], ['输出', '10', 'ok']];
      B.forEach(([a, b, cls], i) => {
        const x = 10 + i * 84;
        s += box(x, 60, 72, 64, a, b, cls);
        if (i < B.length - 1) s += arrow(x + 72, 92, x + 84, 92);
      });
      s += txt(180, 40, '特征提取：卷积找局部图案，池化缩小尺寸', 'sub') + txt(590, 40, '分类：和 MLP 一样', 'sub');
      s += txt(380, 160, '每层卷积后接 ReLU；参数一共约 6.2 万个（第 5 课的 MLP 有 10.9 万个）', 'sub');
      return { w: 770, h: 180, svg: s };
    },
    /** 普通块 vs 残差块 */
    resblock: () => {
      let s = '';
      const col = (x0, title, res) => {
        let t = txt(x0 + 90, 22, title, '');
        t += circ(x0 + 90, 52, 16, 'x', 'node');
        const steps = [['3×3 卷积', 'hl'], ['BN + ReLU', 'alt'], ['3×3 卷积', 'hl'], ['BN', 'alt']];
        steps.forEach(([l, c], i) => { t += arrow(x0 + 90, i ? 106 + (i - 1) * 50 + 34 : 68, x0 + 90, 106 + i * 50 - 2) + box(x0 + 30, 106 + i * 50, 120, 34, l, '', c); });
        if (res) {
          t += arrow(x0 + 90, 290, x0 + 90, 316) + circ(x0 + 90, 332, 14, '+', 'node');
          t += path([[x0 + 106, 52], [x0 + 175, 52], [x0 + 175, 332], [x0 + 105, 332]], 'hl');
          t += txt(x0 + 182, 196, '捷径', 'sub', 'start') + txt(x0 + 182, 212, '（恒等）', 'sub', 'start');
          t += arrow(x0 + 90, 346, x0 + 90, 372) + box(x0 + 40, 372, 100, 30, 'ReLU', '', 'alt') + txt(x0 + 90, 424, '输出 = ReLU( F(x) + x )', 'sub');
        } else {
          t += arrow(x0 + 90, 290, x0 + 90, 372) + box(x0 + 40, 372, 100, 30, 'ReLU', '', 'alt') + txt(x0 + 90, 424, '输出 = ReLU( F(x) )', 'sub');
        }
        return t;
      };
      s += col(10, '普通块：学 H(x)', false) + col(300, '残差块：学 F(x) = H(x) − x', true);
      return { w: 560, h: 440, svg: s };
    },
    /** RNN 按时间展开：同一个单元在每一步重复使用 */
    rnn: () => {
      let s = '';
      const xs = ['床', '前', '明', '月'], ys = ['前', '明', '月', '光'];
      xs.forEach((x, t) => {
        const cx = 120 + t * 150;
        s += txt(cx, 232, `输入 x${t + 1}「${x}」`, 'sub') + arrow(cx, 216, cx, 172);
        s += box(cx - 45, 112, 90, 58, 'RNN 单元', '同一组 W', 'hl');
        s += arrow(cx, 112, cx, 70) + txt(cx, 56, `预测「${ys[t]}」`, 'sub');
        if (t < xs.length - 1) s += arrow(cx + 45, 141, cx + 105, 141, 'alt', `h${t + 1}`, -8);
      });
      s += arrow(15, 141, 75, 141, 'alt', 'h0', -8);
      s += txt(370, 20, '把一个 RNN 单元沿时间「展开」：隐状态 h 从左往右传，记住前文', 'sub');
      return { w: 700, h: 250, svg: s };
    },
    /** Transformer 编码器-解码器总览 */
    transformer: () => {
      let s = '';
      s += box(30, 300, 200, 40, '输入：November 9, 2026', '', '') + arrow(130, 300, 130, 270);
      s += box(30, 230, 200, 40, '字符嵌入 + 位置编码', '', 'alt') + arrow(130, 230, 130, 200);
      s += `<rect x="20" y="70" width="220" height="130" rx="12" class="box" style="fill:none;stroke-dasharray:6 4"/>` + txt(130, 88, '编码器 × 2 层', 'sub');
      s += box(40, 100, 180, 36, '多头自注意力', '', 'hl') + box(40, 150, 180, 36, '前馈网络', '', 'ok');
      s += path([[130, 70], [130, 40], [330, 40], [330, 140], [355, 140]], 'hl') + txt(240, 32, 'memory：编码后的输入', 'sub');
      s += box(400, 300, 220, 40, '已写出的：<s> 2 0 2 6 -', '', '') + arrow(510, 300, 510, 270);
      s += box(400, 230, 220, 40, '字符嵌入 + 位置编码', '', 'alt') + arrow(510, 230, 510, 210);
      s += `<rect x="360" y="40" width="300" height="170" rx="12" class="box" style="fill:none;stroke-dasharray:6 4"/>` + txt(510, 58, '解码器 × 2 层', 'sub');
      s += box(400, 160, 220, 34, '带掩码的自注意力', '不偷看后面', 'hl') + box(400, 115, 220, 34, '交叉注意力', '去输入里找信息', 'hl') + box(400, 70, 220, 34, '前馈网络', '', 'ok');
      s += arrow(660, 87, 700, 87) + box(700, 62, 110, 50, '线性 + softmax', '', 'alt') + arrow(755, 62, 755, 30) + txt(755, 22, '下一个字符「1」', '');
      s += txt(420, 372, '每个子层外面都有残差连接（第 8 课）和 LayerNorm', 'sub');
      return { w: 830, h: 385, svg: s };
    },
    /** 自编码器与 VAE */
    vae: () => {
      let s = '';
      s += box(10, 100, 90, 50, '输入图片', '784 个像素') + arrow(100, 125, 130, 125);
      s += box(130, 95, 110, 60, '编码器', '784→128', 'hl') + arrow(240, 110, 290, 85) + arrow(240, 140, 290, 165);
      s += box(290, 60, 90, 44, '均值 μ', '2 维', 'alt') + box(290, 146, 90, 44, '方差 σ²', '2 维', 'alt');
      s += arrow(380, 82, 430, 115) + arrow(380, 168, 430, 135);
      s += circ(455, 125, 24, 'z', 'node') + txt(455, 172, 'z = μ + σ·ε', 'sub') + txt(455, 188, 'ε ~ N(0, 1)', 'sub');
      s += arrow(479, 125, 520, 125) + box(520, 95, 110, 60, '解码器', '128→784', 'ok') + arrow(630, 125, 660, 125);
      s += box(660, 100, 90, 50, '重建图片', '784 个像素');
      s += txt(380, 26, '损失 = 重建误差（像不像原图）+ KL 散度（每个分布要靠近标准正态）', 'sub');
      s += txt(220, 232, '普通自编码器：编码器直接输出一个点 z，没有方差、不采样，也没有 KL 项', 'sub', 'start');
      return { w: 770, h: 250, svg: s };
    },
    /** GAN：生成器与判别器的博弈 */
    gan: () => {
      let s = '';
      s += box(10, 40, 100, 50, '随机噪声 z', '32 个数') + arrow(110, 65, 150, 65);
      s += box(150, 35, 120, 60, '生成器 G', '噪声 → 假图', 'hl') + arrow(270, 65, 320, 100);
      s += box(10, 150, 100, 50, '真实图片', 'MNIST') + arrow(110, 175, 320, 130);
      s += box(320, 85, 130, 60, '判别器 D', '图 → 真的概率', 'alt') + arrow(450, 115, 500, 115);
      s += box(500, 90, 150, 50, '真？假？', 'D 的输出 0~1', '');
      s += txt(330, 230, 'D 的目标：真图判 1、假图判 0　　G 的目标：让 D 把假图判成 1', 'sub');
      s += path([[575, 140], [575, 205], [210, 205], [210, 97]], 'alt') + txt(395, 198, '梯度穿过 D 传回 G：告诉 G「怎样画更像真的」', 'sub');
      return { w: 670, h: 245, svg: s };
    },
    /** 两层感知机解异或 */
    'xor-net': () => {
      let s = '';
      s += circ(60, 70, 22, 'x₁') + circ(60, 170, 22, 'x₂');
      s += circ(260, 70, 30, '或', 'node') + circ(260, 170, 30, '与非', 'node');
      [[70, 70], [70, 170], [170, 70], [170, 170]].forEach(([a, b]) => { s += arrow(84, a, 228, b > 100 ? b - 4 : b + 4, 'hl'); });
      s += circ(460, 120, 30, '与', 'node');
      s += arrow(290, 74, 429, 114, 'hl') + arrow(290, 166, 429, 126, 'hl');
      s += arrow(490, 120, 560, 120) + circ(585, 120, 22, 'y', 'node', '异或');
      s += txt(60, 225, '输入层', 'sub') + txt(260, 225, '隐藏层（两条直线）', 'sub') + txt(460, 225, '输出层（取交集）', 'sub');
      return { w: 640, h: 240, svg: s };
    },
  };

  Widgets.diagram = {
    title: '结构示意图',
    KINDS,
    render(el, opts) {
      const k = KINDS[opts.kind];
      if (!k) throw new Error(`没有这种示意图：${opts.kind}`);
      const { w, h, svg } = k(opts);
      el.innerHTML = `<div class="diagram"><svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(opts.title || '结构示意图')}">${svg}</svg></div>${opts.caption ? `<p class="small muted">${App.mdInline(opts.caption)}</p>` : ''}`;
    },
    helpers: { box, circ, arrow, txt, path },
  };
})();
