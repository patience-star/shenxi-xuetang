/* 互动组件：手写数字识别
 * 在左边写一个数字（鼠标或手指），模型实时给出 0~9 的概率；下面展示模型内部：
 *   05_mlp_mnist  ：两个隐藏层的激活值，以及被激活最强的隐藏神经元的「权重模板」
 *   07_lenet_mnist：两个卷积层输出的特征图，以及第一层的卷积核
 * 参数：model（脚本名，在案例里可省略，默认用本课脚本）。 */
(function () {
  const INSIDE = {};

  /** 多层感知机：隐藏层激活 + 权重模板 */
  INSIDE['05_mlp_mnist'] = {
    html: `<h4>模型内部：隐藏层在想什么</h4>
      <p class="small muted">第 1 隐藏层 128 个神经元、第 2 隐藏层 64 个神经元的输出（ReLU 之后，颜色越深越活跃；灰色 = 0，没被激活）：</p>
      <div class="dp-h1 vec"></div><div class="dp-h2 vec" style="margin-top:6px"></div>
      <p class="small muted" style="margin-top:12px">第 1 层里<b>被激活得最强的 8 个神经元</b>，以及它们的 784 个权重排回 28×28 的样子（蓝 = 这里有笔画会加分，红 = 这里有笔画会扣分）。每个神经元就像一个「模板」：输入和它越像，激活越大。</p>
      <div class="thumbs dp-tpl"></div>`,
    draw(box, W, out) {
      const strip = (el, v) => {
        const m = Math.max(1e-6, ...v);
        el.innerHTML = Array.from(v, (x) => `<i title="${x.toFixed(2)}" style="background:${x > 0 ? Charts.seq(0.15 + 0.85 * (x / m)).bg : 'var(--surface-2)'}"></i>`).join('');
      };
      strip(box.querySelector('.dp-h1'), out.h1);
      strip(box.querySelector('.dp-h2'), out.h2);
      const top = NN.topk(out.h1, 8), w1 = W['fc1.weight'].d, tpl = box.querySelector('.dp-tpl');
      tpl.innerHTML = '';
      top.forEach((j) => {
        const f = document.createElement('figure');
        f.append(Charts.canvasOf(w1.subarray(j * 784, (j + 1) * 784), 28, 28, { mode: 'div' }), Object.assign(document.createElement('figcaption'), { textContent: `#${j} 激活 ${out.h1[j].toFixed(1)}` }));
        tpl.appendChild(f);
      });
    },
  };

  /** LeNet：卷积核 + 两层特征图 */
  INSIDE['07_lenet_mnist'] = {
    html: `<h4>模型内部：卷积层看到了什么</h4>
      <p class="small muted">第 1 层的 6 个 5×5 卷积核（蓝 = 正权重，红 = 负权重；正负相邻处就是在找边缘）：</p>
      <div class="thumbs dp-k" style="grid-template-columns:repeat(6,minmax(40px,64px))"></div>
      <p class="small muted" style="margin-top:10px">每个卷积核在整张图上滑一遍，得到一张 28×28 的<b>特征图</b>（ReLU 之后，越深表示这里越「像」卷积核要找的图案）：</p>
      <div class="thumbs dp-c1" style="grid-template-columns:repeat(6,minmax(48px,90px))"></div>
      <p class="small muted" style="margin-top:10px">第 2 层的 16 张 10×10 特征图：每张都综合了第 1 层全部 6 张特征图，看的是更大范围、更抽象的组合（拐角、圆弧、交叉……）：</p>
      <div class="thumbs dp-c2" style="grid-template-columns:repeat(8,minmax(36px,60px))"></div>`,
    draw(box, W, out) {
      const k = W['conv1.weight'].d, put = (sel, n, size, src, mode) => {
        const el = box.querySelector(sel);
        if (el.children.length !== n) { el.innerHTML = ''; for (let i = 0; i < n; i++) { const c = document.createElement('canvas'); c.className = 'thumb'; el.appendChild(c); } }
        for (let i = 0; i < n; i++) Charts.img(el.children[i], src.subarray(i * size * size, (i + 1) * size * size), size, size, { mode });
      };
      if (!box.querySelector('.dp-k').children.length) put('.dp-k', 6, 5, k, 'div');
      put('.dp-c1', 6, 28, out.c1.d, 'seq');
      put('.dp-c2', 16, 10, out.c2.d, 'seq');
    },
  };

  Widgets['digit-pad'] = {
    title: '手写数字识别：写一个数字试试',
    INSIDE,
    async render(el, opts) {
      const name = opts.model || opts.py?.name;
      const py = opts.py?.name === name ? opts.py : await App.loadPy(name);
      const W = NN.weights(py), M = NN.models[name], ins = INSIDE[name];
      const demoX = NN.decode(py.extras.demo_x), demoY = py.extras.demo_y;
      el.innerHTML = `
        <div class="split">
          <div>
            <div class="dp-pad"></div>
            <div class="row" style="margin-top:8px"><button type="button" class="btn sm" data-a="clear">清除</button><button type="button" class="btn sm ghost" data-a="sample">随机一张测试图</button></div>
            <p class="small muted">提示：数字写大一点、写在中间，线条粗细适中。</p>
          </div>
          <div>
            <div class="row" style="align-items:flex-end;gap:16px">
              <div><div class="small muted">模型看到的 28×28</div><canvas class="dp-28 big-thumb" style="max-width:96px"></canvas></div>
              <div><div class="small muted">预测</div><div class="dp-pred" style="font-size:3rem;font-weight:700;line-height:1.1">?</div><div class="dp-conf small muted"></div></div>
            </div>
            <div class="dp-bars" style="margin-top:10px"></div>
          </div>
        </div>
        <div class="dp-inside">${ins ? ins.html : ''}</div>`;
      const c28 = el.querySelector('.dp-28');
      let fromSample = false;
      const pad = Charts.pad(el.querySelector('.dp-pad'), {
        onStart: () => { if (fromSample) { fromSample = false; pad.clear(); } },
        onChange: (x) => predict(x),
      });
      function predict(x, label) {
        Charts.img(c28, x, 28, 28, { mode: 'ink', min: 0, max: 1 });
        if (label == null && pad.isEmpty()) {
          el.querySelector('.dp-pred').textContent = '?';
          el.querySelector('.dp-conf').textContent = '在左边写一个 0~9 的数字';
          Charts.bars(el.querySelector('.dp-bars'), Array.from({ length: 10 }, (_, d) => ({ label: String(d), p: 0 })));
          return;
        }
        const out = M.forward(W, x), k = NN.argmax(out.probs);
        el.querySelector('.dp-pred').textContent = k;
        el.querySelector('.dp-conf').textContent = `置信度 ${(out.probs[k] * 100).toFixed(1)}%${label != null ? `（这张测试图的真实标签：${label}${label === k ? ' ✓' : ' ✗'}）` : ''}`;
        Charts.bars(el.querySelector('.dp-bars'), Array.from(out.probs, (p, d) => ({ label: String(d), p })), { hi: k });
        if (ins) ins.draw(el.querySelector('.dp-inside'), W, out);
      }
      el.addEventListener('click', (e) => {
        const a = e.target.closest('[data-a]')?.dataset.a;
        if (a === 'clear') { fromSample = false; pad.clear(); }
        if (a === 'sample') {
          const i = Math.floor(Math.random() * demoY.length);
          const x = Float32Array.from(demoX.d.subarray(i * 784, (i + 1) * 784), (v) => v / 255);
          pad.show28(x);
          fromSample = true;
          predict(x, demoY[i]);
        }
      });
      predict(new Float32Array(784));
    },
  };
})();
