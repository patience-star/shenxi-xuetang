/* 互动组件：激活函数与它们的导数
 * 实线是函数本身，虚线是导数（反向传播时要乘上的「局部导数」）。拖动竖线看任意一点的值，
 * 右侧「连乘 10 层」显示：如果每层的局部导数都是这个值，梯度传过 10 层后还剩多少。 */
Widgets.activations = {
  title: '激活函数：给网络加上「弯曲」',
  render(el) {
    const s = (x) => 1 / (1 + Math.exp(-x));
    const erfA = (x) => { const t = 1 / (1 + 0.3275911 * Math.abs(x)); const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; };
    const F = {
      sigmoid: { name: 'Sigmoid', f: s, d: (x) => s(x) * (1 - s(x)), range: '(0, 1)', note: '输出像概率，适合二分类的输出层；两端饱和、导数最大只有 0.25，深层网络里容易梯度消失。' },
      tanh: { name: 'Tanh', f: Math.tanh, d: (x) => 1 - Math.tanh(x) ** 2, range: '(−1, 1)', note: '以 0 为中心，比 sigmoid 好训练；但两端同样饱和。RNN / LSTM 里常用。' },
      relu: { name: 'ReLU', f: (x) => Math.max(0, x), d: (x) => (x > 0 ? 1 : 0), range: '[0, +∞)', note: '正半轴导数恒为 1，梯度不会衰减，计算也最便宜——现代网络的默认选择。缺点：输入一直为负的神经元梯度为 0，可能「死掉」。' },
      leaky: { name: 'Leaky ReLU', f: (x) => (x > 0 ? x : 0.1 * x), d: (x) => (x > 0 ? 1 : 0.1), range: '(−∞, +∞)', note: '负半轴留一个小斜率（这里是 0.1），神经元不会彻底死掉。' },
      gelu: { name: 'GELU', f: (x) => 0.5 * x * (1 + erfA(x / Math.SQRT2)), d: (x) => 0.5 * (1 + erfA(x / Math.SQRT2)) + (x * Math.exp(-x * x / 2)) / Math.sqrt(2 * Math.PI), range: '约 (−0.17, +∞)', note: '平滑版的 ReLU，Transformer（BERT、GPT）里的标配。' },
    };
    el.innerHTML = `
      <div class="seg act-seg">${Object.entries(F).map(([k, v], i) => `<button type="button" data-k="${k}"${i ? '' : ' class="on"'}>${v.name}</button>`).join('')}</div>
      <div class="split wide-left" style="margin-top:10px">
        <canvas class="viz drag act-cv" aria-label="激活函数曲线"></canvas>
        <div><div class="kpis act-k"></div><p class="act-note small"></p></div>
      </div>
      <p class="small muted">拖动或点击图上任意位置移动竖线。实线 = f(x)，虚线 = f′(x)；浅色阴影是导数小于 0.05 的「饱和区」，梯度几乎传不过去。</p>`;
    const cv = el.querySelector('.act-cv'), W = 560, H = 300, ctx = Charts.hidpi(cv, W, H);
    let key = 'sigmoid', cx = 1;
    const X = (x) => 30 + ((x + 5) / 10) * (W - 40), Y = (y) => 20 + (1 - (y + 1.5) / 5) * (H - 40);
    function draw() {
      const fn = F[key], css = Charts.css;
      ctx.clearRect(0, 0, W, H);
      // 饱和区
      ctx.fillStyle = css('--warn-bg');
      for (let px = 30; px < W - 10; px += 2) { const x = -5 + ((px - 30) / (W - 40)) * 10; if (fn.d(x) < 0.05) ctx.fillRect(px, 10, 2, H - 30); }
      ctx.strokeStyle = css('--grid'); ctx.fillStyle = css('--chart-ink'); ctx.lineWidth = 1;
      for (let x = -4; x <= 4; x += 2) { ctx.beginPath(); ctx.moveTo(X(x), 10); ctx.lineTo(X(x), H - 20); ctx.stroke(); ctx.fillText(String(x), X(x) - 4, H - 6); }
      for (let y = -1; y <= 3; y++) { ctx.beginPath(); ctx.moveTo(30, Y(y)); ctx.lineTo(W - 10, Y(y)); ctx.stroke(); ctx.fillText(String(y), 10, Y(y) + 4); }
      ctx.strokeStyle = css('--axis'); ctx.beginPath(); ctx.moveTo(30, Y(0)); ctx.lineTo(W - 10, Y(0)); ctx.moveTo(X(0), 10); ctx.lineTo(X(0), H - 20); ctx.stroke();
      const curve = (g, color, dash) => {
        ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.setLineDash(dash); ctx.beginPath();
        for (let i = 0; i <= 300; i++) { const x = -5 + i / 30, y = Math.max(-1.5, Math.min(3.5, g(x))); i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)); }
        ctx.stroke(); ctx.setLineDash([]);
      };
      curve(fn.f, css('--c1'), []);
      curve(fn.d, css('--c2'), [7, 5]);
      ctx.strokeStyle = css('--text'); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(cx), 10); ctx.lineTo(X(cx), H - 20); ctx.stroke();
      [[fn.f(cx), css('--c1')], [fn.d(cx), css('--c2')]].forEach(([v, c]) => { ctx.fillStyle = c; ctx.beginPath(); ctx.arc(X(cx), Y(Math.max(-1.5, Math.min(3.5, v))), 5, 0, Math.PI * 2); ctx.fill(); });
      const d = fn.d(cx), p10 = d ** 10;
      const kpi = (l, v, sub) => `<div class="kpi"><div class="label">${l}</div><div class="value">${v}</div>${sub ? `<div class="sub">${sub}</div>` : ''}</div>`;
      el.querySelector('.act-k').innerHTML = kpi('x', cx.toFixed(2)) + kpi(`f(x)`, fn.f(cx).toFixed(3), `输出范围 ${fn.range}`) + kpi('导数 f′(x)', d.toFixed(3)) + kpi('连乘 10 层', p10 < 1e-4 ? p10.toExponential(1) : p10.toFixed(4), '梯度还剩多少');
      el.querySelector('.act-note').textContent = fn.note;
    }
    el.querySelector('.act-seg').addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      key = b.dataset.k;
      el.querySelectorAll('.act-seg button').forEach((x) => x.classList.toggle('on', x === b));
      draw();
    });
    let dragging = false;
    const move = (e) => { const [x] = Charts.at(cv, e, W, H); cx = Math.max(-5, Math.min(5, -5 + ((x - 30) / (W - 40)) * 10)); draw(); };
    cv.addEventListener('pointerdown', (e) => { dragging = true; cv.setPointerCapture(e.pointerId); move(e); });
    cv.addEventListener('pointermove', (e) => { if (dragging) move(e); });
    cv.addEventListener('pointerup', () => { dragging = false; });
    draw();
  },
};
