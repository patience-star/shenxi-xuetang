/* 互动组件：一个神经元学「摄氏 → 华氏」
 * 数据：pytorch/02_gradient.py 里的 40 个带测量误差的温度。在浏览器里实时做梯度下降，
 * 对比「原始输入」和「标准化输入」：前者 b 爬得极慢，后者几十步就学会。 */
Widgets['neuron-fit'] = {
  title: '一个神经元学「摄氏 → 华氏」',
  async render(el, opts) {
    const py = opts.py || (await App.loadPy(opts.script || '02_gradient'));
    const C = py.extras.C, Fv = py.extras.F, n = C.length;
    const mu = C.reduce((a, b) => a + b, 0) / n;
    const sd = Math.sqrt(C.reduce((a, b) => a + (b - mu) ** 2, 0) / (n - 1));
    el.innerHTML = `
      <div class="controls">
        <label class="inline"><input type="checkbox" class="nf-norm"> 标准化输入（把摄氏度变成均值 0、标准差 1）</label>
        <label>学习率 η = <output class="nf-lrv"></output><input type="range" class="nf-lr" min="0" max="100" step="1"></label>
        <span class="row"><button type="button" class="btn sm" data-a="10">训练 10 步</button><button type="button" class="btn sm ghost" data-a="run">▶ 自动</button><button type="button" class="btn sm ghost" data-a="reset">重置</button></span>
      </div>
      <div class="split wide-left">
        <canvas class="viz nf-cv" aria-label="数据点与模型的直线"></canvas>
        <div><div class="kpis nf-k"></div><canvas class="viz nf-loss" aria-label="损失曲线"></canvas><p class="small muted">损失曲线（对数刻度）</p></div>
      </div>
      <p class="verdict nf-v"></p>`;
    const normBox = el.querySelector('.nf-norm'), lrIn = el.querySelector('.nf-lr'), lrOut = el.querySelector('.nf-lrv');
    const cv = el.querySelector('.nf-cv'), lv = el.querySelector('.nf-loss'), runBtn = el.querySelector('[data-a=run]');
    const W = 520, H = 320, ctx = Charts.hidpi(cv, W, H), LW = 300, LH = 150, lctx = Charts.hidpi(lv, LW, LH);
    // 滑块 0~100 映射到对数刻度的学习率：原始输入 1e-5 ~ 5e-4，标准化 0.01 ~ 1.2
    const lrOf = () => { const t = +lrIn.value / 100; return normBox.checked ? 0.01 * 120 ** t : 1e-5 * 50 ** t; };
    let w, b, losses, steps;
    const xs = () => (normBox.checked ? C.map((c) => (c - mu) / sd) : C);
    const loss = () => { const x = xs(); let s = 0; for (let i = 0; i < n; i++) s += (w * x[i] + b - Fv[i]) ** 2; return s / n; };
    // 换算回原始单位：F = w'·(C − μ)/σ + b' = (w'/σ)·C + (b' − w'·μ/σ)
    const realWB = () => (normBox.checked ? [w / sd, b - (w * mu) / sd] : [w, b]);
    function reset() { w = 0; b = 0; steps = 0; losses = [loss()]; draw(); }
    function train(k) {
      const x = xs(), lr = lrOf();
      for (let s = 0; s < k; s++) {
        let gw = 0, gb = 0;
        for (let i = 0; i < n; i++) { const e = w * x[i] + b - Fv[i]; gw += (2 * e * x[i]) / n; gb += (2 * e) / n; }
        w -= lr * gw; b -= lr * gb; steps++;
        const L = loss();
        losses.push(L);
        if (!Number.isFinite(L) || L > 1e12) break;
      }
      draw();
    }
    function draw() {
      const css = Charts.css, m = { l: 44, r: 12, t: 12, b: 30 };
      const X = (c) => m.l + ((c + 45) / 150) * (W - m.l - m.r), Y = (f) => m.t + (1 - (f + 60) / 290) * (H - m.t - m.b);
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = css('--grid'); ctx.fillStyle = css('--chart-ink'); ctx.lineWidth = 1;
      for (let c = -40; c <= 100; c += 20) { ctx.beginPath(); ctx.moveTo(X(c), m.t); ctx.lineTo(X(c), H - m.b); ctx.stroke(); ctx.fillText(`${c}`, X(c) - 8, H - 12); }
      for (let f = -50; f <= 225; f += 50) { ctx.beginPath(); ctx.moveTo(m.l, Y(f)); ctx.lineTo(W - m.r, Y(f)); ctx.stroke(); ctx.fillText(`${f}`, 4, Y(f) + 4); }
      ctx.fillText('摄氏度 °C →', W - 92, H - 2); ctx.fillText('华氏度 °F', m.l + 4, m.t + 10);
      ctx.fillStyle = css('--c1');
      C.forEach((c, i) => { ctx.beginPath(); ctx.arc(X(c), Y(Fv[i]), 3.5, 0, Math.PI * 2); ctx.fill(); });
      const [rw, rb] = realWB();
      if (Number.isFinite(rw) && Number.isFinite(rb)) {
        ctx.strokeStyle = css('--c2'); ctx.lineWidth = 2.5; ctx.beginPath();
        ctx.moveTo(X(-45), Y(rw * -45 + rb)); ctx.lineTo(X(105), Y(rw * 105 + rb)); ctx.save(); ctx.rect(m.l, m.t, W - m.l - m.r, H - m.t - m.b); ctx.clip(); ctx.stroke(); ctx.restore();
      }
      const L = losses[losses.length - 1];
      const kpi = (label, v, sub) => `<div class="kpi"><div class="label">${label}</div><div class="value">${v}</div><div class="sub">${sub}</div></div>`;
      el.querySelector('.nf-k').innerHTML = kpi('w（斜率）', Number.isFinite(rw) ? rw.toFixed(3) : '∞', '真值 1.8') + kpi('b（截距）', Number.isFinite(rb) ? rb.toFixed(2) : '∞', '真值 32') + kpi('已训练', steps, '步') + kpi('损失 MSE', Number.isFinite(L) ? (L > 1e5 ? L.toExponential(1) : L.toFixed(2)) : '∞', '平方误差的平均');
      // 损失曲线
      lctx.clearRect(0, 0, LW, LH);
      const lg = losses.filter(Number.isFinite).map((v) => Math.log10(Math.max(v, 1e-3)));
      const top = Math.max(5, ...lg), bot = Math.min(0, ...lg);
      lctx.strokeStyle = css('--grid'); lctx.fillStyle = css('--chart-ink');
      for (let e = Math.ceil(bot); e <= top; e++) { const y = 6 + (1 - (e - bot) / (top - bot)) * (LH - 12); lctx.beginPath(); lctx.moveTo(30, y); lctx.lineTo(LW, y); lctx.stroke(); lctx.fillText(`1e${e}`, 0, y + 4); }
      lctx.strokeStyle = css('--c2'); lctx.lineWidth = 2; lctx.beginPath();
      lg.forEach((v, i) => { const x = 30 + (i / Math.max(1, lg.length - 1)) * (LW - 32), y = 6 + (1 - (v - bot) / (top - bot)) * (LH - 12); i ? lctx.lineTo(x, y) : lctx.moveTo(x, y); });
      lctx.stroke();
      const v = el.querySelector('.nf-v');
      if (!Number.isFinite(L) || L > 1e7) { v.className = 'verdict warn nf-v'; v.textContent = '✗ 发散了：学习率太大。原始输入时 w 的梯度里乘着 x（最大到 100），稍大一点的学习率就会让 w 越跳越远。'; }
      else if (Math.abs(rw - 1.8) < 0.05 && Math.abs(rb - 32) < 1.5) { v.className = 'verdict ok nf-v'; v.textContent = `✓ 学会了！用了 ${steps} 步，模型找到了 w ≈ ${rw.toFixed(2)}、b ≈ ${rb.toFixed(1)}——和真实公式 F = 1.8C + 32 几乎一样（剩下的差距来自测量误差）。`; }
      else if (!normBox.checked && steps >= 500) { v.className = 'verdict warn nf-v'; v.textContent = `已经 ${steps} 步了，w 早就接近 1.8，b 却还在慢慢往 32 爬：w 的梯度里乘着 x（几十上百），b 的梯度没有，同一个学习率对 b 来说太小了。勾上「标准化输入」再试试。`; }
      else { v.className = 'verdict nf-v'; v.textContent = steps ? '' : '点「训练 10 步」或「自动」开始。先用原始输入（不勾选）观察 b 有多慢。'; }
    }
    const setLr = () => { lrOut.textContent = lrOf() < 0.01 ? lrOf().toExponential(1) : lrOf().toFixed(2); };
    lrIn.value = 80;
    normBox.addEventListener('change', () => { lrIn.value = normBox.checked ? 40 : 80; setLr(); reset(); });
    lrIn.addEventListener('input', setLr);
    const anim = NN.loop(el, () => { train(normBox.checked ? 1 : 25); if (steps >= 20000 || !Number.isFinite(losses[losses.length - 1])) stop(); });
    const stop = () => { anim.stop(); runBtn.textContent = '▶ 自动'; };
    el.addEventListener('click', (e) => {
      const a = e.target.closest('[data-a]')?.dataset.a;
      if (a === '10') train(10);
      if (a === 'reset') { stop(); reset(); }
      if (a === 'run') { if (anim.running) stop(); else { anim.start(); runBtn.textContent = '⏸ 暂停'; } }
    });
    setLr();
    reset();
  },
};
