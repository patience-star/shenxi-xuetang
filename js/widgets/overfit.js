/* 互动组件：过拟合与正则化（一维回归，在浏览器里实时训练）
 * 真实规律是一条平滑曲线（虚线），我们只拿到少量带噪声的样本（实心点）去训练一个小网络，
 * 再用另一批没见过的点（空心点）检验。看训练损失和验证损失怎样分道扬镳，以及加数据、加正则化、换小模型的效果。 */
Widgets.overfit = {
  title: '过拟合：背答案 vs 学规律',
  render(el) {
    el.innerHTML = `
      <div class="controls">
        <label>训练样本数<select class="of-n"><option>8</option><option selected>12</option><option>30</option><option>100</option></select></label>
        <label>模型大小<select class="of-m"><option value="4">小：1 层 × 4 个神经元</option><option value="16,16">中：2 层 × 16</option><option value="64,64" selected>大：2 层 × 64</option></select></label>
        <label>权重衰减（L2）<select class="of-l2"><option value="0" selected>0（不加）</option><option value="0.0003">0.0003</option><option value="0.003">0.003</option><option value="0.03">0.03</option></select></label>
        <span class="row"><button type="button" class="btn sm" data-a="run">▶ 开始训练</button><button type="button" class="btn sm ghost" data-a="reset">重置</button></span>
      </div>
      <div class="split wide-left">
        <canvas class="viz of-cv" aria-label="训练数据、验证数据和模型曲线"></canvas>
        <div><canvas class="viz of-loss" aria-label="训练与验证损失"></canvas>
          <div class="legend"><span><i class="line" style="background:var(--c1)"></i>训练损失</span><span><i class="line" style="background:var(--c2)"></i>验证损失</span><span><i class="line" style="background:var(--c3)"></i>验证最低点（早停）</span></div>
          <div class="kpis of-k"></div></div>
      </div>
      <p class="verdict of-v"></p>`;
    const $ = (s) => el.querySelector(s);
    const cv = $('.of-cv'), W = 520, H = 320, ctx = Charts.hidpi(cv, W, H), lv = $('.of-loss'), LW = 300, LH = 170, lctx = Charts.hidpi(lv, LW, LH);
    const truth = (x) => Math.sin(2.5 * x) * 0.8 + 0.3 * x;
    let train, val, net, hist, epoch;
    function reset() {
      const r = NN.rng(3), n = +$('.of-n').value;
      const mk = (k, rr) => Array.from({ length: k }, () => { const x = rr.uniform(-2, 2); return [x, truth(x) + rr.randn() * 0.25]; });
      train = mk(n, r);
      val = mk(60, NN.rng(99));
      net = new NN.MLP([1, ...$('.of-m').value.split(',').map(Number), 1], { act: 'tanh', seed: 5 });
      hist = { tr: [], va: [] };
      epoch = 0;
      record();
      draw();
    }
    const mse = (set) => set.reduce((s, [x, y]) => s + (net.forward([x / 2])[0] - y) ** 2, 0) / set.length;
    function record() { hist.tr.push(mse(train)); hist.va.push(mse(val)); }
    function trainEpoch() {
      const l2 = +$('.of-l2').value;
      for (const [x, y] of train) net.backward([2 * (net.forward([x / 2])[0] - y)]);   // 全批量：所有样本的梯度累加
      net.step({ lr: 0.01, opt: 'adam', l2 });
      epoch++;
      if (epoch % 5 === 0) record();
    }
    function draw() {
      const css = Charts.css, X = (x) => 30 + ((x + 2.1) / 4.2) * (W - 40), Y = (y) => 10 + (1 - (y + 2) / 4) * (H - 30);
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = css('--grid'); ctx.lineWidth = 1;
      for (let y = -2; y <= 2; y++) { ctx.beginPath(); ctx.moveTo(30, Y(y)); ctx.lineTo(W - 10, Y(y)); ctx.stroke(); }
      const curve = (g, color, dash, w) => { ctx.strokeStyle = color; ctx.setLineDash(dash); ctx.lineWidth = w; ctx.beginPath(); for (let i = 0; i <= 200; i++) { const x = -2.1 + (i / 200) * 4.2, y = Math.max(-2.2, Math.min(2.2, g(x))); i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)); } ctx.stroke(); ctx.setLineDash([]); };
      curve(truth, css('--muted'), [6, 5], 1.5);
      curve((x) => net.forward([x / 2])[0], css('--c2'), [], 2.5);
      ctx.lineWidth = 1.5;
      val.forEach(([x, y]) => { ctx.strokeStyle = css('--axis'); ctx.beginPath(); ctx.arc(X(x), Y(y), 3.5, 0, Math.PI * 2); ctx.stroke(); });
      train.forEach(([x, y]) => { ctx.fillStyle = css('--c1'); ctx.beginPath(); ctx.arc(X(x), Y(y), 5, 0, Math.PI * 2); ctx.fill(); });
      ctx.fillStyle = css('--chart-ink'); ctx.fillText('虚线 = 真实规律　实心 = 训练样本　空心 = 验证样本　橙线 = 模型', 34, H - 4);
      // 损失曲线（对数）
      lctx.clearRect(0, 0, LW, LH);
      const all = [...hist.tr, ...hist.va].map((v) => Math.log10(Math.max(v, 1e-4)));
      const top = Math.max(...all, 0.5), bot = Math.min(...all, -2);
      const P = (i, v) => [8 + (i / Math.max(1, hist.tr.length - 1)) * (LW - 16), 6 + (1 - (Math.log10(Math.max(v, 1e-4)) - bot) / (top - bot)) * (LH - 12)];
      [[hist.tr, '--c1'], [hist.va, '--c2']].forEach(([arr, c]) => { lctx.strokeStyle = css(c); lctx.lineWidth = 2; lctx.beginPath(); arr.forEach((v, i) => { const [x, y] = P(i, v); i ? lctx.lineTo(x, y) : lctx.moveTo(x, y); }); lctx.stroke(); });
      const best = hist.va.indexOf(Math.min(...hist.va));
      if (hist.va.length > 3) { const [bx] = P(best, 1); lctx.strokeStyle = css('--c3'); lctx.setLineDash([4, 3]); lctx.beginPath(); lctx.moveTo(bx, 0); lctx.lineTo(bx, LH); lctx.stroke(); lctx.setLineDash([]); }
      const tr = hist.tr[hist.tr.length - 1], va = hist.va[hist.va.length - 1];
      const kpi = (l, v) => `<div class="kpi"><div class="label">${l}</div><div class="value">${v}</div></div>`;
      $('.of-k').innerHTML = kpi('训练轮数', epoch) + kpi('训练损失', tr.toFixed(3)) + kpi('验证损失', va.toFixed(3));
      const v = $('.of-v');
      const gap = va - tr, big = $('.of-m').value === '64,64', n = +$('.of-n').value;
      if (epoch < 200) { v.className = 'verdict of-v'; v.textContent = epoch ? '训练中……' : '点「开始训练」。默认设置是「样本很少 + 模型很大」，最容易过拟合。'; }
      else if (tr > 0.2 && va > 0.2) { v.className = 'verdict warn of-v'; v.textContent = '欠拟合：训练损失和验证损失都降不下去——模型太简单（或正则化太强），连训练数据的规律都没学到。'; }
      else if (gap > 0.08 && va > 1.5 * Math.min(...hist.va)) { v.className = 'verdict warn of-v'; v.textContent = `过拟合：训练损失 ${tr.toFixed(3)} 很小，验证损失却从最低 ${Math.min(...hist.va).toFixed(3)} 涨回到 ${va.toFixed(3)}——模型在「背」训练点（曲线为了穿过每个点扭来扭去），而不是学规律。试试增加样本、加权重衰减，或在验证损失最低时停下（早停）。`; }
      else { v.className = 'verdict ok of-v'; v.textContent = `泛化良好：验证损失 ${va.toFixed(3)} 和训练损失接近，模型学到的曲线贴近真实规律。${big && n <= 12 ? '（大模型 + 少量数据能做到这样，是正则化的功劳。）' : ''}`; }
    }
    const anim = NN.loop(el, () => { for (let i = 0; i < 20; i++) trainEpoch(); draw(); if (epoch >= 4000) stop(); });
    const stop = () => { anim.stop(); $('[data-a=run]').textContent = '▶ 开始训练'; };
    el.addEventListener('click', (e) => {
      const a = e.target.closest('[data-a]')?.dataset.a;
      if (a === 'run') { if (anim.running) stop(); else { anim.start(); $('[data-a=run]').textContent = '⏸ 暂停'; } }
      if (a === 'reset') { stop(); reset(); }
    });
    el.querySelectorAll('select').forEach((s) => s.addEventListener('change', () => { stop(); reset(); }));
    reset();
  },
};
