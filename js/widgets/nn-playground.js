/* 互动组件：神经网络训练场（二维分类，在浏览器里实时训练）
 * 选数据、选网络结构（几层、每层几个神经元、激活函数）、学习率，点「开始」就能看到网络边训练边改变分界线。
 * 下面的小方块是每个隐藏神经元在整个平面上的输出：第一层每个神经元只会画「一条直线」，
 * 后面的层把这些直线组合成弯曲的边界——这就是「层层组合特征」。 */
Widgets['nn-playground'] = {
  title: '神经网络训练场',
  render(el, opts) {
    const DATA = { circle: '圆环', xor: '异或', moons: '两个月亮', spiral: '螺旋' };
    el.innerHTML = `
      <div class="controls">
        <label>数据<select class="pg-data">${Object.entries(DATA).map(([k, v]) => `<option value="${k}"${k === (opts.data || 'circle') ? ' selected' : ''}>${v}</option>`).join('')}</select></label>
        <label>隐藏层数<select class="pg-layers">${[0, 1, 2, 3].map((n) => `<option${n === (opts.layers ?? 1) ? ' selected' : ''}>${n}</option>`).join('')}</select></label>
        <label>每层神经元<select class="pg-width">${[1, 2, 3, 4, 6, 8].map((n) => `<option${n === (opts.width ?? 4) ? ' selected' : ''}>${n}</option>`).join('')}</select></label>
        <label>激活函数<select class="pg-act"><option value="tanh">tanh</option><option value="relu">ReLU</option><option value="sigmoid">sigmoid</option></select></label>
        <label>学习率<select class="pg-lr">${[0.003, 0.01, 0.03, 0.1, 0.3].map((v) => `<option${v === 0.03 ? ' selected' : ''}>${v}</option>`).join('')}</select></label>
      </div>
      <div class="row"><button type="button" class="btn sm" data-a="run">▶ 开始训练</button><button type="button" class="btn sm ghost" data-a="step">训练 1 轮</button><button type="button" class="btn sm ghost" data-a="reset">重新初始化</button></div>
      <div class="split wide-left" style="margin-top:10px">
        <canvas class="viz pg-cv" aria-label="分类边界与数据点"></canvas>
        <div><div class="kpis pg-k"></div><canvas class="viz pg-loss" aria-label="损失曲线"></canvas><p class="small muted" style="margin:2px 0 0">损失曲线</p></div>
      </div>
      <h4>每个隐藏神经元「看到」的平面</h4><div class="pg-neurons"></div>
      <p class="small muted pg-tip">挑战：① 1 个隐藏层分开「圆环」至少要几个神经元？ ② 不要隐藏层（0 层）能分开「异或」吗？为什么？ ③ 「螺旋」要多深多宽才能分开？</p>`;
    const $ = (s) => el.querySelector(s);
    const cv = $('.pg-cv'), W = 420, ctx = Charts.hidpi(cv, W, W), lv = $('.pg-loss'), LW = 300, LH = 130, lctx = Charts.hidpi(lv, LW, LH);
    const G = 48, R = 1.25;   // 网格分辨率与坐标范围 [-R, R]
    let pts, net, epoch, lossHist, runBtn = $('[data-a=run]');

    function makeData(kind) {
      const r = NN.rng(7), out = [];
      const add = (x, y, t) => out.push([x + r.randn() * 0.04, y + r.randn() * 0.04, t]);
      if (kind === 'circle') for (let i = 0; i < 240; i++) { const t = i % 2, rad = t ? r.uniform(0, 0.45) : r.uniform(0.7, 1), a = r.uniform(0, Math.PI * 2); add(rad * Math.cos(a), rad * Math.sin(a), t); }
      if (kind === 'xor') while (out.length < 240) { const x = r.uniform(-1, 1), y = r.uniform(-1, 1); if (Math.abs(x) > 0.08 && Math.abs(y) > 0.08) add(x, y, x * y > 0 ? 1 : 0); }
      if (kind === 'moons') for (let i = 0; i < 240; i++) { const t = i % 2, a = r.uniform(0, Math.PI); add(t ? Math.cos(a) * 0.75 - 0.35 : 0.35 - Math.cos(a) * 0.75, t ? Math.sin(a) * 0.75 - 0.2 : 0.2 - Math.sin(a) * 0.75, t); }
      if (kind === 'spiral') for (let i = 0; i < 300; i++) { const t = i % 2, k = Math.floor(i / 2) / 150, rad = 0.08 + k * 0.92, a = k * 3.2 * Math.PI + t * Math.PI; add(rad * Math.cos(a), rad * Math.sin(a), t); }
      return out;
    }
    function build() {
      const L = +$('.pg-layers').value, w = +$('.pg-width').value;
      net = new NN.MLP([2, ...Array(L).fill(w), 1], { act: $('.pg-act').value, out: 'linear', seed: (Math.random() * 1e9) | 0 });
      epoch = 0;
      lossHist = [evalLoss()];
      draw();
    }
    const sig = (z) => 1 / (1 + Math.exp(-z));
    function evalLoss() {
      let L = 0, ok = 0;
      for (const [x, y, t] of pts) { const p = Math.min(1 - 1e-7, Math.max(1e-7, sig(net.forward([x, y])[0]))); L -= t ? Math.log(p) : Math.log(1 - p); ok += (p > 0.5) === !!t; }
      evalLoss.acc = ok / pts.length;
      return L / pts.length;
    }
    /** 一轮：打乱后按小批量（16 个）训练，二分类交叉熵对 logit 的梯度就是 p − t */
    function trainEpoch() {
      const lr = +$('.pg-lr').value, idx = pts.map((_, i) => i).sort(() => Math.random() - 0.5);
      for (let i = 0; i < idx.length; i++) {
        const [x, y, t] = pts[idx[i]];
        net.backward([sig(net.forward([x, y])[0]) - t]);
        if ((i + 1) % 16 === 0 || i === idx.length - 1) net.step({ lr, opt: 'adam' });
      }
      epoch++;
      lossHist.push(evalLoss());
      if (lossHist.length > 400) lossHist.splice(1, 1);
    }
    function draw() {
      const css = Charts.css;
      // 背景：网格上每一点的输出概率（蓝 = 类别 1，橙 = 类别 0），同时记下每个隐藏神经元的输出
      const off = document.createElement('canvas'); off.width = off.height = G;
      const octx = off.getContext('2d'), im = octx.createImageData(G, G);
      const c1 = Charts.rgb('--c1'), c0 = Charts.rgb('--c2'), bg = Charts.rgb('--surface');
      const L = net.sizes.length - 2, maps = [];
      for (let l = 0; l < L; l++) for (let j = 0; j < net.sizes[l + 1]; j++) maps.push({ l, j, v: new Float32Array(G * G) });
      for (let gy = 0; gy < G; gy++) for (let gx = 0; gx < G; gx++) {
        const x = -R + ((gx + 0.5) / G) * 2 * R, y = R - ((gy + 0.5) / G) * 2 * R;
        const p = sig(net.forward([x, y])[0]), k = gy * G + gx;
        maps.forEach((m) => { m.v[k] = net.as[m.l + 1][m.j]; });
        const t = Math.abs(p - 0.5) * 2 * 0.65, c = p > 0.5 ? c1 : c0;
        im.data.set([bg[0] + (c[0] - bg[0]) * t, bg[1] + (c[1] - bg[1]) * t, bg[2] + (c[2] - bg[2]) * t, 255], k * 4);
      }
      octx.putImageData(im, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, 0, 0, W, W);
      const P = (v) => ((v + R) / (2 * R)) * W;
      for (const [x, y, t] of pts) {
        ctx.beginPath(); ctx.arc(P(x), W - P(y), 4, 0, Math.PI * 2);
        ctx.fillStyle = t ? css('--c1') : css('--c2'); ctx.strokeStyle = css('--surface'); ctx.lineWidth = 1.2; ctx.fill(); ctx.stroke();
      }
      // 指标与损失曲线
      const loss = lossHist[lossHist.length - 1];
      const kpi = (l, v) => `<div class="kpi"><div class="label">${l}</div><div class="value">${v}</div></div>`;
      $('.pg-k').innerHTML = kpi('训练轮数', epoch) + kpi('损失', loss.toFixed(3)) + kpi('准确率', `${(evalLoss.acc * 100).toFixed(0)}%`) + kpi('参数个数', net.W.reduce((s, w) => s + w.length, 0) + net.b.reduce((s, b) => s + b.length, 0));
      lctx.clearRect(0, 0, LW, LH);
      const top = Math.max(0.8, ...lossHist);
      lctx.strokeStyle = css('--grid'); lctx.beginPath(); lctx.moveTo(0, LH - 4); lctx.lineTo(LW, LH - 4); lctx.stroke();
      lctx.strokeStyle = css('--c2'); lctx.lineWidth = 2; lctx.beginPath();
      lossHist.forEach((v, i) => { const x = (i / Math.max(1, lossHist.length - 1)) * LW, y = 4 + (1 - v / top) * (LH - 8); i ? lctx.lineTo(x, y) : lctx.moveTo(x, y); });
      lctx.stroke();
      // 隐藏神经元
      const box = $('.pg-neurons'), sigKey = net.sizes.join('-');
      if (box.dataset.key !== sigKey) {   // 结构变了才重建，否则复用画布
        box.dataset.key = sigKey;
        box.innerHTML = L ? Array.from({ length: L }, (_, l) => `<div class="small muted">第 ${l + 1} 隐藏层</div><div class="thumbs pg-l${l}" style="grid-template-columns:repeat(auto-fill,minmax(54px,62px));margin-bottom:8px"></div>`).join('')
          : '<p class="small muted">没有隐藏层：网络只是「一个神经元」，分界线只能是一条直线。</p>';
        maps.forEach((m) => { const c = document.createElement('canvas'); c.className = 'thumb'; box.querySelector(`.pg-l${m.l}`).appendChild(c); });
      }
      const cvs = box.querySelectorAll('canvas');
      maps.forEach((m, i) => Charts.img(cvs[i], m.v, G, G, { mode: 'seq' }));
    }
    const anim = NN.loop(el, () => {
      const t0 = performance.now();
      do trainEpoch(); while (performance.now() - t0 < 12);
      draw();
      if (epoch >= 3000) stop();
    });
    const stop = () => { anim.stop(); runBtn.textContent = '▶ 开始训练'; };
    el.addEventListener('click', (e) => {
      const a = e.target.closest('[data-a]')?.dataset.a;
      if (a === 'run') { if (anim.running) stop(); else { anim.start(); runBtn.textContent = '⏸ 暂停'; } }
      if (a === 'step') { trainEpoch(); draw(); }
      if (a === 'reset') build();
    });
    $('.pg-data').addEventListener('change', () => { pts = makeData($('.pg-data').value); build(); });
    ['.pg-layers', '.pg-width', '.pg-act'].forEach((s) => $(s).addEventListener('change', build));
    pts = makeData($('.pg-data').value);
    build();
  },
};
