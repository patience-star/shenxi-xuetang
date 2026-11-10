/* 互动组件：一维 GAN 的对抗训练（在浏览器里实时训练）
 * 真实数据是一维的数（蓝色曲线是它的分布）；生成器把随机噪声变成数（橙色柱子是它生成的分布）；
 * 判别器给每个位置打「是真的」的概率（绿线）。看两者怎样你追我赶，以及双峰数据上常见的「模式坍塌」。 */
Widgets['gan-1d'] = {
  title: '一维 GAN：生成器与判别器的拉锯战',
  render(el) {
    const DIST = {
      one: { name: '单峰：N(2, 0.5²)', pdf: (x) => Math.exp(-((x - 2) ** 2) / 0.5) / Math.sqrt(0.5 * Math.PI), sample: (r) => 2 + 0.5 * r.randn() },
      two: { name: '双峰：两个高斯混合', pdf: (x) => 0.5 * (Math.exp(-((x + 2) ** 2) / 0.5) + Math.exp(-((x - 2) ** 2) / 0.5)) / Math.sqrt(0.5 * Math.PI), sample: (r) => (r.next() < 0.5 ? -2 : 2) + 0.5 * r.randn() },
    };
    el.innerHTML = `
      <div class="controls">
        <label>真实数据<select class="g1-d">${Object.entries(DIST).map(([k, v]) => `<option value="${k}">${v.name}</option>`).join('')}</select></label>
        <label>学习率<select class="g1-lr"><option>0.001</option><option selected>0.003</option><option>0.01</option></select></label>
        <span class="row"><button type="button" class="btn sm" data-a="run">▶ 开始对抗</button><button type="button" class="btn sm ghost" data-a="step">训练 10 步</button><button type="button" class="btn sm ghost" data-a="reset">重新初始化</button></span>
      </div>
      <canvas class="viz g1-cv" aria-label="真实分布、生成分布与判别器输出"></canvas>
      <div class="legend"><span><i style="background:var(--c1)"></i>真实数据的分布</span><span><i style="background:var(--c2)"></i>生成器产生的数据</span><span><i class="line" style="background:var(--c3)"></i>判别器：D(x) = 这里是真数据的概率</span></div>
      <div class="kpis g1-k"></div>
      <p class="verdict g1-v"></p>`;
    const $ = (s) => el.querySelector(s);
    const cv = $('.g1-cv'), W = 640, H = 280, ctx = Charts.hidpi(cv, W, H), r = NN.rng(11);
    const sig = (z) => 1 / (1 + Math.exp(-z));
    let G, D, steps, dl, gl;
    function reset() {
      G = new NN.MLP([1, 16, 16, 1], { act: 'tanh', seed: (Math.random() * 1e9) | 0 });
      D = new NN.MLP([1, 16, 16, 1], { act: 'tanh', seed: (Math.random() * 1e9) | 0 });
      steps = 0; dl = gl = 0;
      draw();
    }
    /** 一步对抗训练：先更新判别器，再更新生成器（非饱和损失：让 D(G(z)) 尽量接近 1） */
    function trainStep() {
      const dist = DIST[$('.g1-d').value], lr = +$('.g1-lr').value, B = 64;
      let ld = 0, lg = 0;
      D.zeroGrad();
      for (let k = 0; k < B; k++) {
        const xr = dist.sample(r) / 3, pr = sig(D.forward([xr])[0]);
        D.backward([pr - 1]); ld -= Math.log(pr + 1e-9);                       // 真数据：希望 D 输出 1
        const xf = G.forward([r.randn()])[0], pf = sig(D.forward([xf])[0]);
        D.backward([pf]); ld -= Math.log(1 - pf + 1e-9);                       // 假数据：希望 D 输出 0
      }
      D.step({ lr, opt: 'adam', beta1: 0.5 });
      G.zeroGrad();
      for (let k = 0; k < B; k++) {
        const xf = G.forward([r.randn()])[0], pf = sig(D.forward([xf])[0]);
        const dx = D.backward([pf - 1])[0];                                     // 梯度穿过判别器，传到生成的数上
        G.backward([dx]); lg -= Math.log(pf + 1e-9);
      }
      D.zeroGrad();                                                             // 这一步只更新生成器
      G.step({ lr, opt: 'adam', beta1: 0.5 });
      steps++; dl = ld / B; gl = lg / B;
    }
    function draw() {
      const css = Charts.css, dist = DIST[$('.g1-d').value], X = (x) => 20 + ((x + 5) / 10) * (W - 40), base = H - 26;
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = css('--axis'); ctx.beginPath(); ctx.moveTo(20, base); ctx.lineTo(W - 20, base); ctx.stroke();
      ctx.fillStyle = css('--chart-ink');
      for (let x = -4; x <= 4; x += 2) ctx.fillText(String(x), X(x) - 4, H - 8);
      // 生成分布的直方图
      const bins = new Array(50).fill(0), N = 800;
      for (let k = 0; k < N; k++) { const x = G.forward([r.randn()])[0] * 3, b = Math.floor(((x + 5) / 10) * 50); if (b >= 0 && b < 50) bins[b]++; }
      const scale = (base - 20) / 1.0;
      ctx.fillStyle = css('--c2'); ctx.globalAlpha = 0.75;
      bins.forEach((c, b) => { const dens = c / N / 0.2, h = Math.min(base - 10, dens * scale * 0.9); ctx.fillRect(X(-5 + b * 0.2) + 1, base - h, (W - 40) / 50 - 2, h); });
      ctx.globalAlpha = 1;
      // 真实分布
      ctx.strokeStyle = css('--c1'); ctx.lineWidth = 2.5; ctx.beginPath();
      for (let i = 0; i <= 200; i++) { const x = -5 + i / 20, y = base - dist.pdf(x) * scale * 0.9; i ? ctx.lineTo(X(x), y) : ctx.moveTo(X(x), y); }
      ctx.stroke();
      // 判别器
      ctx.strokeStyle = css('--c3'); ctx.lineWidth = 2; ctx.setLineDash([6, 4]); ctx.beginPath();
      for (let i = 0; i <= 200; i++) { const x = -5 + i / 20, y = base - sig(D.forward([x / 3])[0]) * (base - 20); i ? ctx.lineTo(X(x), y) : ctx.moveTo(X(x), y); }
      ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = css('--c3'); ctx.fillText('D=1', W - 44, 24); ctx.fillText('D=0.5', W - 52, base - (base - 20) / 2 + 4);
      const kpi = (l, v) => `<div class="kpi"><div class="label">${l}</div><div class="value">${v}</div></div>`;
      $('.g1-k').innerHTML = kpi('训练步数', steps) + kpi('判别器损失', dl.toFixed(3)) + kpi('生成器损失', gl.toFixed(3));
      const left = bins.slice(0, 25).reduce((a, b) => a + b, 0) / N, two = $('.g1-d').value === 'two';
      $('.g1-v').textContent = !steps ? '点「开始对抗」。一开始生成器只会在 0 附近乱吐数字，判别器很容易分辨。'
        : two && steps > 300 && (left < 0.1 || left > 0.9) ? `模式坍塌：真实数据有两个峰，生成器却只学会了其中一个（${left > 0.5 ? '左' : '右'}边）——只要判别器在那里被骗过就够了。这是 GAN 训练的经典难题，点「重新初始化」再试，结果可能不同。`
        : steps > 400 ? '判别器的绿线越来越平、接近 0.5 时，说明它已经分不清真假：生成器学会了模仿真实分布。实际训练中两者常常来回摆动，很难真正停在平衡点。'
        : '生成器（橙）被判别器的梯度推着，往判别器认为「更像真的」（绿线高）的地方移动；判别器也在不断调整，盯住新的破绽。';
    }
    const anim = NN.loop(el, () => { for (let k = 0; k < 4; k++) trainStep(); draw(); if (steps > 4000) stop(); });
    const stop = () => { anim.stop(); $('[data-a=run]').textContent = '▶ 开始对抗'; };
    el.addEventListener('click', (e) => {
      const a = e.target.closest('[data-a]')?.dataset.a;
      if (a === 'run') { if (anim.running) stop(); else { anim.start(); $('[data-a=run]').textContent = '⏸ 暂停'; } }
      if (a === 'step') { for (let k = 0; k < 10; k++) trainStep(); draw(); }
      if (a === 'reset') { stop(); reset(); }
    });
    $('.g1-d').addEventListener('change', () => { stop(); reset(); });
    reset();
  },
};
