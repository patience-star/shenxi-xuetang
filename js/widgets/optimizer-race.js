/* 互动组件：优化器赛跑
 * 四种优化器从同一点出发、在同一片地形上找最低点：SGD、Momentum、RMSProp、Adam。
 * 地形颜色越深越低（对数刻度的等高带）。观察：狭长山谷里 SGD 来回横跳、Momentum 越冲越快；
 * 鞍点附近 SGD 几乎停住，Adam / RMSProp 很快逃出来。 */
Widgets['optimizer-race'] = {
  title: '优化器赛跑：谁先到谷底？',
  render(el) {
    const S = {
      ravine: { name: '狭长山谷', f: (x, y) => 0.1 * x * x + 2 * y * y, g: (x, y) => [0.2 * x, 4 * y], xr: [-5, 5], yr: [-2.2, 2.2], p0: [-4.5, 1.6], lr1: 0.1, lr2: 0.12, note: '左右方向很平缓、上下方向很陡。SGD 的学习率只能按陡的方向来定，于是沿着山谷底部慢慢挪（把学习率调到 0.4 以上，它又会在陡峭方向来回横跳）；Momentum 把一致的前进方向不断累积，越走越快（学习率太大时冲过头、来回摆）；Adam 给每个方向自动配了合适的步长，几乎走直线。' },
      saddle: { name: '鞍点', f: (x, y) => 0.5 * x * x - 0.5 * y * y + 0.05 * y ** 4, g: (x, y) => [x, -y + 0.2 * y ** 3], xr: [-3.5, 3.5], yr: [-3, 3], p0: [-3, 0.01], lr1: 0.08, lr2: 0.05, note: '中间是马鞍形：沿 x 方向是谷底，沿 y 方向是山顶。从几乎正好在山脊上的位置出发，y 方向的梯度极小，SGD 会在鞍点附近停留很久；Adam 按梯度的「相对大小」迈步，很快滑下去。' },
      banana: { name: '香蕉谷（Rosenbrock）', f: (x, y) => (1 - x) ** 2 + 100 * (y - x * x) ** 2, g: (x, y) => [-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)], xr: [-2, 2], yr: [-1, 3], p0: [-1.6, 2.6], lr1: 0.0015, lr2: 0.06, note: '谷底是一条弯弯的香蕉形窄沟，最低点在 (1, 1)。进沟容易、沿沟底走到终点很难——优化算法的经典测试题。' },
    };
    const OPT = [
      { name: 'SGD', color: '--c8' }, { name: 'Momentum', color: '--c4' }, { name: 'RMSProp', color: '--c3' }, { name: 'Adam', color: '--c1' },
    ];
    el.innerHTML = `
      <div class="controls">
        <label>地形<select class="or-s">${Object.entries(S).map(([k, v]) => `<option value="${k}">${v.name}</option>`).join('')}</select></label>
        <label>SGD / Momentum 学习率 <output class="or-o1"></output><input type="range" class="or-l1" min="0" max="100" value="50"></label>
        <label>RMSProp / Adam 学习率 <output class="or-o2"></output><input type="range" class="or-l2" min="0" max="100" value="50"></label>
        <span class="row"><button type="button" class="btn sm" data-a="run">▶ 开始</button><button type="button" class="btn sm ghost" data-a="reset">重置</button></span>
      </div>
      <canvas class="viz drag or-cv" aria-label="四种优化器在地形上的轨迹"></canvas>
      <div class="legend or-lg" style="margin-top:6px"></div>
      <p class="small or-note"></p>`;
    const $ = (s) => el.querySelector(s);
    const cv = $('.or-cv'), W = 640, H = 380, ctx = Charts.hidpi(cv, W, H);
    let s, st, step, bg;
    // 滑块 0~100 映射到默认值的 1/10 ~ 10 倍（对数刻度）
    const lr = (k) => S[$('.or-s').value][`lr${k}`] * 10 ** ((+$(`.or-l${k}`).value - 50) / 50);
    const X = (x) => ((x - s.xr[0]) / (s.xr[1] - s.xr[0])) * W, Y = (y) => H - ((y - s.yr[0]) / (s.yr[1] - s.yr[0])) * H;
    function reset() {
      s = S[$('.or-s').value];
      step = 0;
      st = OPT.map(() => ({ p: s.p0.slice(), trail: [s.p0.slice()], m: [0, 0], v: [0, 0] }));
      // 背景：对数刻度的等高带
      const gw = 160, gh = 95, off = document.createElement('canvas'); off.width = gw; off.height = gh;
      const o = off.getContext('2d'), im = o.createImageData(gw, gh), vals = [];
      for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) vals.push(s.f(s.xr[0] + ((i + 0.5) / gw) * (s.xr[1] - s.xr[0]), s.yr[1] - ((j + 0.5) / gh) * (s.yr[1] - s.yr[0])));
      const lo = Math.min(...vals), lv = vals.map((v) => Math.log(1 + v - lo)), hi = Math.max(...lv);
      lv.forEach((v, k) => { const t = 1 - Math.floor((v / hi) * 12) / 12, c = Charts.seq(t * 0.85).bg.match(/\d+/g).map(Number); im.data.set([...c, 255], k * 4); });
      o.putImageData(im, 0, 0);
      bg = off;
      $('.or-note').textContent = s.note;
      draw();
    }
    function update() {
      step++;
      const l1 = lr(1), l2 = lr(2), eps = 1e-8;
      st.forEach((o, k) => {
        if (Math.abs(o.p[0]) > 1e3 || Math.abs(o.p[1]) > 1e3 || !Number.isFinite(o.p[0])) return;
        const g = s.g(o.p[0], o.p[1]);
        for (let i = 0; i < 2; i++) {
          if (k === 0) o.p[i] -= l1 * g[i];                                                   // SGD：沿负梯度走
          if (k === 1) { o.m[i] = 0.9 * o.m[i] + g[i]; o.p[i] -= l1 * o.m[i]; }                // Momentum：累积「速度」
          if (k === 2) { o.v[i] = 0.9 * o.v[i] + 0.1 * g[i] ** 2; o.p[i] -= (l2 * g[i]) / (Math.sqrt(o.v[i]) + eps); }   // RMSProp：除以梯度的均方根
          if (k === 3) {                                                                      // Adam：动量 + 自适应步长
            o.m[i] = 0.9 * o.m[i] + 0.1 * g[i]; o.v[i] = 0.999 * o.v[i] + 0.001 * g[i] ** 2;
            o.p[i] -= (l2 * (o.m[i] / (1 - 0.9 ** step))) / (Math.sqrt(o.v[i] / (1 - 0.999 ** step)) + eps);
          }
        }
        o.trail.push(o.p.slice());
      });
    }
    function draw() {
      const css = Charts.css;
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(bg, 0, 0, W, H);
      ctx.fillStyle = css('--surface'); ctx.font = '18px sans-serif';
      const best = s === S.banana ? [1, 1] : s === S.saddle ? null : [0, 0];
      if (best) ctx.fillText('★', X(best[0]) - 7, Y(best[1]) + 7);
      ctx.font = `12px ${css('--font')}`;
      st.forEach((o, k) => {
        ctx.strokeStyle = css(OPT[k].color); ctx.fillStyle = css(OPT[k].color); ctx.lineWidth = 2.2; ctx.beginPath();
        o.trail.forEach((p, i) => { const x = Math.max(-5, Math.min(W + 5, X(p[0]))), y = Math.max(-5, Math.min(H + 5, Y(p[1]))); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
        ctx.stroke();
        const p = o.p;
        ctx.beginPath(); ctx.arc(Math.max(4, Math.min(W - 4, X(p[0]))), Math.max(4, Math.min(H - 4, Y(p[1]))), 6, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = css('--surface'); ctx.lineWidth = 1.5; ctx.stroke();
      });
      ctx.fillStyle = css('--surface'); ctx.fillText(`第 ${step} 步`, 10, 18);
      $('.or-lg').innerHTML = st.map((o, k) => { const f = s.f(o.p[0], o.p[1]); return `<span><i style="background:var(${OPT[k].color})"></i>${OPT[k].name}：f = ${Number.isFinite(f) && f < 1e6 ? f.toFixed(4) : '发散'}</span>`; }).join('');
      $('.or-o1').textContent = lr(1).toPrecision(2); $('.or-o2').textContent = lr(2).toPrecision(2);
    }
    const anim = NN.loop(el, () => { for (let i = 0; i < 2; i++) update(); draw(); if (step >= 800) stop(); });
    const stop = () => { anim.stop(); $('[data-a=run]').textContent = '▶ 开始'; };
    el.addEventListener('click', (e) => {
      const a = e.target.closest('[data-a]')?.dataset.a;
      if (a === 'run') { if (anim.running) stop(); else { anim.start(); $('[data-a=run]').textContent = '⏸ 暂停'; } }
      if (a === 'reset') { stop(); reset(); }
    });
    $('.or-s').addEventListener('change', () => { stop(); $('.or-l1').value = 50; $('.or-l2').value = 50; reset(); });
    ['.or-l1', '.or-l2'].forEach((q) => $(q).addEventListener('input', () => { stop(); reset(); }));
    reset();
  },
};
