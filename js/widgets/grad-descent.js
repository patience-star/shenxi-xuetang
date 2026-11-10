/* 互动组件：梯度下降
 * 一维：切线的斜率就是导数，沿下坡方向迈一步，步长 = 学习率 × 导数；
 * 二维：在等高线图上沿负梯度方向走。调学习率看收敛、震荡和发散；点画布可以换起点。 */
Widgets['grad-descent'] = {
  title: '梯度下降：一步一步走到谷底',
  render(el, opts) {
    const FUN = {
      bowl: { dim: 1, name: '一维 · 碗形 f(x) = x²', f: (x) => x * x, g: (x) => 2 * x, x0: 3.2, xr: [-4, 4] },
      valley: { dim: 1, name: '一维 · 两个山谷（非凸）', f: (x) => 0.05 * x ** 4 - 0.6 * x * x + 0.3 * x + 2, g: (x) => 0.2 * x ** 3 - 1.2 * x + 0.3, x0: 3.7, xr: [-4.3, 4.3] },
      round: { dim: 2, name: '二维 · 圆碗 f = (x² + y²) / 2', a: 1, b: 1, p0: [-3.4, 1.7] },
      ravine: { dim: 2, name: '二维 · 狭长山谷 f = (x² + 10y²) / 2', a: 1, b: 10, p0: [-3.5, 1.3] },
    };
    const start = opts.start || 'bowl';
    el.innerHTML = `
      <div class="controls">
        <label>地形<select class="gd-fn">${Object.entries(FUN).map(([k, v]) => `<option value="${k}"${k === start ? ' selected' : ''}>${v.name}</option>`).join('')}</select></label>
        <label>学习率 η = <output class="gd-lrv"></output><input type="range" class="gd-lr" min="0.01" max="1.2" step="0.01" value="${opts.lr || 0.1}"></label>
        <span class="row"><button type="button" class="btn sm" data-a="step">走一步</button><button type="button" class="btn sm ghost" data-a="run">▶ 自动</button><button type="button" class="btn sm ghost" data-a="reset">重置</button></span>
      </div>
      <canvas class="viz drag gd-cv" aria-label="梯度下降的路径"></canvas>
      <p class="gd-info small"></p>
      <p class="verdict gd-v"></p>
      <p class="small muted gd-tip"></p>`;
    const cv = el.querySelector('.gd-cv'), info = el.querySelector('.gd-info'), verdict = el.querySelector('.gd-v'), tipEl = el.querySelector('.gd-tip');
    const lrIn = el.querySelector('.gd-lr'), lrOut = el.querySelector('.gd-lrv'), runBtn = el.querySelector('[data-a=run]');
    let fn, pts, ctx, W, H, k;
    const num = (v) => (Math.abs(v) >= 1e4 ? v.toExponential(2) : v.toFixed(3));
    const lr = () => +lrIn.value;

    function reset(p0) {
      fn = FUN[el.querySelector('.gd-fn').value];
      W = 640; H = fn.dim === 1 ? 300 : 360;
      ctx = Charts.hidpi(cv, W, H);
      pts = [p0 || (fn.dim === 1 ? [fn.x0] : fn.p0.slice())];
      tipEl.textContent = fn === FUN.bowl ? '挑战：只走 1 步就到达谷底，学习率该取多少？（提示：看下面「下一步」的算式）'
        : fn === FUN.valley ? '试试：从右边出发，小学习率会掉进右边的浅谷（局部最小值）；把学习率调大一些，可能一步跨到左边更深的谷。点画布可以换起点。'
        : fn === FUN.ravine ? '试试：学习率 0.05 时慢慢挪；0.18 时沿陡峭方向来回横跳；超过 0.2 直接发散——这就是第 6 课要讲的 Momentum、Adam 想解决的问题。'
        : '圆碗各个方向一样陡，梯度直指谷底，学习率合适时一路直线下山。';
      draw();
    }
    const grad = (p) => (fn.dim === 1 ? [fn.g(p[0])] : [fn.a * p[0], fn.b * p[1]]);
    const fval = (p) => (fn.dim === 1 ? fn.f(p[0]) : 0.5 * (fn.a * p[0] ** 2 + fn.b * p[1] ** 2));
    const norm = (g) => Math.hypot(...g);

    function step() {
      const p = pts[pts.length - 1];
      if (norm(p) > 1e6) return false;
      const g = grad(p);
      pts.push(p.map((v, i) => v - lr() * g[i]));
      draw();
      return true;
    }

    function draw() {
      const css = Charts.css;
      ctx.clearRect(0, 0, W, H);
      const cur = pts[pts.length - 1], g = grad(cur);
      k = pts.length - 1;
      if (fn.dim === 1) draw1(cur); else draw2(cur);
      // 文字
      const big = norm(cur) > 1e3;
      if (fn.dim === 1) {
        const nx = cur[0] - lr() * g[0];
        info.innerHTML = `第 ${k} 步：x = <b>${num(cur[0])}</b>，f(x) = ${num(fval(cur))}，导数 f′(x) = <b>${num(g[0])}</b>${big ? '' : `<br>下一步：x − η·f′(x) = ${num(cur[0])} − ${lr()} × ${num(g[0])} = <b>${num(nx)}</b>`}`;
      } else {
        info.innerHTML = `第 ${k} 步：位置 (${num(cur[0])}, ${num(cur[1])})，f = ${num(fval(cur))}，梯度 ∇f = (<b>${num(g[0])}</b>, <b>${num(g[1])}</b>)，沿 −∇f 方向走 η·|∇f| = ${num(lr() * norm(g))}`;
      }
      const gs = pts.slice(-5).map((p) => grad(p)[fn.dim === 1 ? 0 : 1]);
      const osc = gs.length >= 4 && gs.every((v, i) => !i || v * gs[i - 1] < 0);
      if (big) { verdict.className = 'verdict warn gd-v'; verdict.textContent = '✗ 发散了！学习率太大：每一步都跨过谷底，而且落点越来越高，越跳越远。'; }
      else if (norm(g) < 1e-3) {
        verdict.className = 'verdict ok gd-v';
        verdict.textContent = fn === FUN.bowl && k === 1 ? '🎉 一步到位！f(x) = x² 的导数是 2x，η = 0.5 时 x − 0.5 × 2x = 0。曲线越弯（二阶导越大），合适的步长就越小——这正是调学习率的直觉。'
          : `✓ 到达谷底（梯度几乎为 0），用了 ${k} 步。`;
      } else if (osc) { verdict.className = 'verdict warn gd-v'; verdict.textContent = '在谷底两侧来回震荡：导数的正负号每一步都在翻转，说明步子迈大了。'; }
      else { verdict.className = 'verdict gd-v'; verdict.textContent = k ? '' : '点「走一步」开始；也可以点画布换一个起点。'; }
    }

    function draw1(cur) {
      const css = Charts.css, [x0, x1] = fn.xr, m = { l: 36, r: 12, t: 14, b: 26 };
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i <= 200; i++) { const v = fn.f(x0 + ((x1 - x0) * i) / 200); lo = Math.min(lo, v); hi = Math.max(hi, v); }
      const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
      const X = (x) => m.l + ((x - x0) / (x1 - x0)) * (W - m.l - m.r), Y = (y) => m.t + (1 - (y - lo) / (hi - lo)) * (H - m.t - m.b);
      // 坐标轴
      ctx.strokeStyle = css('--grid'); ctx.fillStyle = css('--chart-ink'); ctx.lineWidth = 1;
      for (let x = Math.ceil(x0); x <= x1; x++) { ctx.beginPath(); ctx.moveTo(X(x), m.t); ctx.lineTo(X(x), H - m.b); ctx.stroke(); ctx.fillText(String(x), X(x) - 4, H - 8); }
      // 曲线
      ctx.strokeStyle = css('--c1'); ctx.lineWidth = 2.5; ctx.beginPath();
      for (let i = 0; i <= 300; i++) { const x = x0 + ((x1 - x0) * i) / 300; i ? ctx.lineTo(X(x), Y(fn.f(x))) : ctx.moveTo(X(x), Y(fn.f(x))); }
      ctx.stroke();
      // 轨迹
      const clampX = (x) => Math.max(x0, Math.min(x1, x));
      ctx.strokeStyle = css('--c2'); ctx.fillStyle = css('--c2'); ctx.lineWidth = 1.5;
      pts.forEach((p, i) => {
        const x = clampX(p[0]), y = Math.max(lo, Math.min(hi, fn.f(x)));
        if (i) { const q = clampX(pts[i - 1][0]); ctx.beginPath(); ctx.moveTo(X(q), Y(Math.min(hi, fn.f(q)))); ctx.lineTo(X(x), Y(y)); ctx.stroke(); }
        ctx.beginPath(); ctx.arc(X(x), Y(y), i === pts.length - 1 ? 7 : 3, 0, Math.PI * 2); ctx.fill();
      });
      // 切线（斜率 = 导数）
      const x = cur[0];
      if (x >= x0 && x <= x1) {
        const s = fn.g(x), y = fn.f(x);
        ctx.strokeStyle = css('--muted'); ctx.setLineDash([6, 4]); ctx.lineWidth = 1.5; ctx.beginPath();
        ctx.moveTo(X(x - 0.9), Y(y - 0.9 * s)); ctx.lineTo(X(x + 0.9), Y(y + 0.9 * s)); ctx.stroke(); ctx.setLineDash([]);
        // 下一步的位移箭头（画在底部）
        const nx = clampX(x - lr() * s), yb = H - m.b - 8;
        ctx.strokeStyle = css('--c3'); ctx.fillStyle = css('--c3'); ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(X(x), yb); ctx.lineTo(X(nx), yb); ctx.stroke();
        if (Math.abs(X(nx) - X(x)) > 6) { const d = Math.sign(X(nx) - X(x)); ctx.beginPath(); ctx.moveTo(X(nx), yb); ctx.lineTo(X(nx) - d * 7, yb - 4); ctx.lineTo(X(nx) - d * 7, yb + 4); ctx.fill(); }
        ctx.fillText('下一步', Math.min(X(x), X(nx)) + 2, yb - 6);
      } else {
        ctx.fillStyle = css('--err'); ctx.fillText('→ 已经飞出画面', x > x1 ? W - 110 : m.l + 4, m.t + 14);
      }
    }

    function draw2(cur) {
      const css = Charts.css, sx = W / 8, sy = H / 4.5;
      const X = (x) => W / 2 + x * sx, Y = (y) => H / 2 - y * sy;
      ctx.lineWidth = 1;
      [0.25, 1, 2.25, 4, 6.25, 9, 12.25, 16, 20.25, 25].forEach((c, i) => {
        ctx.strokeStyle = i % 2 ? css('--grid') : css('--axis');
        ctx.beginPath(); ctx.ellipse(X(0), Y(0), Math.sqrt((2 * c) / fn.a) * sx, Math.sqrt((2 * c) / fn.b) * sy, 0, 0, Math.PI * 2); ctx.stroke();
      });
      ctx.fillStyle = css('--c3'); ctx.font = '16px sans-serif'; ctx.fillText('★', X(0) - 6, Y(0) + 6); ctx.font = `12px ${css('--font')}`;
      const clip = (p) => [Math.max(-4.2, Math.min(4.2, p[0])), Math.max(-2.4, Math.min(2.4, p[1]))];
      ctx.strokeStyle = css('--c2'); ctx.fillStyle = css('--c2'); ctx.lineWidth = 1.8; ctx.beginPath();
      pts.forEach((p, i) => { const [a, b] = clip(p); i ? ctx.lineTo(X(a), Y(b)) : ctx.moveTo(X(a), Y(b)); });
      ctx.stroke();
      pts.forEach((p, i) => { const [a, b] = clip(p); ctx.beginPath(); ctx.arc(X(a), Y(b), i === pts.length - 1 ? 7 : 3, 0, Math.PI * 2); ctx.fill(); });
      const g = grad(cur), n = norm(g);
      if (n > 1e-6 && norm(cur) < 10) {
        const L = Math.min(1.3, 0.35 * n), ux = -g[0] / n, uy = -g[1] / n, [a, b] = cur;
        ctx.strokeStyle = css('--c3'); ctx.fillStyle = css('--c3'); ctx.lineWidth = 2.2;
        ctx.beginPath(); ctx.moveTo(X(a), Y(b)); ctx.lineTo(X(a + L * ux), Y(b + L * uy)); ctx.stroke();
        ctx.fillText('−∇f（最陡下坡）', X(a + L * ux) + 6, Y(b + L * uy) - 4);
      }
    }

    // 交互
    lrOut.textContent = lrIn.value;
    lrIn.addEventListener('input', () => { lrOut.textContent = lrIn.value; draw(); });
    el.querySelector('.gd-fn').addEventListener('change', () => { anim.stop(); runBtn.textContent = '▶ 自动'; reset(); });
    let acc = 0, last = 0;
    const anim = NN.loop(el, () => {
      const t = performance.now();
      acc += Math.min(200, t - (last || t)); last = t;
      while (acc > 150) {
        acc -= 150;
        const g = grad(pts[pts.length - 1]);
        if (norm(g) < 1e-3 || norm(pts[pts.length - 1]) > 1e3 || pts.length > 300 || !step()) { anim.stop(); runBtn.textContent = '▶ 自动'; break; }
      }
    });
    el.addEventListener('click', (e) => {
      const a = e.target.closest('[data-a]')?.dataset.a;
      if (a === 'step') step();
      if (a === 'reset') { anim.stop(); runBtn.textContent = '▶ 自动'; reset(); }
      if (a === 'run') {
        if (anim.running) { anim.stop(); runBtn.textContent = '▶ 自动'; }
        else { last = 0; acc = 150; anim.start(); runBtn.textContent = '⏸ 暂停'; }
      }
    });
    cv.addEventListener('pointerdown', (e) => {
      const [x, y] = Charts.at(cv, e, W, H);
      anim.stop(); runBtn.textContent = '▶ 自动';
      if (fn.dim === 1) { const m = { l: 36, r: 12 }; reset([fn.xr[0] + ((x - m.l) / (W - m.l - m.r)) * (fn.xr[1] - fn.xr[0])]); }
      else reset([(x - W / 2) / (W / 8), -(y - H / 2) / (H / 4.5)]);
    });
    reset();
  },
};
