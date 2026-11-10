/* 互动组件：感知机
 * 平面上的点分两类（● 输出 1，○ 输出 0），感知机就是一条直线 w₁x₁ + w₂x₂ + b = 0：
 * 直线一侧（浅蓝）输出 1，另一侧输出 0。可以手动拖滑块调直线，也可以让「感知机学习规则」一步步自己学。
 * 预设：与、或、异或（学不会！），也可以自己点出数据。 */
Widgets.perceptron = {
  title: '感知机：一条会自己挪动的直线',
  render(el, opts) {
    const SETS = {
      and: { name: '与 AND', pts: [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]] },
      or: { name: '或 OR', pts: [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]] },
      xor: { name: '异或 XOR', pts: [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]] },
      custom: { name: '自己点', pts: [[0.1, 0.9, 1], [0.3, 1.2, 1], [-0.2, 0.6, 1], [1.0, 0.1, 0], [1.3, 0.4, 0], [0.8, -0.3, 0]] },
    };
    const START = { w1: 0.3, w2: -0.6, b: 0.2 };
    el.innerHTML = `
      <div class="controls">
        <label>数据<select class="pc-set">${Object.entries(SETS).map(([k, v]) => `<option value="${k}"${k === (opts.set || 'and') ? ' selected' : ''}>${v.name}</option>`).join('')}</select></label>
        <label>w₁ = <output class="pc-o1"></output><input type="range" class="pc-w1" min="-2" max="2" step="0.05"></label>
        <label>w₂ = <output class="pc-o2"></output><input type="range" class="pc-w2" min="-2" max="2" step="0.05"></label>
        <label>b = <output class="pc-ob"></output><input type="range" class="pc-b" min="-2" max="2" step="0.05"></label>
        <label>学习率 η<select class="pc-lr"><option>0.1</option><option selected>0.3</option><option>1</option></select></label>
      </div>
      <div class="row"><button type="button" class="btn sm" data-a="one">学一个样本</button><button type="button" class="btn sm ghost" data-a="run">▶ 自动学习</button><button type="button" class="btn sm ghost" data-a="reset">重置直线</button>
        <span class="pc-custom" hidden>点击画布添加：<span class="seg"><button type="button" class="on" data-c="1">● 类别 1</button><button type="button" data-c="0">○ 类别 0</button></span> <button type="button" class="btn sm ghost" data-a="clear">清空</button></span></div>
      <div class="split wide-left" style="margin-top:10px">
        <canvas class="viz drag pc-cv" aria-label="感知机的分界线与数据点"></canvas>
        <div><div class="kpis pc-k"></div><div class="pc-log small"></div></div>
      </div>
      <p class="verdict pc-v"></p>`;
    const $ = (s) => el.querySelector(s);
    const cv = $('.pc-cv'), W = 440, H = 440, ctx = Charts.hidpi(cv, W, H);
    const R = [-0.7, 1.7], X = (x) => ((x - R[0]) / (R[1] - R[0])) * W, Y = (y) => H - ((y - R[0]) / (R[1] - R[0])) * H;
    const inv = (px, py) => [R[0] + (px / W) * (R[1] - R[0]), R[0] + ((H - py) / H) * (R[1] - R[0])];
    let pts, w1, w2, b, cursor, epoch, epochErr, updates, cls = 1, lastIdx = -1, history = [];
    const predict = (p) => (w1 * p[0] + w2 * p[1] + b > 0 ? 1 : 0);
    const sync = () => { $('.pc-w1').value = w1; $('.pc-w2').value = w2; $('.pc-b').value = b; $('.pc-o1').textContent = w1.toFixed(2); $('.pc-o2').textContent = w2.toFixed(2); $('.pc-ob').textContent = b.toFixed(2); };
    function load() {
      const k = $('.pc-set').value;
      pts = SETS[k].pts.map((p) => p.slice());
      $('.pc-custom').hidden = k !== 'custom';
      resetLine();
    }
    function resetLine() { ({ w1, w2, b } = START); cursor = 0; epoch = 1; epochErr = 0; updates = 0; lastIdx = -1; history = []; sync(); draw(); }
    /** 感知机学习规则：看一个样本，错了才改 */
    function learnOne() {
      if (!pts.length) return;
      const i = cursor % pts.length, p = pts[i], y = predict(p), err = p[2] - y, eta = +$('.pc-lr').value;
      lastIdx = i;
      if (err) {
        const old = [w1, w2, b];
        w1 = +(w1 + eta * err * p[0]).toFixed(4); w2 = +(w2 + eta * err * p[1]).toFixed(4); b = +(b + eta * err).toFixed(4);
        epochErr++; updates++;
        history.unshift(`<b>样本 (${p[0]}, ${p[1]})</b> 真实 ${p[2]}、预测 ${y} → <span style="color:var(--err)">错了</span>，误差 ${err > 0 ? '+1' : '−1'}：<br>w₁ ← ${old[0].toFixed(2)} ${err > 0 ? '+' : '−'} ${eta}×${p[0]} = ${w1.toFixed(2)}；w₂ ← ${old[1].toFixed(2)} ${err > 0 ? '+' : '−'} ${eta}×${p[1]} = ${w2.toFixed(2)}；b ← ${old[2].toFixed(2)} ${err > 0 ? '+' : '−'} ${eta} = ${b.toFixed(2)}`);
      } else history.unshift(`样本 (${p[0]}, ${p[1]}) 真实 ${p[2]}、预测 ${y} → 对了，不改`);
      history = history.slice(0, 4);
      cursor++;
      let done = false;
      if (cursor % pts.length === 0) { done = epochErr === 0; if (!done) { epoch++; epochErr = 0; } }
      sync(); draw();
      return done;
    }
    function draw() {
      const css = Charts.css;
      ctx.clearRect(0, 0, W, H);
      // 两侧区域：按格子着色（浅蓝 = 输出 1）
      const step = 11, pos = css('--primary-weak');
      for (let px = 0; px < W; px += step) for (let py = 0; py < H; py += step) {
        const [x, y] = inv(px + step / 2, py + step / 2);
        if (w1 * x + w2 * y + b > 0) { ctx.fillStyle = pos; ctx.fillRect(px, py, step, step); }
      }
      // 坐标轴
      ctx.strokeStyle = css('--axis'); ctx.lineWidth = 1; ctx.fillStyle = css('--chart-ink');
      ctx.beginPath(); ctx.moveTo(X(0), 0); ctx.lineTo(X(0), H); ctx.moveTo(0, Y(0)); ctx.lineTo(W, Y(0)); ctx.stroke();
      [0, 1].forEach((v) => { ctx.fillText(String(v), X(v) + 3, Y(0) + 14); if (v) ctx.fillText(String(v), X(0) + 4, Y(v) + 4); });
      ctx.fillText('x₁', W - 16, Y(0) - 6); ctx.fillText('x₂', X(0) + 6, 12);
      // 分界线 w1·x1 + w2·x2 + b = 0
      const norm = Math.hypot(w1, w2);
      if (norm > 1e-9) {
        const [a, c] = Math.abs(w2) > Math.abs(w1) ? [[R[0], -(w1 * R[0] + b) / w2], [R[1], -(w1 * R[1] + b) / w2]] : [[-(w2 * R[0] + b) / w1, R[0]], [-(w2 * R[1] + b) / w1, R[1]]];
        ctx.strokeStyle = css('--primary'); ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(X(a[0]), Y(a[1])); ctx.lineTo(X(c[0]), Y(c[1])); ctx.stroke();
        // 法向量（权重方向）：从直线上离原点最近的点出发
        const t = -b / (norm * norm), ox = w1 * t, oy = w2 * t, L = 0.35;
        ctx.strokeStyle = css('--c3'); ctx.fillStyle = css('--c3'); ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(X(ox), Y(oy)); ctx.lineTo(X(ox + (w1 / norm) * L), Y(oy + (w2 / norm) * L)); ctx.stroke();
        ctx.fillText('w', X(ox + (w1 / norm) * L) + 4, Y(oy + (w2 / norm) * L));
      }
      // 数据点
      pts.forEach((p, i) => {
        const ok = predict(p) === p[2], x = X(p[0]), y = Y(p[1]);
        ctx.lineWidth = 3; ctx.strokeStyle = p[2] ? css('--c1') : css('--c2'); ctx.fillStyle = p[2] ? css('--c1') : css('--surface');
        ctx.beginPath(); ctx.arc(x, y, 10, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        if (!ok) { ctx.strokeStyle = css('--err'); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, 16, 0, Math.PI * 2); ctx.stroke(); }
        if (i === lastIdx) { ctx.strokeStyle = css('--text'); ctx.setLineDash([3, 3]); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x, y, 21, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]); }
      });
      const wrong = pts.filter((p) => predict(p) !== p[2]).length;
      const kpi = (label, v) => `<div class="kpi"><div class="label">${label}</div><div class="value">${v}</div></div>`;
      $('.pc-k').innerHTML = kpi('当前分错', `${wrong} / ${pts.length}`) + kpi('第几轮', epoch) + kpi('累计修改', updates);
      $('.pc-log').innerHTML = history.length ? `<p class="muted" style="margin:0 0 4px">最近几步：</p>${history.map((h) => `<div class="notice" style="margin:4px 0">${h}</div>`).join('')}` : '<p class="muted">红圈 = 当前被分错的点。点「学一个样本」：感知机按顺序看一个样本，<b>分对了不动，分错了把直线往这个点的正确一侧推一下</b>。</p>';
      const v = $('.pc-v'), isXor = $('.pc-set').value === 'xor';
      if (!wrong) { v.className = 'verdict ok pc-v'; v.textContent = `✓ 全部分对！${updates ? `一共修改了 ${updates} 次。` : ''}只要数据是「线性可分」的（存在一条直线能分开），感知机学习规则保证在有限步内找到这样的直线。`; }
      else if (isXor && epoch > 4) { v.className = 'verdict warn pc-v'; v.textContent = `已经第 ${epoch} 轮了，还是至少有 1 个点分错：异或的 (0,1)、(1,0) 要在一侧，(0,0)、(1,1) 要在另一侧，平面上不存在这样的一条直线。直线只会来回摆动，永远停不下来——这就是 1969 年让神经网络研究陷入低谷的「异或难题」。`; }
      else { v.className = 'verdict pc-v'; v.textContent = ''; }
    }
    ['w1', 'w2', 'b'].forEach((k) => $(`.pc-${k}`).addEventListener('input', (e) => { if (k === 'w1') w1 = +e.target.value; if (k === 'w2') w2 = +e.target.value; if (k === 'b') b = +e.target.value; sync(); draw(); }));
    $('.pc-set').addEventListener('change', () => { stop(); load(); });
    let acc = 0, last = 0;
    const anim = NN.loop(el, () => {
      const t = performance.now(); acc += Math.min(300, t - (last || t)); last = t;
      if (acc < 350) return;
      acc = 0;
      if (learnOne() || epoch > 30) stop();
    });
    const stop = () => { anim.stop(); $('[data-a=run]').textContent = '▶ 自动学习'; };
    el.addEventListener('click', (e) => {
      const a = e.target.closest('[data-a]')?.dataset.a, c = e.target.closest('[data-c]');
      if (c) { cls = +c.dataset.c; el.querySelectorAll('[data-c]').forEach((x) => x.classList.toggle('on', x === c)); }
      if (a === 'one') learnOne();
      if (a === 'reset') { stop(); resetLine(); }
      if (a === 'clear') { stop(); pts = []; resetLine(); }
      if (a === 'run') { if (anim.running) stop(); else { last = 0; acc = 350; anim.start(); $('[data-a=run]').textContent = '⏸ 暂停'; } }
    });
    cv.addEventListener('pointerdown', (e) => {
      if ($('.pc-set').value !== 'custom') return;
      const [px, py] = Charts.at(cv, e, W, H), [x, y] = inv(px, py);
      pts.push([+x.toFixed(2), +y.toFixed(2), cls]);
      draw();
    });
    load();
  },
};
