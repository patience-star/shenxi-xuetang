/* 互动组件：RNN 的循环与「梯度连乘」
 * 上：一个只有 1 个神经元的 RNN 按时间展开，h_t = tanh(w·h_(t−1) + u·x_t)，同一组 w、u 在每一步重复使用；
 * 下：反向传播穿过时间（BPTT）时，∂h_T/∂h_t 是很多个 w·tanh′ 的乘积——乘数小于 1 就指数衰减（梯度消失），大于 1 就爆炸。 */
Widgets['rnn-cell'] = {
  title: 'RNN：同一个公式，一步一步往下传',
  render(el) {
    const X = [1, 0, 0, 0.5, 0, 0, 0, 0, -1, 0, 0, 0];
    el.innerHTML = `
      <div class="controls">
        <label>循环权重 w = <output class="rc-ow"></output><input type="range" class="rc-w" min="0.1" max="2.5" step="0.05" value="0.8"></label>
        <label>输入权重 u = <output class="rc-ou"></output><input type="range" class="rc-u" min="0" max="2" step="0.1" value="1"></label>
        <span class="row"><button type="button" class="btn sm" data-a="step">走一步</button><button type="button" class="btn sm ghost" data-a="all">全部展开</button><button type="button" class="btn sm ghost" data-a="reset">重置</button></span>
      </div>
      <div class="diagram"><svg class="rc-svg" viewBox="0 0 760 200" role="img" aria-label="按时间展开的 RNN"></svg></div>
      <p class="verdict rc-v"></p>
      <h4>反向传播穿过时间：最后一步的梯度，传回第 t 步还剩多少？（序列长 30，对数刻度）</h4>
      <div class="rc-chart"></div>
      <p class="small muted rc-g"></p>`;
    const $ = (s) => el.querySelector(s);
    let T = 0;
    const run = (w, u, xs) => { let h = 0; const hs = [], zs = []; xs.forEach((x) => { const z = w * h + u * x; zs.push(z); h = Math.tanh(z); hs.push(h); }); return { hs, zs }; };
    function draw() {
      const w = +$('.rc-w').value, u = +$('.rc-u').value, { hs } = run(w, u, X);
      $('.rc-ow').textContent = w.toFixed(2); $('.rc-ou').textContent = u.toFixed(1);
      let s = '';
      X.forEach((x, t) => {
        const cx = 40 + t * 62, on = t < T, h = hs[t];
        const col = h >= 0 ? `color-mix(in srgb, var(--pos) ${Math.round(Math.abs(h) * 80)}%, var(--surface))` : `color-mix(in srgb, var(--neg) ${Math.round(Math.abs(h) * 80)}%, var(--surface))`;
        s += `<text x="${cx}" y="188" text-anchor="middle" class="sub">x${t + 1}=${x}</text>`;
        s += `<line x1="${cx}" y1="170" x2="${cx}" y2="122" class="arrow ${on ? 'hl' : ''}"/>`;
        s += `<rect x="${cx - 22}" y="78" width="44" height="44" rx="9" style="fill:${on ? col : 'var(--surface-2)'};stroke:${t === T - 1 ? 'var(--c2)' : 'var(--border)'};stroke-width:${t === T - 1 ? 2.5 : 1.5}"/>`;
        s += `<text x="${cx}" y="105" text-anchor="middle" style="font-size:12px">${on ? h.toFixed(2) : 'h' + (t + 1)}</text>`;
        if (t < X.length - 1) s += `<line x1="${cx + 22}" y1="100" x2="${cx + 40}" y2="100" class="arrow ${t < T - 1 ? 'hl' : ''}"/><polygon points="${cx + 40},100 ${cx + 33},96 ${cx + 33},104" class="head ${t < T - 1 ? 'hl' : ''}"/>`;
      });
      s += `<text x="380" y="40" text-anchor="middle" class="sub">每一步：h_t = tanh( w × h_(t−1) + u × x_t )——所有时间步共用同一组 w、u</text>`;
      $('.rc-svg').innerHTML = s;
      $('.rc-v').innerHTML = T ? `第 ${T} 步：h = tanh(${w.toFixed(2)} × ${(T > 1 ? hs[T - 2] : 0).toFixed(2)} + ${u} × ${X[T - 1]}) = <b>${hs[T - 1].toFixed(3)}</b>。隐状态 h 是网络的「记忆」：第 1 步的输入 1 之后再没出现，但它的影响还在一步步往后传（w 越小，忘得越快）。`
        : '点「走一步」：隐状态 h（方框里的数）把过去的信息带到下一步。蓝色为正、红色为负。';
      // BPTT：∂h_30/∂h_t = Π w·tanh′(z_k)
      const xs = Array.from({ length: 30 }, (_, k) => (k % 7 === 0 ? 0.6 : 0.1)), { zs } = run(w, u, xs);
      const g = new Array(30).fill(0);
      let prod = 1;
      for (let t = 29; t >= 0; t--) { g[t] = prod; prod *= w * (1 - Math.tanh(zs[t]) ** 2); }
      Charts.columns($('.rc-chart'), g.map((v, t) => ({ label: String(t + 1), value: Math.log10(Math.max(v, 1e-30)) })), { format: (v) => `1e${Math.round(v)}`, height: 170, table: false, color: () => 'var(--c2)' });
      const first = g[0];
      $('.rc-g').textContent = `梯度传回第 1 步时乘了 29 次「w × tanh′」，结果约为 ${first.toExponential(1)}。${first < 1e-3 ? '这就是梯度消失：很久以前的输入几乎收不到学习信号，普通 RNN 因此记不住长距离的依赖。' : first > 1e3 ? '这就是梯度爆炸：一次更新就可能把参数推飞，所以训练 RNN 要做梯度裁剪。' : '这个 w 让梯度大致保持住了——但真实网络里很难一直维持在这个「刚刚好」的状态，LSTM 用门控结构来解决。'}`;
    }
    el.addEventListener('click', (e) => {
      const a = e.target.closest('[data-a]')?.dataset.a;
      if (a === 'step') T = Math.min(X.length, T + 1);
      if (a === 'all') T = X.length;
      if (a === 'reset') T = 0;
      if (a) draw();
    });
    el.querySelectorAll('input').forEach((i) => i.addEventListener('input', draw));
    draw();
  },
};
