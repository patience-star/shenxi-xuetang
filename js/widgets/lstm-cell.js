/* 互动组件：LSTM 单元里的三个门
 * 细胞状态 c 像一条传送带：遗忘门 f 决定旧记忆留多少，输入门 i 决定写入多少新内容 g，
 * 输出门 o 决定读出多少作为隐状态 h。拖动滑块看每个量怎么变；下面的「记忆实验」对比遗忘门不同取值时记忆能保持多久。 */
Widgets['lstm-cell'] = {
  title: 'LSTM：一条带三个阀门的传送带',
  render(el) {
    const S = { c0: 1.0, f: 0.9, i: 0.5, g: 0.8, o: 0.7 };
    const NAMES = { c0: '旧记忆 c_(t−1)', f: '遗忘门 f', i: '输入门 i', g: '新内容 g', o: '输出门 o' };
    el.innerHTML = `
      <div class="controls">${Object.keys(S).map((k) => `<label>${NAMES[k]} = <output class="lc-o${k}"></output><input type="range" class="lc-r" data-k="${k}" min="${k === 'g' || k === 'c0' ? -1.5 : 0}" max="${k === 'g' || k === 'c0' ? 1.5 : 1}" step="0.05" value="${S[k]}"></label>`).join('')}</div>
      <div class="diagram"><svg class="lc-svg" viewBox="0 0 720 250" role="img" aria-label="LSTM 单元"></svg></div>
      <p class="verdict lc-v"></p>
      <h4>记忆实验：第 1 步写入一个值（c = 1），之后不再写入，每一步只乘遗忘门 f</h4>
      <div class="lc-chart"></div>
      <p class="small muted">普通 RNN 每一步都要把记忆乘上权重再过 tanh，信息很快被冲淡；LSTM 只要学会让遗忘门保持接近 1，信息就能沿传送带保存几十步，梯度也沿这条路顺畅地传回去。</p>`;
    const $ = (s) => el.querySelector(s);
    const valve = (x, y, v, label) => `<g><circle cx="${x}" cy="${y}" r="20" style="fill:color-mix(in srgb, var(--c1) ${Math.round(v * 80)}%, var(--surface));stroke:var(--c1);stroke-width:1.5"/><text x="${x}" y="${y + 4}" text-anchor="middle" style="font-size:12px;font-weight:600">${v.toFixed(2)}</text><text x="${x}" y="${y + 36}" text-anchor="middle" class="sub">${label}</text></g>`;
    function draw() {
      Object.keys(S).forEach((k) => { $(`.lc-o${k}`).textContent = S[k].toFixed(2); });
      const kept = S.f * S.c0, wrote = S.i * S.g, c = kept + wrote, h = S.o * Math.tanh(c);
      const belt = (w) => `stroke:var(--c4);stroke-width:${Math.max(2, Math.min(14, 2 + Math.abs(w) * 6))};stroke-linecap:round`;
      let s = '';
      s += `<text x="20" y="34" class="sub">细胞状态（传送带）</text>`;
      s += `<line x1="40" y1="60" x2="200" y2="60" style="${belt(S.c0)}"/><line x1="240" y1="60" x2="420" y2="60" style="${belt(kept)}"/><line x1="460" y1="60" x2="690" y2="60" style="${belt(c)}"/>`;
      s += `<text x="70" y="86" class="sub">c_(t−1) = ${S.c0.toFixed(2)}</text><text x="600" y="86" class="sub">c_t = ${c.toFixed(2)}</text>`;
      s += `<circle cx="220" cy="60" r="16" class="node"/><text x="220" y="65" text-anchor="middle">×</text>`;
      s += `<circle cx="440" cy="60" r="16" class="node"/><text x="440" y="65" text-anchor="middle">+</text>`;
      s += valve(220, 150, S.f, '遗忘门 f');
      s += `<line x1="220" y1="130" x2="220" y2="78" class="arrow hl"/>`;
      s += valve(400, 190, S.i, '输入门 i') + `<text x="470" y="196" style="font-size:12px">× 新内容 g = ${S.g.toFixed(2)}</text>`;
      s += `<line x1="430" y1="172" x2="440" y2="78" class="arrow hl"/>`;
      s += `<line x1="600" y1="60" x2="600" y2="125" class="arrow"/><rect x="560" y="125" width="80" height="26" rx="6" class="box alt"/><text x="600" y="143" text-anchor="middle" style="font-size:12px">tanh</text>`;
      s += valve(660, 200, S.o, '输出门 o') + `<line x1="640" y1="151" x2="655" y2="180" class="arrow"/>`;
      s += `<text x="505" y="240" class="sub">隐状态 h_t = o × tanh(c_t) = ${h.toFixed(2)}</text>`;
      $('.lc-svg').innerHTML = s;
      $('.lc-v').innerHTML = `c_t = f × c_(t−1) + i × g = ${S.f.toFixed(2)} × ${S.c0.toFixed(2)} + ${S.i.toFixed(2)} × ${S.g.toFixed(2)} = <b>${c.toFixed(2)}</b>；h_t = o × tanh(c_t) = <b>${h.toFixed(2)}</b>。`
        + (S.f > 0.95 && S.i < 0.1 ? '　遗忘门全开、输入门关闭：原样保留旧记忆。' : S.f < 0.1 ? '　遗忘门关闭：旧记忆被清空（比如一句诗写完、开始新的一句）。' : '');
      const steps = Array.from({ length: 30 }, (_, t) => String(t + 1));
      Charts.line($('.lc-chart'), { labels: steps, series: [0.5, 0.8, S.f, 0.99].map((f, k) => ({ name: k === 2 ? `当前遗忘门 f = ${f.toFixed(2)}` : `f = ${f}`, values: steps.map((_, t) => f ** t) })) }, { format: (v) => v.toFixed(2), height: 200, table: false, yMin: 0 });
    }
    el.querySelectorAll('.lc-r').forEach((r) => r.addEventListener('input', () => { S[r.dataset.k] = +r.value; draw(); }));
    draw();
  },
};
