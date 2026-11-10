/* 互动组件：正弦位置编码
 * 注意力本身不区分顺序（打乱输入顺序，结果只是跟着打乱），所以要给每个位置加一个独特的「位置向量」。
 * 上：位置编码矩阵（行 = 位置，列 = 维度）；下：选中位置与其它位置的相似度（点积）——离得越近越相似。 */
Widgets['pos-encoding'] = {
  title: '位置编码：给每个位置一个独特的「指纹」',
  render(el) {
    const P = 50, D = 64;
    const pe = (pos) => { const v = new Float32Array(D); for (let i = 0; i < D; i += 2) { const div = Math.exp(i * (-Math.log(10000) / D)); v[i] = Math.sin(pos * div); v[i + 1] = Math.cos(pos * div); } return v; };
    const M = Array.from({ length: P }, (_, p) => pe(p));
    el.innerHTML = `
      <div class="controls"><label>选中位置 p = <output class="pe-o"></output><input type="range" class="pe-p" min="0" max="${P - 1}" value="10"></label></div>
      <div class="split">
        <div><canvas class="pe-cv viz" aria-label="位置编码矩阵"></canvas><p class="small muted">每一行是一个位置的 64 维编码（蓝正红负）。左边的维度频率高（变化快），右边的维度频率低（变化慢）——像钟表的秒针、分针、时针，合起来能唯一确定一个时刻。</p></div>
        <div><h4>位置 <span class="pe-n"></span> 与各位置的相似度</h4><div class="pe-sim"></div></div>
      </div>
      <p class="small muted">公式：PE(p, 2i) = sin(p / 10000^(2i/d))，PE(p, 2i+1) = cos(p / 10000^(2i/d))。本课案例的 Transformer 用的正是它（d = 64）。</p>`;
    const cv = el.querySelector('.pe-cv'), flat = new Float32Array(P * D);
    M.forEach((v, p) => flat.set(v, p * D));
    function draw() {
      const p = +el.querySelector('.pe-p').value;
      el.querySelector('.pe-o').textContent = p;
      el.querySelector('.pe-n').textContent = p;
      const marked = Float32Array.from(flat);
      Charts.img(cv, marked, D, P, { mode: 'div', min: -1, max: 1 });
      cv.style.aspectRatio = `${D} / ${P}`;
      const ctx = cv.getContext('2d');
      ctx.strokeStyle = Charts.css('--c4'); ctx.lineWidth = 1; ctx.strokeRect(0, p, D, 1);
      const sims = M.map((v) => v.reduce((s, x, k) => s + x * M[p][k], 0) / (D / 2));
      Charts.line(el.querySelector('.pe-sim'), { labels: M.map((_, k) => String(k)), series: [{ name: '相似度', values: sims }] }, { format: (v) => v.toFixed(2), height: 220, table: false, yMin: -1 });
    }
    el.querySelector('.pe-p').addEventListener('input', draw);
    draw();
  },
};
