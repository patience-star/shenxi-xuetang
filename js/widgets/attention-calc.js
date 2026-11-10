/* 互动组件：注意力是怎么算的（几何版）
 * 每个词有一个二维向量（可以拖动）。为了看清本质，这里令「查询 = 键 = 值 = 词向量」：
 * 选一个词做查询，它和每个词的点积就是相似度，除以 √d 后做 softmax 得到注意力权重，
 * 输出 = 所有词向量按权重加权求和。勾选「因果掩码」后，每个词只能看它自己和前面的词。 */
Widgets['attention-calc'] = {
  title: '注意力：按相似度加权求和',
  render(el) {
    const words = [
      { w: '小猫', v: [1.6, 1.2] }, { w: '坐在', v: [-1.4, 0.4] }, { w: '垫子', v: [0.4, -1.5] },
      { w: '上', v: [-1.0, -0.8] }, { w: '它', v: [1.3, 0.9] }, { w: '睡着了', v: [-0.2, 1.6] },
    ];
    let q = 4, drag = -1;
    el.innerHTML = `
      <div class="controls"><label>查询词<select class="ac-q">${words.map((x, i) => `<option value="${i}"${i === q ? ' selected' : ''}>${x.w}</option>`).join('')}</select></label>
        <label>缩放 <output class="ac-so"></output><input type="range" class="ac-s" min="0.3" max="3" step="0.1" value="1"></label>
        <label class="inline"><input type="checkbox" class="ac-mask"> 因果掩码（只看前面）</label></div>
      <div class="split">
        <div><canvas class="viz drag ac-cv" aria-label="词向量平面"></canvas><p class="small muted">拖动任意一个点改变它的向量。橙色箭头是查询词的输出：所有词向量按注意力权重的加权平均。</p></div>
        <div><h4>「<span class="ac-qn"></span>」对每个词的注意力</h4><div class="ac-bars"></div><div class="ac-calc small" style="margin-top:8px"></div></div>
      </div>
      <h4>完整的注意力矩阵（每一行是一个查询词，行内加起来等于 1）</h4>
      <div class="ac-heat"></div>`;
    const $ = (s) => el.querySelector(s);
    const cv = $('.ac-cv'), W = 380, H = 300, ctx = Charts.hidpi(cv, W, H), R = 2.2;
    const X = (x) => W / 2 + (x / R) * (W / 2 - 16), Y = (y) => H / 2 - (y / R) * (H / 2 - 16);
    const attRow = (i, scale, mask) => {
      const s = words.map((k, j) => (mask && j > i ? -Infinity : ((words[i].v[0] * k.v[0] + words[i].v[1] * k.v[1]) / Math.SQRT2) * scale));
      return { s, a: NN.softmax(Float32Array.from(s)) };
    };
    function draw() {
      const scale = +$('.ac-s').value, mask = $('.ac-mask').checked, css = Charts.css;
      $('.ac-so').textContent = `×${scale.toFixed(1)}`;
      const { s, a } = attRow(q, scale, mask), out = [0, 1].map((d) => words.reduce((acc, k, j) => acc + a[j] * k.v[d], 0));
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = css('--grid'); ctx.beginPath(); ctx.moveTo(X(-R), Y(0)); ctx.lineTo(X(R), Y(0)); ctx.moveTo(X(0), Y(-R)); ctx.lineTo(X(0), Y(R)); ctx.stroke();
      words.forEach((k, j) => {
        ctx.strokeStyle = j === q ? css('--primary') : css('--axis'); ctx.lineWidth = j === q ? 3 : 1.5;
        ctx.globalAlpha = 0.3 + 0.7 * Math.min(1, a[j] * 2.5);
        ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(k.v[0]), Y(k.v[1])); ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.fillStyle = j === q ? css('--primary') : css('--c1');
        ctx.beginPath(); ctx.arc(X(k.v[0]), Y(k.v[1]), 7, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = css('--text'); ctx.font = `13px ${css('--font')}`; ctx.fillText(k.w, X(k.v[0]) + 9, Y(k.v[1]) - 6);
      });
      ctx.strokeStyle = css('--c2'); ctx.fillStyle = css('--c2'); ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(out[0]), Y(out[1])); ctx.stroke();
      ctx.beginPath(); ctx.arc(X(out[0]), Y(out[1]), 5, 0, Math.PI * 2); ctx.fill();
      $('.ac-qn').textContent = words[q].w;
      Charts.bars($('.ac-bars'), words.map((k, j) => ({ label: k.w, p: a[j] })), { hi: NN.argmax(a) });
      $('.ac-calc').innerHTML = `相似度 = 查询·键 / √2（d = 2）× 缩放：${words.map((k, j) => `${k.w} ${Number.isFinite(s[j]) ? s[j].toFixed(2) : '屏蔽'}`).join('，')}。<br>softmax 之后得到上面的权重；输出向量 = (${out[0].toFixed(2)}, ${out[1].toFixed(2)})。<br>${scale > 2 ? '缩放很大：softmax 变得很「尖」，几乎只看最像的一个词（梯度也会变小）——这就是公式里要除以 √d 的原因。' : scale < 0.6 ? '缩放很小：权重接近平均，注意力「没有重点」。' : ''}`;
      Charts.heatmap($('.ac-heat'), words.map((_, i) => Array.from(attRow(i, scale, mask).a)), { rowLabels: words.map((k) => k.w), colLabels: words.map((k) => k.w), min: 0, max: 1, values: true, labelW: 46, rowTitle: '查询', colTitle: '键' });
    }
    $('.ac-q').addEventListener('change', (e) => { q = +e.target.value; draw(); });
    $('.ac-s').addEventListener('input', draw);
    $('.ac-mask').addEventListener('change', draw);
    cv.addEventListener('pointerdown', (e) => {
      const [x, y] = Charts.at(cv, e, W, H);
      drag = words.findIndex((k) => Math.hypot(X(k.v[0]) - x, Y(k.v[1]) - y) < 16);
      if (drag >= 0) cv.setPointerCapture(e.pointerId);
    });
    cv.addEventListener('pointermove', (e) => {
      if (drag < 0) return;
      const [x, y] = Charts.at(cv, e, W, H);
      words[drag].v = [Math.max(-R, Math.min(R, ((x - W / 2) / (W / 2 - 16)) * R)), Math.max(-R, Math.min(R, (-(y - H / 2) / (H / 2 - 16)) * R))];
      draw();
    });
    cv.addEventListener('pointerup', () => { drag = -1; });
    draw();
  },
};
