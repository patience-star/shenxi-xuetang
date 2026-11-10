/* 互动组件：二维隐空间漫游（第 12 课，VAE 与普通自编码器对比）
 * 左：2000 张测试图片编码后的坐标（颜色 = 真实数字）。在平面上拖动十字，右边实时显示解码器从这一点「画」出的图；
 * 下：把整个隐空间按网格解码出来的「地图」。还可以手写一个数字，看 VAE 把它放到隐空间的哪里。 */
Widgets['latent-explorer'] = {
  title: '隐空间漫游：拖动十字，看解码器画出什么',
  async render(el, opts) {
    const name = opts.model || opts.py?.name || '12_vae_mnist';
    const py = opts.py?.name === name ? opts.py : await App.loadPy(name);
    const W = NN.weights(py), M = NN.models[name], ys = py.extras.y;
    const Z = { vae: NN.decode(py.extras.z_vae), ae: NN.decode(py.extras.z_ae) };
    const color = (d) => `hsl(${(d * 36 + 200) % 360} 70% 50%)`;
    let kind = opts.kind || 'vae', cur = [0, 0], dragging = false, mark = null;
    el.innerHTML = `
      <div class="seg le-seg"><button type="button" data-k="vae">VAE</button><button type="button" data-k="ae">普通自编码器</button></div>
      <div class="split" style="margin-top:10px">
        <canvas class="viz drag le-cv" aria-label="二维隐空间散点图"></canvas>
        <div>
          <div class="row" style="align-items:flex-end;gap:14px"><canvas class="le-img big-thumb" style="max-width:160px"></canvas><div class="le-z small"></div></div>
          <div class="row" style="margin-top:8px"><button type="button" class="btn sm ghost" data-a="rand">从 N(0, 1) 随机取一点</button></div>
          <div class="le-padbox" style="margin-top:12px"><p class="small muted" style="margin:0 0 4px">手写一个数字，看 VAE 把它编码到哪里（★）：</p><div class="le-pad" style="max-width:160px"></div><button type="button" class="btn sm ghost" data-a="clear" style="margin-top:6px">清除</button></div>
        </div>
      </div>
      <p class="verdict le-v"></p>
      <h4>隐空间地图：在 <span class="le-range"></span> 范围内按 12 × 12 网格解码</h4>
      <canvas class="le-map viz" style="max-width:480px" aria-label="隐空间网格解码"></canvas>`;
    const $ = (s) => el.querySelector(s);
    const cv = $('.le-cv'), S = 380, ctx = Charts.hidpi(cv, S, S);
    let range;
    const toPx = (z) => [((z[0] - range[0]) / (range[1] - range[0])) * S, S - ((z[1] - range[2]) / (range[3] - range[2])) * S];
    const toZ = (px, py2) => [range[0] + (px / S) * (range[1] - range[0]), range[2] + ((S - py2) / S) * (range[3] - range[2])];
    const decode = (z) => M.decode(W, Float32Array.from(z), kind === 'vae' ? 'vae.dec.' : 'ae_dec.');
    function setKind(k) {
      kind = k;
      el.querySelectorAll('.le-seg button').forEach((b) => b.classList.toggle('on', b.dataset.k === k));
      const z = Z[k].d, n = ys.length;
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (let i = 0; i < n; i++) { x0 = Math.min(x0, z[2 * i]); x1 = Math.max(x1, z[2 * i]); y0 = Math.min(y0, z[2 * i + 1]); y1 = Math.max(y1, z[2 * i + 1]); }
      const px = (x1 - x0) * 0.08, py2 = (y1 - y0) * 0.08;
      range = [x0 - px, x1 + px, y0 - py2, y1 + py2];
      cur = [(x0 + x1) / 2, (y0 + y1) / 2];
      mark = null;
      $('.le-padbox').hidden = k !== 'vae';
      $('.le-range').textContent = `横轴 ${range[0].toFixed(1)}~${range[1].toFixed(1)}、纵轴 ${range[2].toFixed(1)}~${range[3].toFixed(1)}`;
      // 地图
      const G = 12, map = new Float32Array(G * 28 * G * 28);
      for (let gy = 0; gy < G; gy++) for (let gx = 0; gx < G; gx++) {
        const img = decode([range[0] + ((gx + 0.5) / G) * (range[1] - range[0]), range[3] - ((gy + 0.5) / G) * (range[3] - range[2])]);
        for (let r = 0; r < 28; r++) for (let c = 0; c < 28; c++) map[(gy * 28 + r) * G * 28 + gx * 28 + c] = img[r * 28 + c];
      }
      Charts.img($('.le-map'), map, G * 28, G * 28, { mode: 'ink', min: 0, max: 1 });
      $('.le-map').style.aspectRatio = '1';
      draw();
    }
    function draw() {
      const z = Z[kind].d, css = Charts.css;
      ctx.clearRect(0, 0, S, S);
      ctx.fillStyle = css('--surface-2'); ctx.fillRect(0, 0, S, S);
      const cen = Array.from({ length: 10 }, () => [0, 0, 0]);
      for (let i = 0; i < ys.length; i++) {
        const [x, y] = toPx([z[2 * i], z[2 * i + 1]]);
        ctx.fillStyle = color(ys[i]); ctx.globalAlpha = 0.55; ctx.fillRect(x - 1.5, y - 1.5, 3, 3);
        cen[ys[i]][0] += x; cen[ys[i]][1] += y; cen[ys[i]][2]++;
      }
      ctx.globalAlpha = 1; ctx.font = `bold 16px ${css('--font')}`;
      cen.forEach(([x, y, n], d) => { if (!n) return; ctx.fillStyle = css('--surface'); ctx.fillText(d, x / n - 4, y / n + 6); ctx.fillStyle = color(d); ctx.fillText(d, x / n - 5, y / n + 5); });
      const [cx, cy] = toPx(cur);
      ctx.strokeStyle = css('--text'); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(cx - 12, cy); ctx.lineTo(cx + 12, cy); ctx.moveTo(cx, cy - 12); ctx.lineTo(cx, cy + 12); ctx.stroke();
      if (mark) { const [mx, my] = toPx(mark); ctx.fillStyle = css('--c2'); ctx.font = '20px sans-serif'; ctx.fillText('★', mx - 8, my + 7); }
      const img = decode(cur);
      Charts.img($('.le-img'), img, 28, 28, { mode: 'ink', min: 0, max: 1 });
      $('.le-z').innerHTML = `隐向量 z = (${cur[0].toFixed(2)}, ${cur[1].toFixed(2)})<br><span class="muted">只有 2 个数，解码器就从它「画」出了 784 个像素</span>`;
      $('.le-v').textContent = kind === 'vae'
        ? 'VAE 的隐空间紧凑而连续：各个数字的区域彼此相邻、中间平滑过渡，在原点附近随便取一点都能解码出像样的数字——所以 VAE 可以用来「生成」。'
        : '普通自编码器的隐空间：各数字的点散得很开，数字之间留着大片空白；拖到空白处，解码出来常常是模糊的四不像。点「随机取一点」试试——它只会压缩还原，不适合直接采样生成。';
    }
    const move = (e) => { const [x, y] = Charts.at(cv, e, S, S); cur = toZ(x, y); draw(); };
    cv.addEventListener('pointerdown', (e) => { dragging = true; cv.setPointerCapture(e.pointerId); move(e); });
    cv.addEventListener('pointermove', (e) => { if (dragging) move(e); });
    cv.addEventListener('pointerup', () => { dragging = false; });
    const r = NN.rng(Date.now() % 1e6);
    const pad = Charts.pad($('.le-pad'), {
      onChange: (x) => { if (pad.isEmpty()) { mark = null; draw(); return; } mark = Array.from(M.encode(W, x)); cur = mark.slice(); draw(); },
    });
    el.addEventListener('click', (e) => {
      const k = e.target.closest('[data-k]'), a = e.target.closest('[data-a]')?.dataset.a;
      if (k) setKind(k.dataset.k);
      if (a === 'rand') { cur = [r.randn(), r.randn()]; draw(); }
      if (a === 'clear') pad.clear();
    });
    setKind(kind);
  },
};
