/* 互动组件：图片就是一个数字矩阵
 * 数据：pytorch/01_tensor.py 真实读取的一批 MNIST 图片（前 16 张）。
 * 点缩略图选一张，指针 / 手指在大图上移动，查看每个像素的值和它在张量里的下标。 */
Widgets['pixel-grid'] = {
  title: '图片就是一个数字矩阵',
  async render(el, opts) {
    const py = opts.py || (await App.loadPy(opts.script || '01_tensor'));
    const D = NN.decode(py.extras.digits), labels = py.extras.labels;
    const n = D.shape[0], S = 28, W = 420, cs = W / S;
    el.innerHTML = `
      <p class="small muted">下面是脚本真实读取的一批图片中的前 ${n} 张：张量 <code>images</code> 的形状是 <code>[64, 1, 28, 28]</code>，这里取了 <code>images[:${n}]</code>。点一张缩略图，再把指针（手机上用手指点）移到大图上。</p>
      <div class="thumbs pg-thumbs"></div>
      <div class="split" style="margin-top:12px">
        <canvas class="viz pg-big" aria-label="放大的 28×28 像素网格"></canvas>
        <div>
          <p class="pg-expr"></p>
          <div class="pg-nb"></div>
          <p class="small muted">每个格子是一个像素：0 是背景，255 是最亮的笔迹。<code>ToTensor()</code> 把它们除以 255 变成 0~1 的小数——神经网络「看到」的就是这 28 × 28 = 784 个数字，它并不知道这是一张图。</p>
        </div>
      </div>`;
    const thumbs = el.querySelector('.pg-thumbs'), big = el.querySelector('.pg-big');
    const pix = (i) => D.d.subarray(i * S * S, (i + 1) * S * S);
    let cur = 0, hover = [12, 14];
    for (let i = 0; i < n; i++) {
      const f = document.createElement('figure');
      const c = Charts.canvasOf(pix(i), S, S, { mode: 'ink', min: 0, max: 255 });
      c.tabIndex = 0;
      c.setAttribute('role', 'button');
      c.setAttribute('aria-label', `第 ${i} 张，标签 ${labels[i]}`);
      f.append(c, Object.assign(document.createElement('figcaption'), { textContent: `[${i}] 是 ${labels[i]}` }));
      thumbs.appendChild(f);
      const pick = () => { cur = i; draw(); };
      c.addEventListener('click', pick);
      c.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
    }
    const ctx = Charts.hidpi(big, W, W);

    function draw() {
      thumbs.querySelectorAll('canvas').forEach((c, i) => c.classList.toggle('on', i === cur));
      const p = pix(cur), bg = Charts.rgb('--surface'), ink = Charts.rgb('--text');
      for (let r = 0; r < S; r++) for (let c = 0; c < S; c++) {
        const t = p[r * S + c] / 255;
        ctx.fillStyle = `rgb(${bg.map((b, k) => Math.round(b + (ink[k] - b) * t)).join(',')})`;
        ctx.fillRect(c * cs, r * cs, cs, cs);
      }
      ctx.strokeStyle = Charts.css('--grid');
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      for (let k = 0; k <= S; k++) { ctx.moveTo(k * cs, 0); ctx.lineTo(k * cs, W); ctx.moveTo(0, k * cs); ctx.lineTo(W, k * cs); }
      ctx.stroke();
      const [r, c] = hover;
      ctx.strokeStyle = Charts.css('--c2');
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.strokeRect((c - 2) * cs, (r - 2) * cs, 5 * cs, 5 * cs);
      ctx.setLineDash([]);
      ctx.strokeRect(c * cs, r * cs, cs, cs);
      const v = p[r * S + c];
      el.querySelector('.pg-expr').innerHTML = `<code>images[${cur}, 0, ${r}, ${c}]</code> = <b>${v}</b> ÷ 255 = <b>${(v / 255).toFixed(3)}</b><br>
        <span class="small muted">下标依次是：第 ${cur} 张 · 第 0 个通道 · 第 ${r} 行 · 第 ${c} 列</span>`;
      let rows = '';
      for (let dr = -2; dr <= 2; dr++) {
        rows += '<tr>';
        for (let dc = -2; dc <= 2; dc++) {
          const rr = r + dr, cc = c + dc, ok = rr >= 0 && rr < S && cc >= 0 && cc < S;
          rows += `<td class="${!dr && !dc ? 'hi' : ''}">${ok ? p[rr * S + cc] : ''}</td>`;
        }
        rows += '</tr>';
      }
      el.querySelector('.pg-nb').innerHTML = `<p class="small muted" style="margin-bottom:4px">虚线框里 5 × 5 个像素的值：</p><table class="nb"><tbody>${rows}</tbody></table>`;
    }
    const move = (e) => {
      const [x, y] = Charts.at(big, e, W, W);
      const h = [Math.max(0, Math.min(S - 1, Math.floor(y / cs))), Math.max(0, Math.min(S - 1, Math.floor(x / cs)))];
      if (h[0] !== hover[0] || h[1] !== hover[1]) { hover = h; draw(); }
    };
    big.addEventListener('pointermove', move);
    big.addEventListener('pointerdown', move);
    draw();
  },
};
