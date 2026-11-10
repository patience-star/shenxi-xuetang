/* 互动组件：卷积是怎么算的
 * ① 数字网格：3×3 卷积核在输入上滑动，每一步把对应位置相乘再相加，得到输出的一个格子；可调步幅、填充。
 * ② 真实图片：同样的卷积核作用在一张手写数字上，看特征图、ReLU 和最大池化之后的样子。 */
Widgets['conv-viz'] = {
  title: '卷积：一个小窗口在图上滑动',
  async render(el) {
    const KERNELS = {
      vert: { name: '竖边检测', k: [1, 0, -1, 1, 0, -1, 1, 0, -1] },
      horz: { name: '横边检测', k: [1, 1, 1, 0, 0, 0, -1, -1, -1] },
      sharp: { name: '锐化', k: [0, -1, 0, -1, 5, -1, 0, -1, 0] },
      blur: { name: '求和（模糊）', k: [1, 1, 1, 1, 1, 1, 1, 1, 1] },
    };
    const INPUTS = {
      edge: { name: '竖直边界', g: (r, c) => (c >= 3 ? 1 : 0) },
      cross: { name: '十字', g: (r, c) => (r === 3 || c === 3 ? 1 : 0) },
      diag: { name: '对角线', g: (r, c) => (r === c ? 1 : 0) },
    };
    const N = 7;
    el.innerHTML = `
      <div class="seg cv-tabs"><button type="button" class="on" data-t="g">① 数字网格</button><button type="button" data-t="i">② 真实图片</button></div>
      <div class="controls">
        <label>卷积核<select class="cv-k">${Object.entries(KERNELS).map(([k, v]) => `<option value="${k}">${v.name}</option>`).join('')}<option value="custom">自定义（点卷积核的格子改数）</option></select></label>
        <label class="cv-g-only">输入<select class="cv-in">${Object.entries(INPUTS).map(([k, v]) => `<option value="${k}">${v.name}</option>`).join('')}</select></label>
        <label class="cv-g-only">步幅 stride<select class="cv-s"><option>1</option><option>2</option></select></label>
        <label class="cv-g-only">填充 padding<select class="cv-p"><option>0</option><option>1</option></select></label>
      </div>
      <div class="cv-grid-mode">
        <div class="row cv-g-only"><button type="button" class="btn sm" data-a="next">下一步</button><button type="button" class="btn sm ghost" data-a="run">▶ 自动</button><button type="button" class="btn sm ghost" data-a="all">全部算完</button><button type="button" class="btn sm ghost" data-a="reset">重置</button></div>
        <div class="row cv-boards" style="align-items:flex-start;gap:18px;margin-top:10px"></div>
        <p class="verdict cv-calc"></p>
        <p class="small muted cv-size"></p>
      </div>
      <div class="cv-img-mode" hidden>
        <div class="row"><button type="button" class="btn sm ghost" data-a="prev">← 上一张</button><button type="button" class="btn sm ghost" data-a="nextimg">下一张 →</button></div>
        <div class="thumbs cv-imgs" style="grid-template-columns:repeat(4,minmax(70px,160px));margin-top:10px"></div>
        <p class="small cv-img-note"></p>
      </div>`;
    const $ = (s) => el.querySelector(s);
    let kernel = KERNELS.vert.k.slice(), pos = 0, timer = null, imgIdx = 0;
    const py = await App.loadPy('01_tensor');
    const digits = NN.decode(py.extras.digits);

    const conv = (inp, n, k, s, p) => {
      const m = Math.floor((n + 2 * p - 3) / s) + 1, out = new Float32Array(m * m);
      for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) {
        let v = 0;
        for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) { const r = i * s - p + a, c = j * s - p + b; if (r >= 0 && r < n && c >= 0 && c < n) v += inp[r * n + c] * k[a * 3 + b]; }
        out[i * m + j] = v;
      }
      return { out, m };
    };

    function drawGrid() {
      const s = +$('.cv-s').value, p = +$('.cv-p').value, gen = INPUTS[$('.cv-in').value].g;
      const inp = Float32Array.from({ length: N * N }, (_, i) => gen(Math.floor(i / N), i % N));
      const { out, m } = conv(inp, N, kernel, s, p), total = m * m;
      pos = Math.min(pos, total);
      const cur = pos > 0 ? pos - 1 : -1, ci = Math.floor(cur / m), cj = cur % m;
      const P = N + 2 * p;
      // 输入（含填充）
      let html = `<div><div class="small muted">输入 ${N}×${N}${p ? '（灰色是补的 0）' : ''}</div><div class="ts-grid" style="grid-template-columns:repeat(${P},auto)">`;
      for (let r = 0; r < P; r++) for (let c = 0; c < P; c++) {
        const rr = r - p, cc = c - p, pad = rr < 0 || rr >= N || cc < 0 || cc >= N, v = pad ? 0 : inp[rr * N + cc];
        const inWin = cur >= 0 && r >= ci * s && r < ci * s + 3 && c >= cj * s && c < cj * s + 3;
        html += `<span class="ts-cell" style="${pad ? 'opacity:.4;' : ''}${v ? 'background:var(--seq-400);color:#fff;' : ''}${inWin ? 'outline:3px solid var(--c2);outline-offset:-2px;' : ''}">${v}</span>`;
      }
      html += '</div></div>';
      // 卷积核
      html += `<div><div class="small muted">卷积核 3×3</div><div class="ts-grid cv-kgrid" style="grid-template-columns:repeat(3,auto)">${kernel.map((v, i) => `<button type="button" class="ts-cell" data-ki="${i}" style="border:0;cursor:pointer;background:${v > 0 ? 'var(--primary-weak)' : v < 0 ? 'var(--err-bg)' : 'var(--surface-2)'};font:inherit">${v}</button>`).join('')}</div></div>`;
      // 输出
      const mx = Math.max(1, ...Array.from(out, Math.abs));
      html += `<div><div class="small muted">输出 ${m}×${m}</div><div class="ts-grid" style="grid-template-columns:repeat(${m},auto)">${Array.from(out, (v, i) => {
        const done = i < pos, t = Math.min(1, Math.abs(v) / mx);
        return `<span class="ts-cell" style="${done ? `background:${v >= 0 ? `color-mix(in srgb, var(--pos) ${Math.round(t * 70)}%, var(--surface-2))` : `color-mix(in srgb, var(--neg) ${Math.round(t * 70)}%, var(--surface-2))`};` : ''}${i === cur ? 'outline:3px solid var(--c2);outline-offset:-2px;' : ''}">${done ? v : ''}</span>`;
      }).join('')}</div></div>`;
      $('.cv-boards').innerHTML = html;
      if (cur >= 0) {
        const terms = [];
        for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) { const r = ci * s - p + a, c = cj * s - p + b, v = r >= 0 && r < N && c >= 0 && c < N ? inp[r * N + c] : 0; terms.push(`${v}×${kernel[a * 3 + b] < 0 ? `(${kernel[a * 3 + b]})` : kernel[a * 3 + b]}`); }
        $('.cv-calc').innerHTML = `输出第 ${ci} 行第 ${cj} 列 = ${terms.join(' + ')} = <b>${out[cur]}</b>`;
      } else $('.cv-calc').textContent = '点「下一步」：橙色框是卷积核当前覆盖的区域，对应位置相乘、全部加起来，就是输出的一个格子。';
      $('.cv-size').innerHTML = `输出边长 = ⌊(n + 2p − k) / s⌋ + 1 = ⌊(${N} + 2×${p} − 3) / ${s}⌋ + 1 = <b>${m}</b>。正数（蓝）表示「这里很像卷积核要找的图案」，负数（红）表示「正好相反」。`;
    }

    function drawImage() {
      const n = 28, x = Float32Array.from(digits.d.subarray(imgIdx * 784, (imgIdx + 1) * 784), (v) => v / 255);
      const { out } = conv(x, n, kernel, 1, 1), act = out.map((v) => Math.max(0, v));
      const pooled = new Float32Array(196);
      for (let i = 0; i < 14; i++) for (let j = 0; j < 14; j++) pooled[i * 14 + j] = Math.max(act[2 * i * 28 + 2 * j], act[2 * i * 28 + 2 * j + 1], act[(2 * i + 1) * 28 + 2 * j], act[(2 * i + 1) * 28 + 2 * j + 1]);
      const box = $('.cv-imgs');
      box.innerHTML = '';
      [[x, 28, { mode: 'ink', min: 0, max: 1 }, '输入 28×28'], [out, 28, { mode: 'div' }, '卷积后（蓝正红负）'], [act, 28, { mode: 'seq' }, 'ReLU：只留正数'], [pooled, 14, { mode: 'seq' }, '2×2 最大池化 → 14×14']].forEach(([d, sz, o, cap]) => {
        const f = document.createElement('figure');
        f.append(Charts.canvasOf(d, sz, sz, o), Object.assign(document.createElement('figcaption'), { textContent: cap }));
        box.appendChild(f);
      });
      const kn = $('.cv-k').value;
      $('.cv-img-note').textContent = kn === 'vert' ? '竖边检测核 = 左边的像素减右边的像素：笔画的左边缘为负（红）、右边缘为正（蓝）。它只对竖直方向的边缘有反应，横笔画的上下边缘几乎不响应。ReLU 只保留了正的那一侧。'
        : kn === 'horz' ? '横边检测核：只对水平方向的边缘有反应。一个卷积层有很多个这样的核，各自负责找一种图案。'
        : kn === 'blur' ? '求和核把每个点周围 3×3 的亮度加起来，图像变模糊、变粗。'
        : kn === 'sharp' ? '锐化核放大中心、减去邻居，边缘更突出。'
        : '自定义卷积核：试着设计一个能找「斜线」的核。';
    }
    const draw = () => ($('.cv-img-mode').hidden ? drawGrid() : drawImage());
    const stop = () => { clearInterval(timer); timer = null; $('[data-a=run]').textContent = '▶ 自动'; };
    el.addEventListener('click', (e) => {
      const t = e.target.closest('[data-t]'), a = e.target.closest('[data-a]')?.dataset.a, ki = e.target.closest('[data-ki]');
      if (t) {
        el.querySelectorAll('.cv-tabs button').forEach((b) => b.classList.toggle('on', b === t));
        $('.cv-grid-mode').hidden = t.dataset.t !== 'g';
        $('.cv-img-mode').hidden = t.dataset.t !== 'i';
        el.querySelectorAll('.cv-g-only').forEach((x) => { x.hidden = t.dataset.t !== 'g'; });
        stop(); draw();
      }
      if (ki) { const i = +ki.dataset.ki; kernel[i] = kernel[i] >= 2 ? -1 : kernel[i] + 1; $('.cv-k').value = 'custom'; draw(); }
      if (a === 'next') { pos++; drawGrid(); }
      if (a === 'all') { stop(); pos = 999; drawGrid(); }
      if (a === 'reset') { stop(); pos = 0; drawGrid(); }
      if (a === 'run') {
        if (timer) stop();
        else { timer = setInterval(() => { if (!el.isConnected) return stop(); pos++; drawGrid(); if (pos >= 49) stop(); }, 600); $('[data-a=run]').textContent = '⏸ 暂停'; }
      }
      if (a === 'prev' || a === 'nextimg') { imgIdx = (imgIdx + (a === 'prev' ? -1 : 1) + digits.shape[0]) % digits.shape[0]; drawImage(); }
    });
    $('.cv-k').addEventListener('change', () => { const k = $('.cv-k').value; if (KERNELS[k]) kernel = KERNELS[k].k.slice(); pos = 0; draw(); });
    ['.cv-in', '.cv-s', '.cv-p'].forEach((s) => $(s).addEventListener('change', () => { pos = 0; drawGrid(); }));
    draw();
  },
};
