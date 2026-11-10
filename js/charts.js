/* =====================================================================
 * 小型 SVG 图表库（供 js/widgets/ 使用）：柱状图、条形图（支持正负）、折线图、
 * 顺序色阶、数字格式化。颜色全部取自 style.css 的 CSS 变量，深浅色模式自动适配。
 * 每个图表自带悬停提示（键盘聚焦同样可见）和「查看数据表」。
 * ===================================================================== */
(function () {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';

  const fmt = {
    int: (n) => Math.round(n).toLocaleString('zh-CN'),
    money: (n) => `¥${Math.round(n).toLocaleString('zh-CN')}`,
    pct: (x, d = 1) => `${(x * 100).toFixed(d).replace('-', '−')}%`,
    num: (n, d = 2) => Number(n).toLocaleString('zh-CN', { maximumFractionDigits: d }),
    wan: (n) => (Math.abs(n) >= 1e4 ? `${(n / 1e4).toFixed(1).replace(/\.0$/, '')}万` : Math.round(n).toLocaleString('zh-CN')),
    signed: (n, f = fmt.int) => (n > 0 ? '+' : n < 0 ? '−' : '') + f(Math.abs(n)),
  };

  /** 覆盖 [lo, hi] 的「整齐」刻度 */
  function niceTicks(lo, hi, n = 4) {
    if (hi === lo) hi = lo + 1;
    const raw = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map((s) => s * mag).find((s) => s >= raw);
    const out = [];
    for (let v = Math.floor(lo / step) * step; v <= Math.ceil(hi / step) * step + step / 2; v += step) out.push(+v.toFixed(10));
    return out;
  }

  function el(tag, attrs, parent) {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  }
  function text(parent, x, y, str, attrs = {}) {
    const t = el('text', { x, y, ...attrs }, parent);
    t.textContent = str;
    return t;
  }
  function svgFor(container, W, H, label) {
    container.classList.add('chart');
    container.innerHTML = '';
    return el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': label || '图表' }, container);
  }
  const width = (c) => Math.max(300, Math.round(c.clientWidth || 640));

  /** 悬停 / 聚焦提示：rowsFn() 返回 [[值, 标签], ...]，每行「值」加粗在前、标签在后 */
  function tip(container, mark, rowsFn) {
    let box = container.querySelector(':scope > .tooltip');
    if (!box) { box = document.createElement('div'); box.className = 'tooltip'; box.hidden = true; container.appendChild(box); }
    const show = (x, y) => {
      box.replaceChildren();
      for (const [v, label] of rowsFn()) {
        const row = document.createElement('div');
        const b = document.createElement('b');
        b.textContent = v;
        row.appendChild(b);
        if (label) row.appendChild(Object.assign(document.createElement('span'), { className: 'muted', textContent: ` ${label}` }));
        box.appendChild(row);
      }
      box.hidden = false;
      const r = container.getBoundingClientRect();
      const left = Math.min(x - r.left + 14, r.width - box.offsetWidth - 4);
      box.style.left = `${Math.max(0, left)}px`;
      box.style.top = `${Math.max(0, y - r.top - box.offsetHeight - 10)}px`;
    };
    mark.setAttribute('tabindex', '0');
    mark.classList.add('mark');
    mark.addEventListener('pointermove', (e) => show(e.clientX, e.clientY));
    mark.addEventListener('focus', () => { const b = mark.getBoundingClientRect(); show(b.left + b.width / 2, b.top); });
    const hide = () => { box.hidden = true; };
    mark.addEventListener('pointerleave', hide);
    mark.addEventListener('blur', hide);
  }

  function dataTable(container, columns, rows) {
    const d = document.createElement('details');
    d.innerHTML = '<summary>查看数据表</summary><div class="table-wrap"><table class="data"><thead><tr></tr></thead><tbody></tbody></table></div>';
    const tr = d.querySelector('thead tr');
    columns.forEach((c) => tr.appendChild(Object.assign(document.createElement('th'), { textContent: c })));
    const tb = d.querySelector('tbody');
    rows.forEach((r) => {
      const row = document.createElement('tr');
      r.forEach((v) => row.appendChild(Object.assign(document.createElement('td'), { textContent: v })));
      tb.appendChild(row);
    });
    container.appendChild(d);
  }

  // 数据端 4px 圆角、基线端直角的柱子路径
  function vPath(x, yEnd, w, yBase, r = 4) {
    r = Math.min(r, w / 2, Math.abs(yBase - yEnd));
    const s = yEnd < yBase ? 1 : -1;
    return `M${x},${yBase}V${yEnd + s * r}Q${x},${yEnd} ${x + r},${yEnd}H${x + w - r}Q${x + w},${yEnd} ${x + w},${yEnd + s * r}V${yBase}Z`;
  }
  function hPath(xBase, xEnd, y, h, r = 4) {
    r = Math.min(r, h / 2, Math.abs(xEnd - xBase));
    const s = xEnd >= xBase ? 1 : -1;
    return `M${xBase},${y}H${xEnd - s * r}Q${xEnd},${y} ${xEnd},${y + r}V${y + h - r}Q${xEnd},${y + h} ${xEnd - s * r},${y + h}H${xBase}Z`;
  }

  /**
   * 柱状图（纵向）。data = [{ label, value, color? }]
   * opts: { format, height, title, yLabel, yMin（纵轴起点，默认含 0）, color(d) → CSS 颜色, table }
   */
  function columns(container, data, opts = {}) {
    const W = width(container), H = opts.height || 260, m = { t: 22, r: 10, b: 40, l: 54 };
    const f = opts.format || fmt.int;
    const svg = svgFor(container, W, H, opts.title);
    const vals = data.map((d) => d.value);
    const tk = niceTicks(opts.yMin ?? Math.min(0, ...vals), Math.max(0, ...vals));
    const y0 = tk[0], y1 = tk[tk.length - 1];
    const base = Math.max(y0, 0); // 柱子从 0 长出；纵轴被截断（yMin > 0）时从轴底长出
    const y = (v) => m.t + (H - m.t - m.b) * (1 - (v - y0) / (y1 - y0));
    tk.forEach((v) => {
      el('line', { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: v === base ? 'baseline' : 'gridline' }, svg);
      text(svg, m.l - 8, y(v) + 4, f(v), { 'text-anchor': 'end' });
    });
    const band = (W - m.l - m.r) / data.length, bw = Math.min(24, band * 0.6);
    data.forEach((d, i) => {
      const x = m.l + band * i + (band - bw) / 2;
      const p = el('path', { d: vPath(x, y(d.value), bw, y(base)), fill: d.color || opts.color?.(d) || 'var(--c1)' }, svg);
      tip(container, p, () => [[f(d.value), d.label], ...(d.extra || [])]);
      text(svg, x + bw / 2, H - m.b + 18, d.label, { 'text-anchor': 'middle' });
      if (data.length <= 12) text(svg, x + bw / 2, d.value >= 0 ? y(d.value) - 6 : y(d.value) + 14, f(d.value), { 'text-anchor': 'middle', class: 'val' });
    });
    if (opts.yLabel) text(svg, 4, 12, opts.yLabel);
    if (opts.table !== false) dataTable(container, ['类别', opts.yLabel || '数值'], data.map((d) => [d.label, f(d.value)]));
  }

  /**
   * 条形图（横向，支持负值 → 正负两色）。data = [{ label, value, color?, note? }]
   * opts: { format, title, color(d), labelWidth, table }
   */
  function hbars(container, data, opts = {}) {
    const W = width(container), m = { t: 8, b: 8 };
    const f = opts.format || fmt.int;
    const svg = svgFor(container, W, 100, opts.title);
    const lo = Math.min(0, ...data.map((d) => d.value)), hi = Math.max(0, ...data.map((d) => d.value));
    const notes = data.map((d) => d.note ?? f(d.value));
    const measure = (str) => {
      const t = text(svg, 0, -100, str, { class: 'val' });
      const w = t.getComputedTextLength() || String(str).length * 8;
      t.remove();
      return w;
    };
    const nw = notes.map(measure);
    const room = (neg) => Math.max(0, ...nw.filter((_, i) => (data[i].value < 0) === neg).map((w) => w + 10));
    // 默认数值标在条形末端，两侧按最长的标注留出空间；
    // 容器太窄（手机）时改成两行：类别名下面写数值，条形占满剩余宽度
    let labelW = opts.labelWidth || 96;
    let left = labelW + room(true), right = W - Math.max(16, room(false));
    const compact = right - left < (W - labelW) * 0.45;
    if (compact) { labelW = Math.max(labelW, ...nw.map((w) => w + 14)); left = labelW; right = W - 12; }
    const rowH = compact ? 42 : 30;
    const H = m.t + m.b + rowH * data.length;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const x = (v) => left + ((v - lo) / ((hi - lo) || 1)) * (right - left);
    el('line', { x1: x(0), x2: x(0), y1: m.t - 4, y2: H - m.b + 4, class: 'baseline' }, svg);
    data.forEach((d, i) => {
      const h = compact ? 14 : 18, yy = m.t + i * rowH + (rowH - h) / 2;
      const color = d.color || opts.color?.(d) || (d.value < 0 ? 'var(--neg)' : 'var(--c1)');
      const p = el('path', { d: hPath(x(0), x(d.value), yy, h), fill: color }, svg);
      tip(container, p, () => [[f(d.value), d.label], ...(d.extra || [])]);
      if (compact) {
        text(svg, labelW - 8, yy + h / 2 - 3, d.label, { 'text-anchor': 'end', class: 'val' });
        text(svg, labelW - 8, yy + h / 2 + 13, notes[i], { 'text-anchor': 'end' });
      } else {
        text(svg, labelW - 8, yy + h / 2 + 4, d.label, { 'text-anchor': 'end', class: 'val' });
        const vx = d.value >= 0 ? x(d.value) + 6 : x(d.value) - 6;
        text(svg, vx, yy + h / 2 + 4, notes[i], { 'text-anchor': d.value >= 0 ? 'start' : 'end', class: 'val' });
      }
    });
    if (opts.table !== false) dataTable(container, ['类别', '数值'], data.map((d) => [d.label, d.note ?? f(d.value)]));
  }

  /**
   * 折线图。chart = { labels: [...x 轴], series: [{ name, values, color? }] }
   * opts: { format, height, title, yMin, everyNthLabel, table }
   * 指针悬停时十字线吸附到最近的 x，提示框列出所有系列的值。
   */
  function line(container, chart, opts = {}) {
    const W = width(container), H = opts.height || 260, m = { t: 16, r: 16, b: 34, l: 54 };
    const f = opts.format || fmt.int;
    const svg = svgFor(container, W, H, opts.title);
    const all = chart.series.flatMap((s) => s.values).filter((v) => v != null);
    const tk = niceTicks(opts.yMin ?? Math.min(0, ...all), Math.max(...all));
    const y0 = tk[0], y1 = tk[tk.length - 1], n = chart.labels.length;
    const x = (i) => m.l + (n === 1 ? 0 : (i / (n - 1)) * (W - m.l - m.r));
    const y = (v) => m.t + (H - m.t - m.b) * (1 - (v - y0) / (y1 - y0));
    tk.forEach((v) => {
      el('line', { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: v === y0 ? 'baseline' : 'gridline' }, svg);
      text(svg, m.l - 8, y(v) + 4, f(v), { 'text-anchor': 'end' });
    });
    const every = opts.everyNthLabel || Math.ceil(n / Math.max(2, Math.floor((W - m.l - m.r) / 70)));
    chart.labels.forEach((lb, i) => { if (i % every === 0) text(svg, x(i), H - m.b + 18, lb, { 'text-anchor': 'middle' }); });
    chart.series.forEach((s, k) => {
      const color = s.color || `var(--c${k + 1})`;
      const pts = s.values.map((v, i) => (v == null ? null : [x(i), y(v)]));
      const d = pts.reduce((acc, p, i) => (p ? acc + `${acc && pts[i - 1] ? 'L' : 'M'}${p[0]},${p[1]}` : acc), '');
      el('path', { d, class: 'line', stroke: color }, svg);
      const lastI = s.values.length - 1 - [...s.values].reverse().findIndex((v) => v != null);
      if (pts[lastI]) el('circle', { cx: pts[lastI][0], cy: pts[lastI][1], r: 4, class: 'dot', fill: color }, svg);
    });
    // 十字线 + 提示
    const cross = el('line', { y1: m.t, y2: H - m.b, class: 'crosshair', visibility: 'hidden' }, svg);
    const hit = el('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent' }, svg);
    let idx = 0;
    tip(container, hit, () => [[chart.labels[idx], ''], ...chart.series.map((s) => [s.values[idx] == null ? '—' : f(s.values[idx]), s.name])]);
    hit.addEventListener('pointermove', (e) => {
      const r = svg.getBoundingClientRect();
      const px = ((e.clientX - r.left) / r.width) * W;
      idx = Math.max(0, Math.min(n - 1, Math.round(((px - m.l) / (W - m.l - m.r)) * (n - 1))));
      cross.setAttribute('x1', x(idx)); cross.setAttribute('x2', x(idx)); cross.setAttribute('visibility', 'visible');
    });
    hit.addEventListener('pointerleave', () => cross.setAttribute('visibility', 'hidden'));
    if (chart.series.length > 1) {
      const lg = document.createElement('div');
      lg.className = 'legend';
      chart.series.forEach((s, k) => {
        const item = document.createElement('span');
        const key = document.createElement('i');
        key.className = 'line';
        key.style.background = s.color || `var(--c${k + 1})`;
        item.append(key, s.name);
        lg.appendChild(item);
      });
      container.prepend(lg);
    }
    if (opts.table !== false) dataTable(container, ['', ...chart.series.map((s) => s.name)], chart.labels.map((lb, i) => [lb, ...chart.series.map((s) => (s.values[i] == null ? '—' : f(s.values[i])))]));
  }

  /** 顺序色阶：t ∈ [0,1] → { bg, ink }（深色模式下色阶自动反转，见 style.css 的 --seq-*） */
  function seq(t) {
    const cs = getComputedStyle(document.documentElement);
    const stops = ['--seq-100', '--seq-250', '--seq-400', '--seq-550', '--seq-700'].map((v) => {
      const h = cs.getPropertyValue(v).trim().replace('#', '');
      return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
    });
    t = Math.max(0, Math.min(1, t)) * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(t)), u = t - i;
    const c = stops[i].map((a, k) => Math.round(a + (stops[i + 1][k] - a) * u));
    const lum = (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255;
    return { bg: `rgb(${c.join(',')})`, ink: lum > 0.55 ? '#0b0b0b' : '#ffffff' };
  }

  /** 统计小工具：标准正态分布的累积概率 Φ(z)（Abramowitz–Stegun 7.1.26 近似，误差 < 1.5e-7） */
  function normCdf(z) {
    const t = 1 / (1 + 0.3275911 * Math.abs(z) / Math.SQRT2);
    const erf = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z / 2);
    return z >= 0 ? (1 + erf) / 2 : (1 - erf) / 2;
  }
  /** 双样本比例 z 检验：返回 { pa, pb, z, p }（双侧） */
  function propTest(xa, na, xb, nb) {
    const pa = xa / na, pb = xb / nb, pool = (xa + xb) / (na + nb);
    const se = Math.sqrt(pool * (1 - pool) * (1 / na + 1 / nb));
    const z = se ? (pb - pa) / se : 0;
    return { pa, pb, z, p: 2 * (1 - normCdf(Math.abs(z))) };
  }

  /* ---------------- 以下为深析学堂新增：像素图、热力图、概率条、手写板 ---------------- */

  /** 读取 CSS 颜色变量（#rrggbb）→ [r, g, b]，深浅色模式自动跟随 */
  function rgb(name) {
    const h = getComputedStyle(document.documentElement).getPropertyValue(name).trim().replace('#', '');
    return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) || 0);
  }
  /** 读取 CSS 变量的原始值（画布作图用，如 css('--c1')） */
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  /**
   * 高清画布：内部分辨率 = 逻辑尺寸 × 设备像素比，CSS 宽度 100%、按比例缩放。
   * 返回已缩放的 2D 上下文，之后一律按逻辑坐标（0…w, 0…h）作图。
   */
  function hidpi(canvas, w, h) {
    const r = Math.min(3, window.devicePixelRatio || 1);
    canvas.width = Math.round(w * r);
    canvas.height = Math.round(h * r);
    canvas.style.aspectRatio = `${w} / ${h}`;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(r, 0, 0, r, 0, 0);
    ctx.font = `12px ${css('--font') || 'sans-serif'}`;
    return ctx;
  }
  /** 指针在画布上的逻辑坐标（配合 hidpi 使用） */
  function at(canvas, e, w, h) {
    const r = canvas.getBoundingClientRect();
    return [((e.clientX - r.left) * w) / r.width, ((e.clientY - r.top) * h) / r.height];
  }

  /**
   * 把一维数组画成 w×h 像素的图（canvas 内部就是 w×h，CSS 放大时保持像素格）。
   * opts.mode：'gray' 0 黑 → 1 白（MNIST 原始存法）；'ink' 背景色 → 文字色（像纸上的笔迹）；
   *            'div' 负值偏红、正值偏蓝、0 为背景色（看权重、特征图的正负）；'seq' 主题顺序色阶。
   * opts.min / opts.max 不给时按数据自动取。
   */
  function img(canvas, data, w, h, opts = {}) {
    canvas.width = w;
    canvas.height = h;
    canvas.classList.add('pixelated');
    const ctx = canvas.getContext('2d');
    const im = ctx.createImageData(w, h);
    let lo = opts.min, hi = opts.max;
    if (lo == null || hi == null) {
      let a = Infinity, b = -Infinity;
      for (let i = 0; i < w * h; i++) { if (data[i] < a) a = data[i]; if (data[i] > b) b = data[i]; }
      lo = lo ?? a; hi = hi ?? b;
    }
    const mode = opts.mode || 'gray';
    const bg = rgb('--surface'), ink = rgb('--text'), pos = rgb('--pos'), neg = rgb('--neg');
    const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
    const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
    for (let i = 0; i < w * h; i++) {
      const v = data[i];
      let c;
      if (mode === 'div') c = v >= 0 ? mix(bg, pos, Math.min(1, v / m)) : mix(bg, neg, Math.min(1, -v / m));
      else {
        const t = hi > lo ? Math.max(0, Math.min(1, (v - lo) / (hi - lo))) : 0;
        if (mode === 'ink') c = mix(bg, ink, t);
        else if (mode === 'seq') { const s = seq(t).bg.match(/\d+/g).map(Number); c = s; }
        else c = [255 * t, 255 * t, 255 * t];
      }
      im.data[4 * i] = c[0]; im.data[4 * i + 1] = c[1]; im.data[4 * i + 2] = c[2]; im.data[4 * i + 3] = 255;
    }
    ctx.putImageData(im, 0, 0);
    return canvas;
  }

  /** 彩色小图：data 是 HWC 排列的 0–255 数组（如 CIFAR-10 的 32×32×3） */
  function rgbImg(canvas, data, w, h) {
    canvas.width = w;
    canvas.height = h;
    canvas.classList.add('pixelated');
    const ctx = canvas.getContext('2d');
    const im = ctx.createImageData(w, h);
    for (let i = 0; i < w * h; i++) {
      im.data[4 * i] = data[3 * i]; im.data[4 * i + 1] = data[3 * i + 1]; im.data[4 * i + 2] = data[3 * i + 2]; im.data[4 * i + 3] = 255;
    }
    ctx.putImageData(im, 0, 0);
    return canvas;
  }

  /** 新建一个画了图的 canvas（便于批量生成小图），cls 为 CSS 类名 */
  function canvasOf(data, w, h, opts = {}, cls = 'thumb') {
    const c = document.createElement('canvas');
    c.className = cls;
    if (opts.title) c.title = opts.title;
    return img(c, data, w, h, opts);
  }

  /**
   * 热力图（SVG）：m 是二维数组（行 × 列），颜色用主题顺序色阶。
   * opts: { rowLabels, colLabels, min, max, format, values（在格子里写数值）, cell（格子边长）, rowTitle, colTitle }
   * 悬停 / 聚焦格子显示 行、列、数值。
   */
  function heatmap(container, m, opts = {}) {
    const rows = m.length, cols = m[0]?.length || 0;
    const f = opts.format || ((v) => v.toFixed(2));
    const lw = opts.rowLabels ? (opts.labelW || 28) : 4, th = opts.colLabels ? (opts.labelH || 22) : 4;
    const avail = Math.max(200, (container.clientWidth || 600) - lw - 8);
    const cs = opts.cell || Math.max(12, Math.min(40, Math.floor(avail / Math.max(cols, 1))));
    const W = lw + cols * cs + 4, H = th + rows * cs + 4;
    const svg = svgFor(container, W, H, opts.title || '热力图');
    svg.style.maxWidth = `${W}px`;
    let lo = opts.min, hi = opts.max;
    if (lo == null) lo = Math.min(...m.flat());
    if (hi == null) hi = Math.max(...m.flat());
    m.forEach((row, i) => {
      if (opts.rowLabels) text(svg, lw - 6, th + i * cs + cs / 2 + 4, opts.rowLabels[i], { 'text-anchor': 'end' });
      row.forEach((v, j) => {
        const c = seq(hi > lo ? (v - lo) / (hi - lo) : 0);
        const r = el('rect', { x: lw + j * cs, y: th + i * cs, width: cs - 1, height: cs - 1, rx: 2, fill: c.bg }, svg);
        tip(container, r, () => [[f(v), `${opts.rowTitle || '行'} ${opts.rowLabels ? opts.rowLabels[i] : i + 1} · ${opts.colTitle || '列'} ${opts.colLabels ? opts.colLabels[j] : j + 1}`]]);
        if (opts.values && cs >= 26) text(svg, lw + j * cs + cs / 2, th + i * cs + cs / 2 + 4, f(v), { 'text-anchor': 'middle', fill: c.ink, style: 'fill:' + c.ink + ';font-size:10px' });
      });
    });
    if (opts.colLabels) opts.colLabels.forEach((lb, j) => text(svg, lw + j * cs + cs / 2, th - 7, lb, { 'text-anchor': 'middle' }));
    return svg;
  }

  /**
   * 概率条（HTML）：items = [{ label, p, note? }]，p ∈ [0,1]。opts: { hi: 高亮的下标, max: 只显示前几个 }
   */
  function bars(container, items, opts = {}) {
    const list = opts.max ? items.slice(0, opts.max) : items;
    container.innerHTML = `<div class="pbars">${list.map((d, i) => `
      <div class="pbar${i === opts.hi ? ' hi' : ''}"><span class="lb">${d.label}</span>
        <span class="track"><i style="width:${Math.max(0, Math.min(1, d.p)) * 100}%"></i></span>
        <b>${d.note ?? fmt.pct(d.p, 1)}</b></div>`).join('')}</div>`;
  }

  /**
   * 手写板：鼠标或手指在方框里书写。opts: { onChange(x28), lineWidth }
   * 返回 { clear(), get28(), isEmpty(), canvas }。
   * get28() 按 MNIST 的做法预处理：把笔迹缩放到 20×20 的方框内，再按「质心」放到 28×28 的中心，
   * 得到 784 个 0–1 的数（0 = 背景，1 = 笔迹）——和训练数据长得一样，模型才认得出来。
   */
  function pad(container, opts = {}) {
    const N = 280;
    const cv = document.createElement('canvas');
    cv.width = N;
    cv.height = N;
    cv.className = 'pad';
    cv.setAttribute('aria-label', '手写板：按住鼠标或手指书写');
    container.appendChild(cv);
    const ctx = cv.getContext('2d');
    let drawing = false, last = null, empty = true, queued = false;
    const style = () => {
      ctx.strokeStyle = ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text').trim() || '#000';
      ctx.lineWidth = opts.lineWidth || 20;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
    };
    const pos = (e) => { const r = cv.getBoundingClientRect(); return [((e.clientX - r.left) * N) / r.width, ((e.clientY - r.top) * N) / r.height]; };
    const changed = () => {
      if (queued || !opts.onChange) return;
      queued = true;
      requestAnimationFrame(() => { queued = false; opts.onChange(get28()); });
    };
    cv.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      opts.onStart?.();
      style();
      drawing = true;
      empty = false;
      cv.setPointerCapture(e.pointerId);
      last = pos(e);
      ctx.beginPath(); ctx.arc(last[0], last[1], ctx.lineWidth / 2, 0, Math.PI * 2); ctx.fill();
      changed();
    });
    cv.addEventListener('pointermove', (e) => {
      if (!drawing) return;
      const p = pos(e);
      ctx.beginPath(); ctx.moveTo(last[0], last[1]); ctx.lineTo(p[0], p[1]); ctx.stroke();
      last = p;
      changed();
    });
    const end = () => { if (drawing) { drawing = false; changed(); } };
    cv.addEventListener('pointerup', end);
    cv.addEventListener('pointercancel', end);

    function get28() {
      const out = new Float32Array(784);
      if (empty) return out;
      const a = ctx.getImageData(0, 0, N, N).data;
      let x0 = N, y0 = N, x1 = -1, y1 = -1;
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        if (a[4 * (y * N + x) + 3] > 12) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
      }
      if (x1 < 0) return out;
      // ① 面积平均缩放：笔迹外框的长边缩到 20 像素
      const bw = x1 - x0 + 1, bh = y1 - y0 + 1, S = Math.max(bw, bh) / 20;
      const ox = (28 - bw / S) / 2, oy = (28 - bh / S) / 2;
      const tmp = new Float32Array(784);
      for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
        const v = a[4 * (y * N + x) + 3] / 255;
        if (!v) continue;
        const tx = Math.floor(ox + (x - x0 + 0.5) / S), ty = Math.floor(oy + (y - y0 + 0.5) / S);
        if (tx >= 0 && tx < 28 && ty >= 0 && ty < 28) tmp[ty * 28 + tx] += v / (S * S);
      }
      // ② 按质心平移到中心
      let sum = 0, cx = 0, cy = 0;
      for (let i = 0; i < 784; i++) { const v = Math.min(1, tmp[i]); tmp[i] = v; sum += v; cx += v * ((i % 28) + 0.5); cy += v * (Math.floor(i / 28) + 0.5); }
      const dx = Math.round(14 - cx / sum), dy = Math.round(14 - cy / sum);
      for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
        const sx = x - dx, sy = y - dy;
        if (sx >= 0 && sx < 28 && sy >= 0 && sy < 28) out[y * 28 + x] = tmp[sy * 28 + sx];
      }
      return out;
    }
    return {
      canvas: cv,
      get28,
      isEmpty: () => empty,
      clear() { ctx.clearRect(0, 0, N, N); empty = true; changed(); },
      /** 把一张 28×28 的图（0~1）放大画到板上（用于展示测试集样例；不触发 onChange） */
      show28(x28) {
        ctx.clearRect(0, 0, N, N);
        style();
        for (let i = 0; i < 784; i++) {
          if (x28[i] <= 0) continue;
          ctx.globalAlpha = Math.min(1, x28[i]);
          ctx.fillRect((i % 28) * (N / 28), Math.floor(i / 28) * (N / 28), N / 28 + 0.5, N / 28 + 0.5);
        }
        ctx.globalAlpha = 1;
        empty = false;
      },
    };
  }

  window.Charts = { fmt, niceTicks, columns, hbars, line, seq, tip, dataTable, normCdf, propTest, rgb, css, hidpi, at, img, rgbImg, canvasOf, heatmap, bars, pad };
})();
