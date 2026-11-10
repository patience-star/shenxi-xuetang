/* 互动组件：张量形状实验室
 * ① 变形：对一个小张量连续做 reshape / 转置 / unsqueeze / squeeze / sum / flatten，
 *    每个格子的颜色代表元素的值，能直接看出「reshape 不改顺序、转置会改顺序、sum 会让一维消失」。
 * ② 广播：任选两个形状，按「从右往左对齐」的规则判断能否相加，并画出每个结果格子来自哪里。 */
Widgets['tensor-shape'] = {
  title: '张量形状实验室',
  render(el) {
    const numel = (s) => s.reduce((a, b) => a * b, 1);
    const strides = (s) => { const st = s.map(() => 1); for (let i = s.length - 2; i >= 0; i--) st[i] = st[i + 1] * s[i + 1]; return st; };
    const shp = (s) => `[${s.join(', ')}]`;
    const unravel = (k, s) => { const st = strides(s); return s.map((_, d) => Math.floor(k / st[d]) % s[d]); };

    function permute(t, p) {
      const ns = p.map((i) => t.shape[i]), st = strides(t.shape);
      const data = Array.from({ length: t.data.length }, (_, k) => { const idx = unravel(k, ns); return t.data[idx.reduce((o, v, d) => o + v * st[p[d]], 0)]; });
      return { shape: ns, data };
    }
    function sumDim(t, dim) {
      const ns = t.shape.filter((_, i) => i !== dim), ost = strides(ns), data = new Array(numel(ns)).fill(0);
      t.data.forEach((v, k) => { const idx = unravel(k, t.shape).filter((_, i) => i !== dim); data[idx.reduce((o, x, d) => o + x * ost[d], 0)] += v; });
      return { shape: ns, data };
    }
    function grid(t, prefix, max) {
      const cell = (v) => { const c = Charts.seq(0.12 + 0.88 * (Math.abs(v) / max)); return `<span class="ts-cell" style="background:${c.bg};color:${c.ink}">${v}</span>`; };
      const s = t.shape;
      if (s.length === 0) return `<div class="ts-grid" style="grid-template-columns:auto">${cell(t.data[0])}</div>`;
      if (s.length === 1) return `<div class="ts-grid" style="grid-template-columns:repeat(${s[0]},auto)">${t.data.map(cell).join('')}</div>`;
      if (s.length === 2) return `<div class="ts-grid" style="grid-template-columns:repeat(${s[1]},auto)">${t.data.map(cell).join('')}</div>`;
      const sub = numel(s.slice(1));
      return `<div class="ts-nest">${Array.from({ length: s[0] }, (_, i) => {
        const p = prefix ? `${prefix.slice(0, -1)}, ${i}]` : `x[${i}]`;
        return `<div class="ts-block"><div class="small muted">${p}</div>${grid({ shape: s.slice(1), data: t.data.slice(i * sub, (i + 1) * sub) }, p, max)}</div>`;
      }).join('')}</div>`;
    }

    el.innerHTML = `
      <div class="seg ts-tabs" role="tablist"><button type="button" class="on" data-t="r">① 变形</button><button type="button" data-t="b">② 广播</button></div>
      <div class="ts-r">
        <div class="controls"><label>起点<select class="ts-start">
          <option value="6">torch.arange(6)</option>
          <option value="3,4" selected>torch.arange(12).reshape(3, 4)</option>
          <option value="2,3,4">torch.arange(24).reshape(2, 3, 4)</option>
        </select></label></div>
        <div class="ts-ops row"></div>
        <pre class="code ts-code"></pre>
        <p class="verdict ts-why"></p>
        <div class="ts-view"></div>
      </div>
      <div class="ts-b" hidden>
        <div class="row ts-presets"></div>
        <div class="controls">
          <label>A 的形状<input type="text" class="ts-a" value="3, 1"></label>
          <label>B 的形状<input type="text" class="ts-b2" value="1, 4"></label>
        </div>
        <div class="ts-align"></div>
        <p class="verdict ts-bv"></p>
        <div class="ts-bview"></div>
      </div>`;
    el.querySelector('.ts-tabs').addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      el.querySelectorAll('.ts-tabs button').forEach((x) => x.classList.toggle('on', x === b));
      el.querySelector('.ts-r').hidden = b.dataset.t !== 'r';
      el.querySelector('.ts-b').hidden = b.dataset.t !== 'b';
    });

    /* ---------- ① 变形 ---------- */
    let hist = [];   // [{ t, code, why }]
    const start = () => {
      const s = el.querySelector('.ts-start').value.split(',').map(Number);
      const n = numel(s);
      hist = [{ t: { shape: s, data: Array.from({ length: n }, (_, i) => i) }, code: el.querySelector('.ts-start').selectedOptions[0].textContent, why: '每个格子写的是元素的值，颜色越深值越大。注意元素按「行优先」顺序排列：先排满最后一维，再进到前一维。' }];
      show();
    };
    const factor2 = (n) => { const r = []; for (let a = 2; a < n; a++) if (n % a === 0) r.push([a, n / a]); return r; };
    function ops(t) {
      const n = t.data.length, s = t.shape, list = [];
      const shapes = [[n], ...factor2(n)];
      if (n >= 8) for (const [a, b] of factor2(n)) for (const [c, d] of factor2(b)) if (shapes.length < 9) shapes.push([a, c, d]);
      shapes.filter((x) => shp(x) !== shp(s)).slice(0, 7).forEach((x) => list.push([`reshape(${x.join(', ')})`, () => ({ t: { shape: x, data: t.data }, why: `reshape 只改「怎么分组」，元素的排列顺序不变（颜色从浅到深的顺序还和原来一样）。${numel(x)} 个元素，形状从 ${shp(s)} 变成 ${shp(x)}。` })]));
      if (s.length === 2) list.push(['.T', () => ({ t: permute(t, [1, 0]), why: '转置把行和列互换：第 i 行第 j 列的元素跑到了第 j 行第 i 列。按行读出来的顺序变了——这就是它和 reshape 的本质区别。' })]);
      if (s.length === 3) {
        list.push(['permute(0, 2, 1)', () => ({ t: permute(t, [0, 2, 1]), why: 'permute 按给定顺序重新排列维度：这里把后两维互换，相当于对每一块分别转置。' })]);
        list.push(['permute(2, 0, 1)', () => ({ t: permute(t, [2, 0, 1]), why: '把原来的第 2 维挪到最前面——图像从 HWC 变成 CHW 就是这种操作。' })]);
      }
      if (s.length < 4) list.push(['unsqueeze(0)', () => ({ t: { shape: [1, ...s], data: t.data }, why: '在最前面插入一个长度为 1 的维。元素一个没变，只是多套了一层括号——给单张图片加「批量维」就靠它。' })]);
      if (s.includes(1)) list.push(['squeeze()', () => ({ t: { shape: s.filter((d) => d !== 1), data: t.data }, why: '去掉所有长度为 1 的维，元素不变。' })]);
      if (s.length > 1) list.push(['flatten()', () => ({ t: { shape: [n], data: t.data }, why: '展平成一维（按行优先顺序），全连接层的输入常常需要这样。' })]);
      s.forEach((d, k) => { if (n > 1) list.push([`sum(dim=${k})`, () => ({ t: sumDim(t, k), why: `沿第 ${k} 维求和：把这一维上的 ${d} 个数加成 1 个数，于是这一维消失了，形状 ${shp(s)} → ${shp(s.filter((_, i) => i !== k))}。` })]); });
      return list;
    }
    function show() {
      const cur = hist[hist.length - 1].t;
      el.querySelector('.ts-code').textContent = hist.map((h, i) => (i ? `x = x${h.code.startsWith('.') ? '' : '.'}${h.code}` : `x = ${h.code}`).padEnd(42) + `# 形状 ${shp(h.t.shape)}`).join('\n');
      el.querySelector('.ts-why').textContent = hist[hist.length - 1].why;
      const max = Math.max(1, ...cur.data.map(Math.abs));
      el.querySelector('.ts-view').innerHTML = `<p class="small muted">x 的形状：<b>${shp(cur.shape)}</b>，共 ${cur.data.length} 个元素</p>${grid(cur, '', max)}`;
      const box = el.querySelector('.ts-ops');
      box.innerHTML = '';
      for (const [label, fn] of ops(cur)) {
        const b = Object.assign(document.createElement('button'), { type: 'button', className: 'btn sm ghost', textContent: label });
        b.addEventListener('click', () => { const r = fn(); hist.push({ ...r, code: label }); show(); });
        box.appendChild(b);
      }
      if (hist.length > 1) {
        const u = Object.assign(document.createElement('button'), { type: 'button', className: 'btn sm', textContent: '↶ 撤销' });
        u.addEventListener('click', () => { hist.pop(); show(); });
        box.appendChild(u);
      }
    }
    el.querySelector('.ts-start').addEventListener('change', start);
    start();

    /* ---------- ② 广播 ---------- */
    const PRESETS = [['3, 1', '1, 4', '外加：列 + 行'], ['2, 3', '3', '每行加偏置'], ['4', '4, 1', '常见的坑'], ['2, 3', '2', '会报错'], ['64, 10', '10', '一批得分 + 偏置'], ['64, 1, 28, 28', '1', '标准化']];
    el.querySelector('.ts-presets').innerHTML = PRESETS.map(([a, b, t], i) => `<button type="button" class="btn sm ghost" data-p="${i}">[${a}] + [${b}]：${t}</button>`).join('');
    el.querySelector('.ts-presets').addEventListener('click', (e) => {
      const b = e.target.closest('[data-p]');
      if (!b) return;
      const [a, c] = PRESETS[+b.dataset.p];
      el.querySelector('.ts-a').value = a;
      el.querySelector('.ts-b2').value = c;
      bcast();
    });
    const parse = (s) => s.split(/[,，\s]+/).filter(Boolean).map(Number);
    function bcast() {
      const A = parse(el.querySelector('.ts-a').value), B = parse(el.querySelector('.ts-b2').value);
      const bv = el.querySelector('.ts-bv'), view = el.querySelector('.ts-bview');
      if (!A.length || !B.length || [...A, ...B].some((d) => !Number.isInteger(d) || d < 1 || d > 4096)) { bv.textContent = '请输入用逗号分隔的正整数，例如 3, 1'; view.innerHTML = ''; return; }
      const nd = Math.max(A.length, B.length);
      const pa = [...Array(nd - A.length).fill(null), ...A], pb = [...Array(nd - B.length).fill(null), ...B];
      let bad = -1;
      const R = pa.map((a, i) => { const x = a ?? 1, y = pb[i] ?? 1; if (x !== y && x !== 1 && y !== 1) { bad = i; return null; } return Math.max(x, y); });
      const note = (a, r) => (a === null ? '<span class="tag">补 1</span>' : a === 1 && r > 1 ? '<span class="tag warn">扩展</span>' : '');
      el.querySelector('.ts-align').innerHTML = `<div class="table-wrap"><table class="data"><thead><tr><th></th>${pa.map((_, i) => `<th>第 ${i - nd} 维</th>`).join('')}</tr></thead><tbody>
        <tr><td>A ${shp(A)}</td>${pa.map((a, i) => `<td>${a ?? '–'} ${R[i] !== null ? note(a, R[i]) : ''}</td>`).join('')}</tr>
        <tr><td>B ${shp(B)}</td>${pb.map((b, i) => `<td>${b ?? '–'} ${R[i] !== null ? note(b, R[i]) : ''}</td>`).join('')}</tr>
        <tr><td><b>结果</b></td>${R.map((r, i) => `<td>${r === null ? '<b style="color:var(--err)">✗</b>' : `<b>${r}</b>`}</td>`).join('')}</tr></tbody></table></div>
        <p class="small muted">规则：把两个形状<b>靠右对齐</b>，逐维比较——相等就保留；有一个是 1，就把它「复制」成另一个的长度；缺的维当作 1；两个数不相等且都不是 1 就报错。</p>`;
      if (bad >= 0) {
        bv.className = 'verdict warn ts-bv';
        bv.textContent = `✗ 无法广播：第 ${bad - nd} 维上 ${pa[bad]} 和 ${pb[bad]} 既不相等，也没有一个是 1。PyTorch 会报错 “The size of tensor a must match the size of tensor b”。`;
        view.innerHTML = '';
        return;
      }
      bv.className = 'verdict ok ts-bv';
      bv.textContent = `✓ 可以运算，结果形状 ${shp(R)}（${numel(R).toLocaleString()} 个元素）${shp(R) !== shp(A) && shp(R) !== shp(B) ? '——比 A、B 都大，注意这是不是你想要的！' : ''}`;
      // 画图：两者都不超过 2 维且结果不超过 48 个元素时，画出每个格子的来源
      if (nd > 2 || numel(R) > 48) { view.innerHTML = '<p class="small muted">形状较大，这里只演示形状规则；不超过 2 维、48 个元素时会画出每个格子。</p>'; return; }
      const r2 = nd === 1 ? [1, R[0]] : R, a2 = nd === 1 ? [1, pa[0] ?? 1] : pa.map((x) => x ?? 1), b2 = nd === 1 ? [1, pb[0] ?? 1] : pb.map((x) => x ?? 1);
      const valA = (i, j) => i * a2[1] + j + 1, valB = (i, j) => (i * b2[1] + j + 1) * 10;
      const mat = (rows, cols, f, cls) => `<div class="ts-grid" style="grid-template-columns:repeat(${cols},auto)">${Array.from({ length: rows * cols }, (_, k) => { const i = Math.floor(k / cols), j = k % cols; return `<span class="ts-cell ${cls}" data-i="${i}" data-j="${j}">${f(i, j)}</span>`; }).join('')}</div>`;
      view.innerHTML = `<div class="row" style="align-items:flex-start;gap:18px">
        <div><div class="small muted">A</div>${mat(a2[0], a2[1], valA, 'ta')}</div><div style="padding-top:22px">+</div>
        <div><div class="small muted">B（值乘了 10 方便区分）</div>${mat(b2[0], b2[1], valB, 'tb')}</div><div style="padding-top:22px">=</div>
        <div><div class="small muted">结果（指针移到格子上看它由哪两个数相加）</div>${mat(r2[0], r2[1], (i, j) => valA(a2[0] === 1 ? 0 : i, a2[1] === 1 ? 0 : j) + valB(b2[0] === 1 ? 0 : i, b2[1] === 1 ? 0 : j), 'tr')}</div></div>`;
      view.querySelectorAll('.tr').forEach((c) => {
        const on = (flag) => {
          const i = +c.dataset.i, j = +c.dataset.j;
          c.classList.toggle('hl', flag);
          view.querySelector(`.ta[data-i="${a2[0] === 1 ? 0 : i}"][data-j="${a2[1] === 1 ? 0 : j}"]`)?.classList.toggle('hl', flag);
          view.querySelector(`.tb[data-i="${b2[0] === 1 ? 0 : i}"][data-j="${b2[1] === 1 ? 0 : j}"]`)?.classList.toggle('hl', flag);
        };
        c.addEventListener('pointerenter', () => on(true));
        c.addEventListener('pointerleave', () => on(false));
        c.addEventListener('pointerdown', () => { view.querySelectorAll('.hl').forEach((x) => x.classList.remove('hl')); on(true); });
      });
    }
    el.querySelector('.ts-a').addEventListener('input', bcast);
    el.querySelector('.ts-b2').addEventListener('input', bcast);
    bcast();
  },
};
