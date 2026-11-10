/* 互动组件：计算图与反向传播
 * 一个神经元 + 平方损失：L = (σ(w·x + b) − y)²，画成计算图。
 * 「前向一步」从左往右算出每个节点的值（蓝色）；「反向一步」从损失出发，
 * 把梯度（橙色）乘上每个节点的局部导数往回传。最后可以用梯度更新一次参数，看损失变小。 */
Widgets['comp-graph'] = {
  title: '计算图：前向算值，反向传梯度',
  render(el) {
    const sig = (z) => 1 / (1 + Math.exp(-z));
    const P = { x: 1.5, w: 0.8, b: -0.5, y: 1 };
    let fwd = 0, bwd = 0, lastLoss = null;
    const N = {   // 节点位置
      x: [60, 60], w: [60, 160], mul: [200, 110], b: [200, 225], add: [330, 160], sig: [460, 160], y: [460, 262], sub: [590, 200], sq: [715, 200],
    };
    const E = [['x', 'mul'], ['w', 'mul'], ['mul', 'add'], ['b', 'add'], ['add', 'sig'], ['sig', 'sub'], ['y', 'sub'], ['sub', 'sq']];
    const OPS = { mul: '×', add: '+', sig: 'σ', sub: '−', sq: '²' };
    const NAME = { mul: 'u', add: 'z', sig: 'a', sub: 'e', sq: 'L' };

    function compute() {
      const u = P.w * P.x, z = u + P.b, a = sig(z), e = a - P.y, L = e * e;
      const dL = 1, de = 2 * e, da = de, dz = da * a * (1 - a), du = dz, db = dz, dw = du * P.x, dx = du * P.w;
      return { u, z, a, e, L, g: { sq: dL, sub: de, sig: da, add: dz, mul: du, b: db, w: dw, x: dx } };
    }
    // 第 k 步前向 / 反向解锁了哪些节点
    const FWD = ['mul', 'add', 'sig', 'sub', 'sq'];
    const BWD = [['sq'], ['sub'], ['sig'], ['add'], ['mul', 'b'], ['w', 'x']];
    const LOCAL = { 'sub→sq': '×2e', 'sig→sub': '×1', 'add→sig': '×a(1−a)', 'mul→add': '×1', 'b→add': '×1', 'w→mul': '×x', 'x→mul': '×w' };

    el.innerHTML = `
      <div class="controls">
        ${['x', 'w', 'b'].map((k) => `<label>${k} = <output class="cg-o${k}"></output><input type="range" class="cg-${k}" min="-2" max="2" step="0.1" value="${P[k]}"></label>`).join('')}
        <label>真实标签 y<select class="cg-y"><option value="1" selected>1</option><option value="0">0</option></select></label>
      </div>
      <div class="row"><button type="button" class="btn sm" data-a="f">前向一步 →</button><button type="button" class="btn sm" data-a="b">← 反向一步</button>
        <button type="button" class="btn sm ghost" data-a="all">一键算完</button><button type="button" class="btn sm ghost" data-a="upd">用梯度更新 w、b（η = 1）</button><button type="button" class="btn sm ghost" data-a="reset">重置</button></div>
      <div class="diagram" style="margin-top:8px"><svg class="cg-svg" viewBox="0 0 780 300" role="img" aria-label="计算图"></svg></div>
      <div class="verdict cg-v"></div>`;
    const $ = (s) => el.querySelector(s);
    const f3 = (v) => (Math.abs(v) < 5e-5 ? '0.0000' : v.toFixed(4));

    function draw() {
      const r = compute(), s = [];
      const known = new Set(['x', 'w', 'b', 'y', ...FWD.slice(0, fwd)]);
      const graded = new Set(BWD.slice(0, bwd).flat());
      const val = { x: P.x, w: P.w, b: P.b, y: P.y, mul: r.u, add: r.z, sig: r.a, sub: r.e, sq: r.L };
      for (const [a, b] of E) {
        const [x1, y1] = N[a], [x2, y2] = N[b], back = graded.has(a) && a !== 'y';
        const ang = Math.atan2(y2 - y1, x2 - x1), sx = x1 + 25 * Math.cos(ang), sy = y1 + 25 * Math.sin(ang), ex = x2 - 27 * Math.cos(ang), ey = y2 - 27 * Math.sin(ang);
        s.push(`<line x1="${sx}" y1="${sy}" x2="${ex}" y2="${ey}" class="arrow ${back ? 'alt' : known.has(b) ? 'hl' : ''}"/>`);
        const hd = (t) => `${ex - 8 * Math.cos(ang - t)},${ey - 8 * Math.sin(ang - t)}`;
        s.push(`<polygon points="${ex},${ey} ${hd(0.45)} ${hd(-0.45)}" class="head ${back ? 'alt' : known.has(b) ? 'hl' : ''}"/>`);
        if (back && LOCAL[`${a}→${b}`]) s.push(`<text x="${(sx + ex) / 2}" y="${(sy + ey) / 2 + 16}" text-anchor="middle" class="sub" style="fill:var(--c2)">${LOCAL[`${a}→${b}`]}</text>`);
      }
      for (const [id, [x, y]] of Object.entries(N)) {
        const op = OPS[id], show = known.has(id);
        s.push(`<circle cx="${x}" cy="${y}" r="24" class="${op ? 'node' : 'box'}" style="${graded.has(id) ? 'stroke:var(--c2);stroke-width:2.5' : ''}"/>`);
        s.push(`<text x="${x}" y="${y + 6}" text-anchor="middle" style="font-size:${op ? 20 : 15}px;font-weight:600">${op || id}</text>`);
        const label = op ? `${NAME[id]} = ${show ? f3(val[id]) : '?'}` : `${id} = ${val[id]}`;
        s.push(`<text x="${x}" y="${y - 32}" text-anchor="middle" style="fill:var(--primary);font-size:12.5px;font-weight:600">${label}</text>`);
        if (graded.has(id) && id !== 'y') s.push(`<text x="${x}" y="${y + 42}" text-anchor="middle" style="fill:var(--c2);font-size:12.5px;font-weight:600">∂L/∂${op ? NAME[id] : id} = ${f3(r.g[id])}</text>`);
      }
      $('.cg-svg').innerHTML = s.join('');
      // 解说
      const v = $('.cg-v');
      const tips = {
        f: ['', `<b>前向 1</b>：乘法节点 u = w·x = ${P.w} × ${P.x} = ${f3(r.u)}`, `<b>前向 2</b>：加法节点 z = u + b = ${f3(r.u)} + (${P.b}) = ${f3(r.z)}`,
          `<b>前向 3</b>：sigmoid 节点 a = σ(z) = 1 / (1 + e^−z) = ${f3(r.a)}，把任意数压到 0~1 之间`, `<b>前向 4</b>：误差 e = a − y = ${f3(r.a)} − ${P.y} = ${f3(r.e)}`,
          `<b>前向 5</b>：损失 L = e² = ${f3(r.L)}。前向结束：每个中间结果都记下来了，反向传播要用。`],
        b: ['', '<b>反向 1</b>：起点 ∂L/∂L = 1（损失对自己的导数）', `<b>反向 2</b>：L = e²，局部导数 2e → ∂L/∂e = 1 × 2 × (${f3(r.e)}) = ${f3(r.g.sub)}`,
          `<b>反向 3</b>：e = a − y，对 a 的局部导数是 1 → ∂L/∂a = ${f3(r.g.sig)}`,
          `<b>反向 4</b>：a = σ(z)，局部导数 σ′(z) = a(1 − a) = ${f3(r.a * (1 - r.a))} → ∂L/∂z = ${f3(r.g.sig)} × ${f3(r.a * (1 - r.a))} = ${f3(r.g.add)}`,
          `<b>反向 5</b>：加法节点把梯度<b>原样复制</b>给两个输入 → ∂L/∂u = ∂L/∂b = ${f3(r.g.add)}`,
          `<b>反向 6</b>：乘法节点把梯度乘上<b>另一个输入</b> → ∂L/∂w = ${f3(r.g.mul)} × x(${P.x}) = <b>${f3(r.g.w)}</b>；∂L/∂x = ${f3(r.g.mul)} × w(${P.w}) = ${f3(r.g.x)}。全部梯度到手！`],
      };
      let html = bwd ? tips.b[bwd] : fwd ? tips.f[fwd] : '点「前向一步」开始：蓝色是前向算出的值。前向必须一步步来——后面的节点要用前面的结果。';
      if (bwd === 6) html += `<br>梯度为${r.g.w < 0 ? '负' : '正'}：把 w ${r.g.w < 0 ? '调大' : '调小'}能让损失变小。点「用梯度更新」试试。`;
      if (lastLoss !== null && fwd === 5) html += `<br><b>更新后</b>：损失从 ${f3(lastLoss)} 变成了 ${f3(r.L)} ${r.L < lastLoss ? '↓ 变小了！' : ''}——这就是一次完整的训练迭代：前向 → 反向 → 更新。`;
      v.innerHTML = html;
      ['x', 'w', 'b'].forEach((k) => { $(`.cg-o${k}`).textContent = P[k]; $(`.cg-${k}`).value = P[k]; });
    }
    el.addEventListener('click', (e) => {
      const a = e.target.closest('[data-a]')?.dataset.a;
      if (!a) return;
      if (a === 'f') { if (fwd < 5) fwd++; else if (bwd < 6) bwd++; lastLoss = null; }
      if (a === 'b') { if (fwd < 5) fwd = 5; else if (bwd < 6) bwd++; lastLoss = null; }
      if (a === 'all') { fwd = 5; bwd = 6; lastLoss = null; }
      if (a === 'reset') { fwd = 0; bwd = 0; lastLoss = null; Object.assign(P, { x: 1.5, w: 0.8, b: -0.5 }); P.y = +$('.cg-y').value; }
      if (a === 'upd') {
        const r = compute();
        lastLoss = r.L;
        P.w = +(P.w - r.g.w).toFixed(4);
        P.b = +(P.b - r.g.b).toFixed(4);
        fwd = 5; bwd = 6;
      }
      draw();
    });
    ['x', 'w', 'b'].forEach((k) => $(`.cg-${k}`).addEventListener('input', (e) => { P[k] = +e.target.value; lastLoss = null; draw(); }));
    $('.cg-y').addEventListener('change', (e) => { P.y = +e.target.value; lastLoss = null; draw(); });
    draw();
  },
};
