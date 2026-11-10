/* 互动组件：Transformer 日期翻译（在浏览器里运行）
 * 输入任意写法的日期，模型逐字写出 YYYY-MM-DD；下面的热力图是注意力权重：
 *   交叉注意力：写每个输出字符时，看向输入的哪些字符；编码器自注意力：输入字符之间互相看。
 * 可以选第几层、第几个头，或者 4 个头取平均。 */
Widgets['date-translate'] = {
  title: '日期翻译：看 Transformer 在「看」哪里',
  async render(el, opts) {
    const name = opts.model || opts.py?.name || '11_transformer_dates';
    const py = opts.py?.name === name ? opts.py : await App.loadPy(name);
    const W = NN.weights(py), M = NN.models[name], itos = py.extras.itos, stoi = new Map(itos.map((c, i) => [c, i]));
    let layer = 1, head = -1, view = 'cross', res = null, src = '';
    el.innerHTML = `
      <div class="controls"><label style="flex:1 1 280px">输入一个日期<input type="text" class="dt-in" value="${App.esc(opts.text || py.extras.examples[1])}" maxlength="30"></label><button type="button" class="btn sm" data-a="go">翻译</button></div>
      <div class="row dt-ex">${py.extras.examples.map((s) => `<button type="button" class="btn sm ghost" data-ex="${App.esc(s)}">${App.esc(s)}</button>`).join('')}</div>
      <div class="dt-out" style="font:600 1.8rem var(--mono);margin:12px 0 4px"></div>
      <div class="dt-steps small muted"></div>
      <div class="controls">
        <span class="seg dt-view"><button type="button" class="on" data-v="cross">交叉注意力（输出 → 输入）</button><button type="button" data-v="enc">编码器自注意力（输入 ↔ 输入）</button></span>
        <span class="seg dt-layer"><button type="button" data-l="0">第 1 层</button><button type="button" class="on" data-l="1">第 2 层</button></span>
        <span class="seg dt-head"><button type="button" class="on" data-h="-1">4 头平均</button>${[0, 1, 2, 3].map((h) => `<button type="button" data-h="${h}">头 ${h + 1}</button>`).join('')}</span>
      </div>
      <div class="dt-heat"></div>
      <p class="small muted dt-note"></p>`;
    const $ = (s) => el.querySelector(s);
    function run() {
      src = $('.dt-in').value.trim();
      const bad = [...src].filter((c) => !stoi.has(c));
      if (!src || bad.length) { $('.dt-out').textContent = '—'; $('.dt-steps').textContent = bad.length ? `模型没见过这些字符：「${[...new Set(bad)].join('')}」。它只认识训练数据里出现过的字符。` : '请输入日期'; $('.dt-heat').innerHTML = ''; return; }
      const t0 = performance.now();
      res = M.translate(W, [...src].map((c) => stoi.get(c)));
      const ms = performance.now() - t0, out = res.ids.map((i) => itos[i]);
      $('.dt-out').textContent = out.join('');
      $('.dt-steps').innerHTML = `逐字生成（括号里是模型给这个字符的概率）：${out.map((c, k) => `${App.esc(c)}<sub>${(res.probs[k][res.ids[k]] * 100).toFixed(0)}%</sub>`).join(' ')} · 用时 ${ms.toFixed(0)} 毫秒`;
      drawHeat();
    }
    function drawHeat() {
      if (!res) return;
      const pick = (A) => (head >= 0 ? A[head] : A[0].map((row, i) => row.map((_, j) => A.reduce((s, h) => s + h[i][j], 0) / A.length)));
      const ins = [...src].map((c) => (c === ' ' ? '␣' : c)), outs = res.ids.map((i) => itos[i]);
      if (view === 'cross') {
        Charts.heatmap($('.dt-heat'), pick(res.cross[layer]), { rowLabels: outs, colLabels: ins, min: 0, max: 1, rowTitle: '输出', colTitle: '输入', labelW: 24 });
        $('.dt-note').textContent = '每一行是一个输出字符：颜色越深，写这个字符时越「看」那个输入字符。写年份时看输入里的年份，写月份时看月份……注意力自己学会了「对齐」，没有任何人教它。';
      } else {
        Charts.heatmap($('.dt-heat'), pick(res.enc[layer]), { rowLabels: ins, colLabels: ins, min: 0, max: 1, rowTitle: '查询', colTitle: '键', labelW: 24 });
        $('.dt-note').textContent = '编码器里每个输入字符都会看所有输入字符，用上下文更新自己的表示——比如「月」前面的数字知道自己是月份。';
      }
    }
    el.addEventListener('click', (e) => {
      const t = e.target;
      const ex = t.closest('[data-ex]'), v = t.closest('[data-v]'), l = t.closest('[data-l]'), h = t.closest('[data-h]');
      if (ex) { $('.dt-in').value = ex.dataset.ex; run(); }
      if (t.closest('[data-a=go]')) run();
      const seg = (b, cls) => el.querySelectorAll(`${cls} button`).forEach((x) => x.classList.toggle('on', x === b));
      if (v) { view = v.dataset.v; seg(v, '.dt-view'); drawHeat(); }
      if (l) { layer = +l.dataset.l; seg(l, '.dt-layer'); drawHeat(); }
      if (h) { head = +h.dataset.h; seg(h, '.dt-head'); drawHeat(); }
    });
    $('.dt-in').addEventListener('keydown', (e) => { if (e.key === 'Enter') run(); });
    run();
  },
};
