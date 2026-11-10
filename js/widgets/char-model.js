/* 互动组件：字符级语言模型（第 9 课 RNN、第 10 课 LSTM，都在浏览器里运行）
 * mode = 'next'  ：输入半句诗，看下一个字的概率（点候选字可以接着往下写），以及每个字读完后的隐状态
 * mode = 'poem'  ：写藏头诗 / 随便写一首，可以调温度
 * mode = 'gates' ：（只限 LSTM）逐字查看遗忘门、输入门、输出门的开合
 * 参数：mode，model（脚本名，在案例里可省略） */
Widgets['char-model'] = {
  title: '写诗的循环网络',
  async render(el, opts) {
    const name = opts.model || opts.py?.name;
    const py = opts.py?.name === name ? opts.py : await App.loadPy(name);
    const W = NN.weights(py), M = NN.models[name], itos = NN.vocabOf(py), stoi = new Map(itos.map((c, i) => [c, i]));
    const PUNCT = new Set([stoi.get('，'), stoi.get('。')]);
    const mode = opts.mode || 'next';
    const idOf = (c) => stoi.get(c) ?? 2;
    /** 从 <s> 开始逐字读入，返回每一步的状态 */
    const read = (text) => { let st = M.step(W, 1, M.init()); const states = [st]; for (const c of text) { st = M.step(W, idOf(c), st); states.push(st); } return states; };
    const sample = (logits, T, allowPunct) => {
      const z = Float32Array.from(logits);
      z[0] = z[1] = z[2] = -1e9;
      if (!allowPunct) PUNCT.forEach((i) => { z[i] = -1e9; });
      const p = NN.softmax(z, T);
      let r = Math.random(), k = 0;
      for (; k < p.length - 1; k++) { r -= p[k]; if (r <= 0) break; }
      return { k, p: p[k] };
    };
    const vecStrip = (v, lim = 1) => `<span class="vec">${Array.from(v.subarray(0, 64), (x) => { const t = Math.min(1, Math.abs(x) / lim); return `<i style="width:6px;height:12px;background:color-mix(in srgb, var(${x >= 0 ? '--pos' : '--neg'}) ${Math.round(t * 85)}%, var(--surface-2))"></i>`; }).join('')}</span>`;

    if (mode === 'next') {
      el.innerHTML = `
        <div class="controls"><label style="flex:1 1 260px">输入一些字（只认得常用的 2500 个字）<input type="text" class="cm-in" value="${opts.text || '床前明月'}" maxlength="40"></label><button type="button" class="btn sm ghost" data-a="clear">清空</button></div>
        <div class="split"><div><h4>下一个字最可能是（点一下接着写）</h4><div class="cm-bars"></div></div>
          <div><h4>每读一个字，隐状态的前 64 维</h4><div class="cm-states small" style="max-height:330px;overflow:auto"></div></div></div>
        <p class="small muted cm-note"></p>`;
      const inp = el.querySelector('.cm-in');
      const update = () => {
        const text = inp.value.replace(/\s/g, ''), states = read(text), last = states[states.length - 1];
        const p = NN.softmax(last.logits), order = NN.topk(p.map((v, i) => (i < 3 ? -1 : v)), 8);
        el.querySelector('.cm-bars').innerHTML = `<div class="pbars">${order.map((k, j) => `<button type="button" class="pbar${j === 0 ? ' hi' : ''}" data-c="${itos[k]}" style="border:0;background:none;padding:2px 0;cursor:pointer;font:inherit;color:inherit;width:100%"><span class="lb" style="font-size:1.15rem">${itos[k]}</span><span class="track"><i style="width:${p[k] / p[order[0]] * 100}%"></i></span><b>${(p[k] * 100).toFixed(1)}%</b></button>`).join('')}</div>`;
        el.querySelector('.cm-states').innerHTML = ['〈开始〉', ...text].map((c, i) => `<div class="row" style="gap:6px;flex-wrap:nowrap"><span style="width:3.4em;flex:none;text-align:right">${App.esc(c)}</span>${vecStrip(states[i].h)}</div>`).join('');
        const unk = [...text].filter((c) => !stoi.has(c));
        el.querySelector('.cm-note').textContent = (unk.length ? `「${[...new Set(unk)].join('')}」不在字表里，被当成 <unk>。` : '') + '隐状态是一个 128 维向量（这里画前 64 维，蓝正红负）：它在读每个字时被更新，把前文的信息「压缩」带到下一步，预测完全取决于它。';
      };
      inp.addEventListener('input', update);
      el.addEventListener('click', (e) => {
        const b = e.target.closest('[data-c]');
        if (b) { inp.value += b.dataset.c; update(); }
        if (e.target.closest('[data-a=clear]')) { inp.value = ''; update(); }
      });
      update();
      return;
    }

    if (mode === 'poem') {
      el.innerHTML = `
        <div class="controls">
          <label>藏头（4 个字）<input type="text" class="cm-heads" value="${opts.heads || '深度学习'}" maxlength="4" style="width:7em"></label>
          <label>温度 T = <output class="cm-to"></output><input type="range" class="cm-t" min="0.3" max="1.6" step="0.1" value="0.8"></label>
          <span class="row"><button type="button" class="btn sm" data-a="acro">写藏头诗</button><button type="button" class="btn sm ghost" data-a="free">随便写一首</button></span>
        </div>
        <div class="cm-poem" style="font-size:1.45rem;line-height:2;letter-spacing:.12em;margin:10px 0;min-height:4em"></div>
        <p class="small muted cm-pnote">温度低：总挑最常见的字，稳但套路；温度高：更敢用生僻的字，新奇但容易不通顺。每次生成都按概率随机抽字，所以同样的设置每次结果都不同。</p>`;
      const gen = (heads) => {
        const T = +el.querySelector('.cm-t').value, lines = [];
        let st = M.step(W, 1, M.init()), line = '';
        for (let k = 0; k < 4; k++) {
          for (let pos = 0; pos < 6; pos++) {
            let c;
            if (pos === 5) c = k % 2 ? '。' : '，';
            else if (pos === 0 && heads) c = heads[k];
            else c = itos[sample(st.logits, T, false).k];
            line += c;
            st = M.step(W, idOf(c), st);
          }
          lines.push(line);
          line = '';
        }
        el.querySelector('.cm-poem').innerHTML = lines.map((l) => (heads ? `<b style="color:var(--primary)">${App.esc(l[0])}</b>${App.esc(l.slice(1))}` : App.esc(l))).join('<br>');
        const unk = heads ? [...heads].filter((c) => !stoi.has(c)) : [];
        el.querySelector('.cm-pnote').textContent = unk.length ? `「${unk.join('')}」不在模型的 2500 个常用字里，模型只能把它当成未知字，后面的字会更乱。` : '温度低：总挑最常见的字，稳但套路；温度高：更敢用生僻的字，新奇但容易不通顺。每次生成都按概率随机抽字，所以同样的设置每次结果都不同。';
      };
      el.querySelector('.cm-to').textContent = '0.8';
      el.querySelector('.cm-t').addEventListener('input', (e) => { el.querySelector('.cm-to').textContent = (+e.target.value).toFixed(1); });
      el.addEventListener('click', (e) => {
        const a = e.target.closest('[data-a]')?.dataset.a;
        if (a === 'acro') { const h = [...el.querySelector('.cm-heads').value.replace(/\s/g, '')]; if (h.length === 4) gen(h); else el.querySelector('.cm-poem').textContent = '请输入正好 4 个字。'; }
        if (a === 'free') gen(null);
      });
      gen([...(opts.heads || '深度学习')]);
      return;
    }

    // mode === 'gates'
    el.innerHTML = `
      <div class="controls"><label style="flex:1 1 260px">读入的句子<input type="text" class="cm-in" value="${opts.text || '床前明月光，疑是地上霜。'}" maxlength="24"></label></div>
      <div class="table-wrap"><table class="data cm-gates"><thead><tr><th>字</th><th>遗忘门 f（均值）</th><th>输入门 i</th><th>输出门 o</th><th>遗忘门的前 64 个单元</th></tr></thead><tbody></tbody></table></div>
      <p class="small muted">门的值在 0~1 之间：遗忘门接近 1 = 保留旧记忆，接近 0 = 清空；输入门决定写入多少新内容；输出门决定把多少记忆读出来给下一层。条形最右端代表 1。</p>`;
    const bar = (v) => `<span class="track" style="display:inline-block;width:90px;height:10px;background:var(--surface-2);border-radius:5px;overflow:hidden;vertical-align:middle"><i style="display:block;height:100%;width:${v * 100}%;background:var(--c1)"></i></span> ${v.toFixed(2)}`;
    const mean = (v) => v.reduce((a, b) => a + b, 0) / v.length;
    const update = () => {
      const text = el.querySelector('.cm-in').value.replace(/\s/g, ''), states = read(text);
      el.querySelector('.cm-gates tbody').innerHTML = [...text].map((c, i) => {
        const s = states[i + 1];
        return `<tr><td style="font-size:1.1rem">${App.esc(c)}</td><td>${bar(mean(s.f))}</td><td>${bar(mean(s.i))}</td><td>${bar(mean(s.o))}</td><td>${vecStrip(s.f.map((x) => x * 2 - 1))}</td></tr>`;
      }).join('');
    };
    el.querySelector('.cm-in').addEventListener('input', update);
    update();
  },
};
