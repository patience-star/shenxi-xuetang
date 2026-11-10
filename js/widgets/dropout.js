/* 互动组件：Dropout
 * 训练模式：每次前向随机「关掉」一部分隐藏神经元（灰色虚线），剩下的输出放大 1/(1−p) 倍；
 * 测试模式：所有神经元都工作、不放大。每次训练相当于在训练一个不同的「子网络」。 */
Widgets.dropout = {
  title: 'Dropout：每次随机关掉一些神经元',
  render(el) {
    const layers = [4, 7, 7, 3];
    let mode = 'train', p = 0.5, mask = null, count = 0;
    el.innerHTML = `
      <div class="controls">
        <label>丢弃概率 p = <output class="do-o"></output><input type="range" class="do-p" min="0.1" max="0.9" step="0.1" value="0.5"></label>
        <span class="row"><button type="button" class="btn sm" data-a="train">训练模式：前向一次</button><button type="button" class="btn sm ghost" data-a="test">测试模式</button></span>
      </div>
      <div class="diagram"><svg class="do-svg" viewBox="0 0 600 300" role="img" aria-label="带 Dropout 的网络"></svg></div>
      <p class="verdict do-v"></p>`;
    const $ = (s) => el.querySelector(s);
    const pos = (l, i) => [70 + l * 150, 150 + (i - (layers[l] - 1) / 2) * 38];
    function sample() { mask = layers.map((n, l) => Array.from({ length: n }, () => (l === 0 || l === layers.length - 1 ? true : Math.random() >= p))); count++; }
    function draw() {
      const on = (l, i) => mode === 'test' || !mask || mask[l][i];
      let s = '';
      for (let l = 0; l < layers.length - 1; l++) for (let i = 0; i < layers[l]; i++) for (let j = 0; j < layers[l + 1]; j++) {
        if (!on(l, i) || !on(l + 1, j)) continue;
        const [x1, y1] = pos(l, i), [x2, y2] = pos(l + 1, j);
        s += `<line x1="${x1 + 13}" y1="${y1}" x2="${x2 - 13}" y2="${y2}" class="arrow" style="stroke-width:1;opacity:.55"/>`;
      }
      layers.forEach((n, l) => {
        for (let i = 0; i < n; i++) {
          const [x, y] = pos(l, i), alive = on(l, i), hidden = l > 0 && l < layers.length - 1;
          s += `<circle cx="${x}" cy="${y}" r="13" class="${alive ? 'node' : 'box'}" style="${alive ? '' : 'stroke-dasharray:3 3;opacity:.45'}"/>`;
          if (hidden && alive && mode === 'train' && mask) s += `<text x="${x}" y="${y + 4}" text-anchor="middle" class="sub" style="font-size:9px">×${(1 / (1 - p)).toFixed(1)}</text>`;
          if (!alive) s += `<text x="${x}" y="${y + 4}" text-anchor="middle" class="sub">✕</text>`;
        }
      });
      ['输入层', '隐藏层 1', '隐藏层 2', '输出层'].forEach((t, l) => { s += `<text x="${pos(l, 0)[0]}" y="292" text-anchor="middle" class="sub">${t}</text>`; });
      $('.do-svg').innerHTML = s;
      const kept = mask ? mask.slice(1, -1).flat().filter(Boolean).length : 14;
      $('.do-v').innerHTML = mode === 'test'
        ? '<b>测试模式</b>（<code>model.eval()</code>）：所有神经元都参与，也不再放大。效果相当于把训练时见过的大量子网络「平均」起来——这就是 Dropout 抗过拟合的原理。'
        : mask ? `<b>训练模式</b>（<code>model.train()</code>）第 ${count} 次前向：14 个隐藏神经元保留了 ${kept} 个。被关掉的神经元这一步既不输出、也不更新；留下的输出乘以 1/(1−p) = ${(1 / (1 - p)).toFixed(2)}，保证下一层收到的信号总量期望不变。再点一次，会是另一组神经元——网络不能依赖任何一个特定的神经元，只能学到更「稳健」的特征。`
        : '点「训练模式：前向一次」看看会关掉哪些神经元。';
      $('.do-o').textContent = p.toFixed(1);
    }
    el.addEventListener('click', (e) => {
      const a = e.target.closest('[data-a]')?.dataset.a;
      if (a === 'train') { mode = 'train'; sample(); draw(); }
      if (a === 'test') { mode = 'test'; draw(); }
    });
    $('.do-p').addEventListener('input', (e) => { p = +e.target.value; if (mask && mode === 'train') sample(); draw(); });
    draw();
  },
};
