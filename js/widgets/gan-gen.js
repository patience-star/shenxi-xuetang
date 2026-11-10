/* 互动组件：DCGAN 生成手写数字（第 13 课，在浏览器里运行）
 * ① 随机噪声 → 16 个数字；② 两个噪声之间插值，看数字怎样平滑地变形；③ 训练过程画廊：同一批噪声在不同训练轮次画出的样子。 */
Widgets['gan-gen'] = {
  title: '生成器画的数字',
  async render(el, opts) {
    const name = opts.model || opts.py?.name || '13_gan_mnist';
    const py = opts.py?.name === name ? opts.py : await App.loadPy(name);
    const W = NN.weights(py), M = NN.models[name], Z = py.extras.z_dim || 32, r = NN.rng(Date.now() % 1e6);
    const snaps = NN.decode(py.extras.snapshots), epochs = py.extras.snap_epochs;
    const noise = () => Float32Array.from({ length: Z }, () => r.randn());
    const toImg = (v) => Float32Array.from(v, (x) => (x + 1) / 2);
    let z1 = noise(), z2 = noise();
    el.innerHTML = `
      <h4>① 随机生成</h4>
      <div class="row"><button type="button" class="btn sm" data-a="new">🎲 换一批噪声</button><span class="small muted gg-t"></span></div>
      <div class="thumbs gg-grid" style="grid-template-columns:repeat(8,minmax(36px,70px));margin-top:8px"></div>
      <h4>② 隐空间插值：从噪声 z₁ 平滑走到 z₂</h4>
      <div class="row"><button type="button" class="btn sm ghost" data-a="ends">换两个端点</button><label class="small">t = <output class="gg-o">0.50</output> <input type="range" class="gg-s" min="0" max="1" step="0.02" value="0.5"></label></div>
      <div class="row" style="align-items:center;gap:14px;margin-top:8px"><canvas class="gg-big big-thumb" style="max-width:120px"></canvas><div class="thumbs gg-row" style="grid-template-columns:repeat(9,minmax(30px,56px));flex:1"></div></div>
      <h4>③ 训练过程：同一批 16 个噪声在不同轮次画出的数字</h4>
      <div class="seg gg-ep">${epochs.map((e, i) => `<button type="button" data-i="${i}"${i === epochs.length - 1 ? ' class="on"' : ''}>第 ${e} 轮</button>`).join('')}</div>
      <div class="thumbs gg-snap" style="grid-template-columns:repeat(8,minmax(36px,70px));margin-top:8px"></div>`;
    const $ = (s) => el.querySelector(s);
    const fill = (box, imgs) => { box.innerHTML = ''; imgs.forEach((v) => box.appendChild(Charts.canvasOf(v, 28, 28, { mode: 'ink', min: 0, max: 1 }))); };
    function grid() {
      const t0 = performance.now(), imgs = Array.from({ length: 16 }, () => toImg(M.forward(W, noise())));
      fill($('.gg-grid'), imgs);
      $('.gg-t').textContent = `生成 16 张用时 ${(performance.now() - t0).toFixed(0)} 毫秒。每张图都来自一组随机的 ${Z} 个数——训练集里并没有这些图。`;
    }
    const lerp = (t) => Float32Array.from(z1, (v, k) => v + (z2[k] - v) * t);
    function inter() {
      fill($('.gg-row'), Array.from({ length: 9 }, (_, k) => toImg(M.forward(W, lerp(k / 8)))));
      const t = +$('.gg-s').value;
      $('.gg-o').textContent = t.toFixed(2);
      Charts.img($('.gg-big'), toImg(M.forward(W, lerp(t))), 28, 28, { mode: 'ink', min: 0, max: 1 });
    }
    function snap(i) {
      el.querySelectorAll('.gg-ep button').forEach((b, k) => b.classList.toggle('on', k === i));
      fill($('.gg-snap'), Array.from({ length: 16 }, (_, k) => Float32Array.from(snaps.d.subarray((i * 16 + k) * 784, (i * 16 + k + 1) * 784), (v) => v / 255)));
    }
    el.addEventListener('click', (e) => {
      const a = e.target.closest('[data-a]')?.dataset.a, ep = e.target.closest('[data-i]');
      if (a === 'new') grid();
      if (a === 'ends') { z1 = noise(); z2 = noise(); inter(); }
      if (ep) snap(+ep.dataset.i);
    });
    $('.gg-s').addEventListener('input', () => { const t = +$('.gg-s').value; $('.gg-o').textContent = t.toFixed(2); Charts.img($('.gg-big'), toImg(M.forward(W, lerp(t))), 28, 28, { mode: 'ink', min: 0, max: 1 }); });
    grid(); inter(); snap(epochs.length - 1);
  },
};
