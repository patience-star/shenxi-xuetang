/* 互动组件：感受野
 * 三层 3×3 卷积叠在一起（可选在中间加 2×2 池化）。指针移到最上层的某个格子，
 * 看它在下面每一层「看得到」多大的区域——层数越多，看到的范围越大。 */
Widgets['receptive-field'] = {
  title: '感受野：越往上，看得越广',
  render(el) {
    el.innerHTML = `
      <div class="controls"><label class="inline"><input type="checkbox" class="rf-pool"> 第一层卷积后加 2×2 最大池化（步幅 2）</label></div>
      <div class="diagram"><svg class="rf-svg" viewBox="0 0 420 470" role="img" aria-label="感受野示意" style="max-width:460px;min-width:0"></svg></div>
      <p class="verdict rf-v"></p>`;
    const $ = (s) => el.querySelector(s);
    let sel = null;
    function layersOf(pool) {
      // 每层：{ n: 边长, 名称 }，以及从上一层到这一层的操作（卷积 3×3 或池化 2×2）
      return pool
        ? [{ n: 14, name: '输入 14×14' }, { n: 12, name: '卷积 3×3 → 12×12', op: 'c' }, { n: 6, name: '池化 2×2 → 6×6', op: 'p' }, { n: 4, name: '卷积 3×3 → 4×4', op: 'c' }]
        : [{ n: 9, name: '输入 9×9' }, { n: 7, name: '卷积 3×3 → 7×7', op: 'c' }, { n: 5, name: '卷积 3×3 → 5×5', op: 'c' }, { n: 3, name: '卷积 3×3 → 3×3', op: 'c' }];
    }
    function draw() {
      const L = layersOf($('.rf-pool').checked), top = L.length - 1;
      if (!sel || sel[0] >= L[top].n || sel[1] >= L[top].n) sel = [Math.floor(L[top].n / 2), Math.floor(L[top].n / 2)];
      // 从最上层往下算每层的区域 [r0, r1] × [c0, c1]
      const region = new Array(L.length);
      region[top] = [sel[0], sel[0], sel[1], sel[1]];
      for (let l = top; l > 0; l--) {
        const [r0, r1, c0, c1] = region[l];
        region[l - 1] = L[l].op === 'c' ? [r0, r1 + 2, c0, c1 + 2] : [2 * r0, 2 * r1 + 1, 2 * c0, 2 * c1 + 1];
      }
      let s = '', y = 6;
      for (let l = top; l >= 0; l--) {
        const n = L[l].n, cs = Math.min(22, 260 / n), x0 = 210 - (n * cs) / 2, [r0, r1, c0, c1] = region[l];
        for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
          const on = r >= r0 && r <= r1 && c >= c0 && c <= c1;
          s += `<rect x="${x0 + c * cs}" y="${y + r * cs}" width="${cs - 1.5}" height="${cs - 1.5}" rx="2" class="box ${on ? 'hl' : ''}" ${l === top ? `data-r="${r}" data-c="${c}" style="cursor:pointer"` : ''}/>`;
        }
        s += `<text x="${x0 - 8}" y="${y + (n * cs) / 2 + 4}" text-anchor="end" class="sub">${L[l].name.split(' ')[0]}</text>`;
        s += `<text x="${x0 + n * cs + 8}" y="${y + (n * cs) / 2 + 4}" class="sub">${r1 - r0 + 1}×${c1 - c0 + 1}</text>`;
        y += n * cs + 18;
      }
      $('.rf-svg').setAttribute('viewBox', `0 0 420 ${y}`);
      $('.rf-svg').innerHTML = s;
      const [a, b, c, d] = region[0];
      $('.rf-v').textContent = `最上层的 1 个格子，能「看到」输入上 ${b - a + 1}×${d - c + 1} 的区域。${$('.rf-pool').checked ? '池化把边长减半，之后每层卷积的感受野增长得快一倍——这就是 CNN 为什么要交替使用卷积和池化。' : '每加一层 3×3 卷积，感受野边长 +2：底层只看局部的笔画边缘，高层就能看到整个数字的形状。'}`;
    }
    el.addEventListener('pointerover', (e) => { const r = e.target.closest('[data-r]'); if (r) { sel = [+r.dataset.r, +r.dataset.c]; draw(); } });
    el.addEventListener('pointerdown', (e) => { const r = e.target.closest('[data-r]'); if (r) { sel = [+r.dataset.r, +r.dataset.c]; draw(); } });
    $('.rf-pool').addEventListener('change', () => { sel = null; draw(); });
    draw();
  },
};
