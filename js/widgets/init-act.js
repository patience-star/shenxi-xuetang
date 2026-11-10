/* 互动组件：初始化与各层激活值
 * 一个 10 层、每层 100 个神经元的网络，喂一批随机输入，画出每一层输出值的分布（直方图），
 * 再从输出端反向传回随机梯度，看每层收到的梯度有多大。换不同的初始化和激活函数对比。 */
Widgets['init-act'] = {
  title: '初始化：信号能不能顺利穿过 10 层？',
  render(el) {
    el.innerHTML = `
      <div class="controls">
        <label>权重初始化<select class="ia-init">
          <option value="small">太小：N(0, 0.01²)</option><option value="large">太大：N(0, 1)</option>
          <option value="xavier" selected>Xavier：方差 1/输入数</option><option value="he">He：方差 2/输入数</option></select></label>
        <label>激活函数<select class="ia-act"><option value="tanh">tanh</option><option value="relu">ReLU</option><option value="sigmoid">sigmoid</option></select></label>
        <button type="button" class="btn sm ghost" data-a="go">换一批随机数再算</button>
      </div>
      <p class="small muted">每个小图是一层输出值的直方图（横轴 −1 ~ 1，ReLU / sigmoid 为 0 ~ 1），下面是这一层输出的标准差：</p>
      <div class="ia-hist" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(80px,1fr));gap:6px"></div>
      <h4>反向传播：每层收到的梯度大小（标准差，对数刻度）</h4>
      <div class="ia-grad"></div>
      <p class="verdict ia-v"></p>`;
    const $ = (s) => el.querySelector(s);
    const N = 200, D = 100, L = 10;
    let seed = 1;
    function run() {
      const r = NN.rng(seed), init = $('.ia-init').value, act = $('.ia-act').value;
      const std = { small: 0.01, large: 1, xavier: Math.sqrt(1 / D), he: Math.sqrt(2 / D) }[init];
      const f = { tanh: Math.tanh, relu: (z) => Math.max(0, z), sigmoid: (z) => 1 / (1 + Math.exp(-z)) }[act];
      const df = { tanh: (z, a) => 1 - a * a, relu: (z) => (z > 0 ? 1 : 0), sigmoid: (z, a) => a * (1 - a) }[act];
      let x = Array.from({ length: N * D }, () => r.randn());
      const Ws = [], Zs = [], As = [];
      for (let l = 0; l < L; l++) {
        const Wl = Float64Array.from({ length: D * D }, () => r.randn() * std), z = new Float64Array(N * D), a = new Float64Array(N * D);
        for (let n = 0; n < N; n++) for (let j = 0; j < D; j++) {
          let s = 0;
          for (let i = 0; i < D; i++) s += x[n * D + i] * Wl[i * D + j];
          z[n * D + j] = s; a[n * D + j] = f(s);
        }
        Ws.push(Wl); Zs.push(z); As.push(a); x = a;
      }
      // 反向：输出端给随机梯度，逐层往回传 dx = (dy ⊙ f'(z)) · Wᵀ
      let g = Array.from({ length: N * D }, () => r.randn());
      const gstd = new Array(L);
      for (let l = L - 1; l >= 0; l--) {
        const dz = g.map((v, k) => v * df(Zs[l][k], As[l][k])), dx = new Float64Array(N * D), Wl = Ws[l];
        for (let n = 0; n < N; n++) for (let i = 0; i < D; i++) { let s = 0; for (let j = 0; j < D; j++) s += dz[n * D + j] * Wl[i * D + j]; dx[n * D + i] = s; }
        const m = dx.reduce((a, b) => a + b, 0) / dx.length;
        gstd[l] = Math.sqrt(dx.reduce((a, b) => a + (b - m) ** 2, 0) / dx.length);
        g = dx;
      }
      // 画直方图
      const lo = act === 'tanh' ? -1 : 0, hi = 1, B = 24, css = Charts.css;
      $('.ia-hist').innerHTML = '';
      As.forEach((a, l) => {
        const h = new Array(B).fill(0);
        let m = 0; for (const v of a) m += v; m /= a.length;
        let sd = 0; for (const v of a) sd += (v - m) ** 2; sd = Math.sqrt(sd / a.length);
        for (const v of a) { const b = Math.floor(((Math.max(lo, Math.min(hi - 1e-9, v)) - lo) / (hi - lo)) * B); h[b]++; }
        const fig = document.createElement('figure'), c = document.createElement('canvas');
        fig.style.margin = '0';
        c.className = 'viz';
        const ctx = Charts.hidpi(c, 80, 56), mx = Math.max(...h);
        ctx.fillStyle = css('--c1');
        h.forEach((v, b) => { const bh = (v / mx) * 50; ctx.fillRect((b / B) * 80, 54 - bh, 80 / B - 0.6, bh); });
        fig.append(c, Object.assign(document.createElement('figcaption'), { className: 'small muted', textContent: `第 ${l + 1} 层 σ=${sd < 0.001 ? sd.toExponential(0) : sd.toFixed(3)}` }));
        $('.ia-hist').appendChild(fig);
      });
      Charts.columns($('.ia-grad'), gstd.map((v, l) => ({ label: `${l + 1}`, value: Math.log10(Math.max(v, 1e-30)) })), { format: (v) => `1e${Math.round(v)}`, height: 180, table: false, color: () => 'var(--c2)' });
      const last = As[L - 1], ok = gstd[0] > 1e-3 && gstd[0] < 1e3;
      let mean = 0; for (const v of last) mean += Math.abs(v); mean /= last.length;
      const v = $('.ia-v');
      v.className = `verdict ${ok && mean > 0.05 && mean < 0.95 ? 'ok' : 'warn'} ia-v`;
      v.textContent = init === 'small' ? '权重太小：每过一层信号都被缩小，几层之后输出几乎全是 0，反向传回的梯度也一样趋近 0——网络学不动。'
        : init === 'large' && act !== 'relu' ? '权重太大：加权和动辄几十，tanh / sigmoid 全部饱和在两端，导数≈0，梯度同样消失。'
        : init === 'large' ? '权重太大 + ReLU：信号每层放大约 7 倍，十层后大到上千万倍，梯度爆炸。'
        : init === 'xavier' && act === 'relu' ? 'ReLU 会把一半的输入置 0，用 Xavier 时信号每层缩小一点，层数多了还是会衰减——ReLU 应该配 He 初始化（方差多乘一个 2，补回被砍掉的那一半）。'
        : act === 'sigmoid' ? 'sigmoid 的输出都是正数、导数最大只有 0.25，即使初始化合适，梯度还是一层层变小：这就是现代网络很少在中间层用 sigmoid 的原因。'
        : '初始化合适：各层输出的分布保持稳定，梯度也能顺利传回第一层。PyTorch 的 nn.Linear 默认使用的就是这类按输入数缩放的初始化。';
    }
    el.addEventListener('change', run);
    el.addEventListener('click', (e) => { if (e.target.closest('[data-a=go]')) { seed++; run(); } });
    run();
  },
};
