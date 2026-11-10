/* 互动组件：真实训练记录的曲线图 / 柱状图
 * 数据来自 tools/run.py 导出的 history、extras（脚本真实运行的结果），悬停可看每个点的数值，图下可展开数据表。
 * 参数：
 *   script   脚本名（在经典案例里可省略，自动用本课脚本）
 *   series   [{ path: 'history.train_loss', name: '训练损失', color? }]   path 是在脚本数据里的路径
 *   x        横轴标签的路径（可省略：用 x0 + i·xStep，默认从 1 开始）
 *   kind     'line'（默认）| 'columns'（只画第一个系列）
 *   format   'num'（默认）| 'pct' | 'int'；digits 小数位
 *   log      true：纵轴用对数刻度（损失跨好几个数量级时）
 *   cap      数值上限（发散的曲线截断显示）
 *   yLabel / xLabel / note / height / yMin
 */
Widgets['py-chart'] = {
  title: '训练记录',
  async render(el, opts) {
    const py = opts.py || (opts.script ? await App.loadPy(opts.script) : null);
    if (!py) throw new Error('本课的脚本数据还没有生成（运行 tools/run.py 之后才有）');
    const get = (path) => path.split('.').reduce((o, k) => (o == null ? o : o[k]), py);
    const base = { pct: (v) => Charts.fmt.pct(v, opts.digits ?? 1), int: Charts.fmt.int, num: (v) => Charts.fmt.num(v, opts.digits ?? 3) }[opts.format || 'num'];
    const tr = (v) => (v == null || !Number.isFinite(v) ? null : opts.log ? Math.log10(Math.max(opts.cap != null ? Math.min(v, opts.cap) : v, 1e-12)) : opts.cap != null ? Math.min(v, opts.cap) : v);
    const fmt = opts.log ? (v) => base(10 ** v) : base;
    const series = opts.series.map((s) => {
      const raw = get(s.path);
      if (!Array.isArray(raw)) throw new Error(`脚本数据里没有找到 ${s.path}`);
      return { name: s.name, color: s.color, values: raw.map(tr) };
    });
    const n = Math.max(...series.map((s) => s.values.length));
    series.forEach((s) => { while (s.values.length < n) s.values.push(null); });
    const xs = opts.x ? get(opts.x) : null;
    const labels = Array.from({ length: n }, (_, i) => (xs ? String(xs[i]) : String((opts.x0 ?? 1) + i * (opts.xStep ?? 1))));
    el.innerHTML = `${opts.note ? `<p class="small muted">${App.mdInline(opts.note)}</p>` : ''}${opts.yLabel ? `<p class="small muted" style="margin:0">纵轴：${App.mdInline(opts.yLabel)}${opts.log ? '（对数刻度）' : ''}</p>` : ''}<div class="pc"></div>`;
    const box = el.querySelector('.pc');
    if (opts.kind === 'columns') Charts.columns(box, series[0].values.map((v, i) => ({ label: labels[i], value: v })), { format: fmt, height: opts.height, yLabel: opts.yLabel });
    else Charts.line(box, { labels, series }, { format: fmt, height: opts.height || 260, yMin: opts.yMin });
    if (opts.xLabel) box.querySelector('svg')?.insertAdjacentHTML('afterend', `<p class="small muted" style="text-align:center;margin:2px 0 0">横轴：${App.esc(opts.xLabel)}</p>`);
  },
};
