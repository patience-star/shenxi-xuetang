/* 互动组件：脚本导出的一组小图片（灰度 [N, H, W] 或彩色 [N, H, W, 3] 的 0~255 数组）
 * 参数：images 图片数据的路径（如 'extras.wrong_x'）；captions 若干个标签数组的路径；
 *       template 图注模板，{0}、{1}… 依次替换成 captions 里对应的值；names 可把数字类别换成名字。
 * 例：{ widget: 'py-images', images: 'extras.wrong_x', captions: ['extras.wrong_y', 'extras.wrong_p'], template: '真 {0} → 判 {1}' } */
Widgets['py-images'] = {
  title: '图片',
  async render(el, opts) {
    const py = opts.py || (opts.script ? await App.loadPy(opts.script) : null);
    if (!py) throw new Error('本课的脚本数据还没有生成（运行 tools/run.py 之后才有）');
    const get = (path) => path.split('.').reduce((o, k) => (o == null ? o : o[k]), py);
    const T = NN.decode(get(opts.images));
    if (!T?.shape) throw new Error(`脚本数据里没有找到图片 ${opts.images}`);
    const [n, h, w, ch] = T.shape, caps = (opts.captions || []).map(get);
    const name = (v) => (opts.names ? opts.names[v] ?? v : v);
    el.innerHTML = `${opts.note ? `<p class="small muted">${App.mdInline(opts.note)}</p>` : ''}<div class="thumbs" style="grid-template-columns:repeat(auto-fill,minmax(${opts.size || 64}px,1fr))"></div>`;
    const box = el.querySelector('.thumbs');
    for (let i = 0; i < n; i++) {
      const f = document.createElement('figure'), c = document.createElement('canvas');
      c.className = 'thumb';
      if (ch === 3) Charts.rgbImg(c, T.d.subarray(i * h * w * 3, (i + 1) * h * w * 3), w, h);
      else Charts.img(c, T.d.subarray(i * h * w, (i + 1) * h * w), w, h, { mode: opts.mode || 'ink', min: 0, max: 255 });
      f.appendChild(c);
      if (opts.template) f.appendChild(Object.assign(document.createElement('figcaption'), { textContent: opts.template.replace(/\{(\d)\}/g, (m, k) => name(caps[+k]?.[i])) }));
      box.appendChild(f);
    }
  },
};
