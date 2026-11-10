/* 互动组件：ResNet-20 图像分类（CIFAR-10，在浏览器里运行）
 * 点一张测试图看模型的判断；也可以上传或用手机拍一张照片（会被缩到 32×32，和训练数据一样小）。 */
Widgets['cifar-demo'] = {
  title: 'ResNet-20 图像分类：点一张图试试',
  async render(el, opts) {
    const name = opts.model || opts.py?.name || '08_resnet_cifar10';
    const py = opts.py?.name === name ? opts.py : await App.loadPy(name);
    const W = NN.weights(py), M = NN.models[name], C = py.extras.classes;
    const X = NN.decode(py.extras.demo_x), Y = py.extras.demo_y, n = X.shape[0];
    el.innerHTML = `
      <div class="split">
        <div>
          <div class="thumbs cd-thumbs" style="grid-template-columns:repeat(auto-fill,minmax(48px,1fr))"></div>
          <div class="row" style="margin-top:10px"><label class="btn sm ghost" style="cursor:pointer">📷 上传 / 拍照<input type="file" accept="image/*" class="cd-file" hidden></label><span class="small muted">图片只在你的浏览器里处理，不会上传到任何地方。</span></div>
        </div>
        <div>
          <div class="row" style="align-items:flex-end;gap:16px"><div><div class="small muted">网络看到的 32×32</div><canvas class="cd-big big-thumb" style="max-width:150px"></canvas></div>
            <div><div class="small muted">预测</div><div class="cd-pred" style="font-size:2rem;font-weight:700">—</div><div class="cd-meta small muted"></div></div></div>
          <div class="cd-bars" style="margin-top:10px"></div>
        </div>
      </div>
      <p class="verdict cd-v"></p>`;
    const $ = (s) => el.querySelector(s);
    function predict(hwc, label) {
      Charts.rgbImg($('.cd-big'), hwc, 32, 32);
      const chw = new Float32Array(3072);
      for (let i = 0; i < 1024; i++) for (let c = 0; c < 3; c++) chw[c * 1024 + i] = hwc[3 * i + c] / 255;
      const t0 = performance.now(), out = M.forward(W, chw), ms = performance.now() - t0;
      const order = NN.topk(out.probs, 5), k = order[0];
      $('.cd-pred').textContent = C[k];
      $('.cd-meta').textContent = `${label != null ? `真实类别：${C[label]} ${label === k ? '✓' : '✗'} · ` : ''}推理用时 ${ms.toFixed(0)} 毫秒`;
      Charts.bars($('.cd-bars'), order.map((j) => ({ label: C[j], p: out.probs[j] })), { hi: 0 });
      $('.cd-v').textContent = label == null ? '这是你自己的图片：注意它被缩到了 32×32，背景、角度、光线都和 CIFAR-10 不同（这叫「分布偏移」），模型认错很正常——训练数据决定了模型的眼界。'
        : label === k ? '' : `认错了：${C[label]} 被认成了 ${C[k]}。放大看看这张 32×32 的小图，有时人也很难分辨。`;
    }
    for (let i = 0; i < n; i++) {
      const c = document.createElement('canvas');
      c.className = 'thumb';
      c.tabIndex = 0;
      c.title = C[Y[i]];
      Charts.rgbImg(c, X.d.subarray(i * 3072, (i + 1) * 3072), 32, 32);
      const pick = () => { $('.cd-thumbs').querySelectorAll('canvas').forEach((x) => x.classList.toggle('on', x === c)); predict(X.d.subarray(i * 3072, (i + 1) * 3072), Y[i]); };
      c.addEventListener('click', pick);
      c.addEventListener('keydown', (e) => { if (e.key === 'Enter') pick(); });
      $('.cd-thumbs').appendChild(c);
    }
    $('.cd-file').addEventListener('change', (e) => {
      const f = e.target.files[0];
      if (!f) return;
      const img = new Image();
      img.onload = () => {
        const s = Math.min(img.width, img.height), cv = document.createElement('canvas');
        cv.width = cv.height = 32;
        const ctx = cv.getContext('2d');
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(img, (img.width - s) / 2, (img.height - s) / 2, s, s, 0, 0, 32, 32);   // 居中裁成正方形再缩小
        const d = ctx.getImageData(0, 0, 32, 32).data, hwc = new Uint8Array(3072);
        for (let i = 0; i < 1024; i++) { hwc[3 * i] = d[4 * i]; hwc[3 * i + 1] = d[4 * i + 1]; hwc[3 * i + 2] = d[4 * i + 2]; }
        URL.revokeObjectURL(img.src);
        $('.cd-thumbs').querySelectorAll('canvas').forEach((x) => x.classList.remove('on'));
        predict(hwc, null);
      };
      img.src = URL.createObjectURL(f);
    });
    $('.cd-thumbs canvas').click();
  },
};
