/* 互动组件：softmax 与交叉熵
 * 拖动 4 个类别的「得分」（logits），看 softmax 怎样把它们变成概率、交叉熵损失是多少，
 * 以及反向传播时每个得分收到的梯度（概率 − 真实值）。 */
Widgets['softmax-ce'] = {
  title: 'softmax + 交叉熵：从得分到概率再到损失',
  render(el) {
    const names = ['猫', '狗', '兔', '鸟'];
    const z = [2.0, 1.0, 0.1, -1.0];
    let truth = 0, T = 1;
    el.innerHTML = `
      <div class="controls">${names.map((n, i) => `<label>「${n}」的得分 z = <output class="sm-o${i}"></output><input type="range" class="sm-z" data-i="${i}" min="-5" max="5" step="0.1" value="${z[i]}"></label>`).join('')}</div>
      <div class="controls">
        <label>真实类别<select class="sm-t">${names.map((n, i) => `<option value="${i}">${n}</option>`).join('')}</select></label>
        <label>温度 T = <output class="sm-to"></output><input type="range" class="sm-temp" min="0.2" max="5" step="0.1" value="1"></label>
      </div>
      <div class="split">
        <div><h4>softmax 概率</h4><div class="sm-bars"></div></div>
        <div><h4>反向传播：每个得分的梯度（p − y）</h4><div class="sm-grad"></div></div>
      </div>
      <div class="table-wrap" style="margin-top:12px"><table class="data sm-tab"></table></div>
      <p class="verdict sm-v"></p>`;
    const $ = (s) => el.querySelector(s);
    function draw() {
      const e = z.map((v) => Math.exp(v / T)), sum = e.reduce((a, b) => a + b, 0), p = e.map((v) => v / sum);
      const loss = -Math.log(p[truth]), k = p.indexOf(Math.max(...p));
      Charts.bars($('.sm-bars'), names.map((n, i) => ({ label: n + (i === truth ? ' ✓' : ''), p: p[i] })), { hi: k });
      $('.sm-grad').innerHTML = `<div class="pbars">${names.map((n, i) => {
        const g = p[i] - (i === truth ? 1 : 0), w = Math.min(50, Math.abs(g) * 50);
        return `<div class="pbar"><span class="lb">${n}</span><span class="track" style="position:relative;background:var(--surface-2)"><i style="position:absolute;top:0;bottom:0;${g >= 0 ? `left:50%;width:${w}%;background:var(--neg)` : `right:50%;width:${w}%;background:var(--pos)`}"></i><i style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--axis)"></i></span><b>${g >= 0 ? '+' : ''}${g.toFixed(3)}</b></div>`;
      }).join('')}</div><p class="small muted">梯度下降会沿<b>负梯度</b>更新：蓝色（负梯度）的得分被推高，红色的被压低——正确类别往上推，其他类别往下压。</p>`;
      $('.sm-tab').innerHTML = `<thead><tr><th></th>${names.map((n) => `<th>${n}</th>`).join('')}<th>合计</th></tr></thead><tbody>
        <tr><td>得分 z</td>${z.map((v) => `<td class="num">${v.toFixed(1)}</td>`).join('')}<td></td></tr>
        <tr><td>e^(z/T)</td>${e.map((v) => `<td class="num">${v.toFixed(3)}</td>`).join('')}<td class="num">${sum.toFixed(3)}</td></tr>
        <tr><td>概率 = e^(z/T) ÷ 合计</td>${p.map((v) => `<td class="num">${v.toFixed(3)}</td>`).join('')}<td class="num">1.000</td></tr></tbody>`;
      $('.sm-v').innerHTML = `交叉熵损失 = −log(p<sub>${names[truth]}</sub>) = −log(${p[truth].toFixed(3)}) = <b>${loss.toFixed(3)}</b>。${p[truth] > 0.9 ? '正确类别的概率很高，损失接近 0。' : p[truth] < 0.1 ? '模型对正确答案很没信心，损失很大——梯度也最大，会被狠狠纠正。' : ''}`;
      names.forEach((_, i) => { $(`.sm-o${i}`).textContent = z[i].toFixed(1); });
      $('.sm-to').textContent = T.toFixed(1);
    }
    el.querySelectorAll('.sm-z').forEach((inp) => inp.addEventListener('input', () => { z[+inp.dataset.i] = +inp.value; draw(); }));
    $('.sm-t').addEventListener('change', (e) => { truth = +e.target.value; draw(); });
    $('.sm-temp').addEventListener('input', (e) => { T = +e.target.value; draw(); });
    draw();
  },
};
