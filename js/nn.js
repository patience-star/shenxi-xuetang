/* =====================================================================
 * 浏览器里的小型神经网络库（纯 JavaScript，不依赖第三方库）
 *  ① 读取 tools/run.py 导出的权重（float16 → Float32Array）
 *  ② 推理算子：全连接、卷积、转置卷积、池化、BatchNorm、LayerNorm、RNN / LSTM 单元、多头注意力
 *  ③ 经典案例模型的前向计算：与 pytorch/ 下的脚本一一对应，
 *     站点自检用「黄金样例」核对两边输出一致（见 goldenCheck）
 *  ④ 可在浏览器里实时训练的小 MLP（反向传播 + SGD / Momentum / Adam），供训练场等互动组件使用
 * 约定：向量用 Float32Array；图像特征用 { d, c, h, w }，内存布局与 PyTorch 相同（C×H×W，行优先）；
 *       权重用 { d, shape }，形状与 PyTorch 的 state_dict 一致（全连接 [输出, 输入]，卷积 [输出通道, 输入通道, kh, kw]）。
 * ===================================================================== */
(function () {
  'use strict';

  /* ---------------- ① 解码 ---------------- */
  function bytes(b64) {
    const s = atob(b64), u = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
    return u;
  }
  let F16 = null;   // float16 → float32 查找表（65536 项，首次使用时生成）
  function f16table() {
    F16 = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      F16[h] = e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
    }
  }
  /** 导出格式 { shape, f16 | f32 | u8: base64 } → { d, shape } */
  function decode(t) {
    if (!t || t.d) return t;
    if (t.f32) return { d: new Float32Array(bytes(t.f32).buffer), shape: t.shape };
    if (t.u8) return { d: bytes(t.u8), shape: t.shape };
    if (t.i32) return { d: new Int32Array(bytes(t.i32).buffer), shape: t.shape };
    if (t.f16) {
      if (!F16) f16table();
      const u = new Uint16Array(bytes(t.f16).buffer), d = new Float32Array(u.length);
      for (let i = 0; i < u.length; i++) d[i] = F16[u[i]];
      return { d, shape: t.shape };
    }
    return t;
  }
  const wcache = new WeakMap();
  /** 某个脚本导出的全部权重：{ 'fc1.weight': { d, shape }, ... }（解码一次后缓存） */
  function weights(py) {
    if (!wcache.has(py)) {
      const W = {};
      for (const [k, t] of Object.entries(py.weights || {})) W[k] = t.same ? W[t.same] : decode(t);   // same：与另一层共享的权重
      wcache.set(py, W);
    }
    return wcache.get(py);
  }

  /* ---------------- ② 推理算子 ---------------- */
  /** 全连接 y = W·x + b（W 形状 [输出, 输入]） */
  function linear(x, W, b) {
    const [O, I] = W.shape, w = W.d, y = new Float32Array(O);
    for (let o = 0; o < O; o++) {
      let s = b ? b.d[o] : 0;
      const off = o * I;
      for (let i = 0; i < I; i++) s += w[off + i] * x[i];
      y[o] = s;
    }
    return y;
  }
  const sigmoid = (v) => 1 / (1 + Math.exp(-v));
  function relu(x) { for (let i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0; return x; }       // 原地修改
  function tanhIn(x) { for (let i = 0; i < x.length; i++) x[i] = Math.tanh(x[i]); return x; }
  function sigmoidIn(x) { for (let i = 0; i < x.length; i++) x[i] = sigmoid(x[i]); return x; }
  /** softmax（先减最大值防止溢出；T 是温度，越大越平均） */
  function softmax(x, T = 1) {
    let m = -Infinity;
    for (const v of x) if (v / T > m) m = v / T;
    const e = Float32Array.from(x, (v) => Math.exp(v / T - m));
    let s = 0;
    for (const v of e) s += v;
    for (let i = 0; i < e.length; i++) e[i] /= s;
    return e;
  }
  const argmax = (x) => { let k = 0; for (let i = 1; i < x.length; i++) if (x[i] > x[k]) k = i; return k; };
  /** 从大到小排序的前 k 个下标 */
  const topk = (x, k) => Array.from(x.keys()).sort((a, b) => x[b] - x[a]).slice(0, k);

  /** 二维卷积：x { d, c, h, w }，W [输出通道, 输入通道, kh, kw] */
  function conv2d(x, W, b, stride = 1, pad = 0) {
    const [OC, IC, KH, KW] = W.shape, { h, w } = x, wd = W.d, xd = x.d;
    const OH = Math.floor((h + 2 * pad - KH) / stride) + 1, OW = Math.floor((w + 2 * pad - KW) / stride) + 1;
    const out = new Float32Array(OC * OH * OW);
    for (let oc = 0; oc < OC; oc++) {
      const bias = b ? b.d[oc] : 0;
      for (let oy = 0; oy < OH; oy++) {
        for (let ox = 0; ox < OW; ox++) {
          let s = bias;
          for (let ic = 0; ic < IC; ic++) {
            for (let ky = 0; ky < KH; ky++) {
              const iy = oy * stride - pad + ky;
              if (iy < 0 || iy >= h) continue;
              const wo = ((oc * IC + ic) * KH + ky) * KW, xo = (ic * h + iy) * w;
              for (let kx = 0; kx < KW; kx++) {
                const ix = ox * stride - pad + kx;
                if (ix >= 0 && ix < w) s += wd[wo + kx] * xd[xo + ix];
              }
            }
          }
          out[(oc * OH + oy) * OW + ox] = s;
        }
      }
    }
    return { d: out, c: OC, h: OH, w: OW };
  }

  /** 转置卷积（「反卷积」，把特征图放大）：W [输入通道, 输出通道, kh, kw]，输出边长 (h−1)·stride − 2·pad + k */
  function convT2d(x, W, b, stride, pad) {
    const [IC, OC, KH, KW] = W.shape, { h, w } = x, wd = W.d, xd = x.d;
    const OH = (h - 1) * stride - 2 * pad + KH, OW = (w - 1) * stride - 2 * pad + KW;
    const out = new Float32Array(OC * OH * OW);
    if (b) for (let oc = 0; oc < OC; oc++) out.fill(b.d[oc], oc * OH * OW, (oc + 1) * OH * OW);
    for (let ic = 0; ic < IC; ic++) {
      for (let iy = 0; iy < h; iy++) {
        for (let ix = 0; ix < w; ix++) {
          const v = xd[(ic * h + iy) * w + ix];
          if (!v) continue;
          for (let oc = 0; oc < OC; oc++) {
            for (let ky = 0; ky < KH; ky++) {
              const oy = iy * stride - pad + ky;
              if (oy < 0 || oy >= OH) continue;
              const wo = ((ic * OC + oc) * KH + ky) * KW, oo = (oc * OH + oy) * OW;
              for (let kx = 0; kx < KW; kx++) {
                const ox = ix * stride - pad + kx;
                if (ox >= 0 && ox < OW) out[oo + ox] += v * wd[wo + kx];
              }
            }
          }
        }
      }
    }
    return { d: out, c: OC, h: OH, w: OW };
  }

  /** 最大池化（k×k 窗口，步幅 s） */
  function maxpool(x, k = 2, s = k) {
    const { c, h, w, d } = x, OH = Math.floor((h - k) / s) + 1, OW = Math.floor((w - k) / s) + 1;
    const out = new Float32Array(c * OH * OW);
    for (let ch = 0; ch < c; ch++) {
      for (let oy = 0; oy < OH; oy++) {
        for (let ox = 0; ox < OW; ox++) {
          let m = -Infinity;
          for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++) m = Math.max(m, d[(ch * h + oy * s + ky) * w + ox * s + kx]);
          out[(ch * OH + oy) * OW + ox] = m;
        }
      }
    }
    return { d: out, c, h: OH, w: OW };
  }
  /** 全局平均池化：每个通道取平均 → 长度为 C 的向量 */
  function gap(x) {
    const { c, h, w, d } = x, out = new Float32Array(c), n = h * w;
    for (let ch = 0; ch < c; ch++) { let s = 0; for (let i = 0; i < n; i++) s += d[ch * n + i]; out[ch] = s / n; }
    return out;
  }
  /** BatchNorm（推理模式，用训练时统计的均值和方差）：W 是权重表，name 是层名前缀，原地修改 */
  function bn2d(x, W, name, eps = 1e-5) {
    const g = W[name + '.weight'].d, b = W[name + '.bias'].d, mu = W[name + '.running_mean'].d, va = W[name + '.running_var'].d;
    const n = x.h * x.w;
    for (let ch = 0; ch < x.c; ch++) {
      const k = g[ch] / Math.sqrt(va[ch] + eps), off = b[ch] - mu[ch] * k;
      for (let i = ch * n; i < (ch + 1) * n; i++) x.d[i] = x.d[i] * k + off;
    }
    return x;
  }
  function add(a, b) { for (let i = 0; i < a.length; i++) a[i] += b[i]; return a; }   // 原地 a += b
  /** LayerNorm：对一个向量做标准化再缩放平移 */
  function layernorm(v, g, b, eps = 1e-5) {
    let m = 0, s = 0;
    for (const x of v) m += x;
    m /= v.length;
    for (const x of v) s += (x - m) ** 2;
    const r = 1 / Math.sqrt(s / v.length + eps), out = new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) out[i] = (v[i] - m) * r * g.d[i] + b.d[i];
    return out;
  }
  /** Embedding：取矩阵 E 的第 i 行 */
  const embed = (E, i) => E.d.slice(i * E.shape[1], (i + 1) * E.shape[1]);

  /** PyTorch nn.RNN（tanh）走一步：h' = tanh(W_ih·x + b_ih + W_hh·h + b_hh)；pre 是 'rnn.' 之类的名字前缀，层号 0 */
  function rnnStep(W, pre, x, h, layer = 0) {
    const s = `_l${layer}`;
    return tanhIn(add(linear(x, W[pre + 'weight_ih' + s], W[pre + 'bias_ih' + s]), linear(h, W[pre + 'weight_hh' + s], W[pre + 'bias_hh' + s])));
  }
  /** PyTorch nn.LSTM 走一步（门的排列顺序 i, f, g, o）：返回新的 h、c 以及各门的值（供可视化） */
  function lstmStep(W, pre, x, h, c, layer = 0) {
    const s = `_l${layer}`;
    const z = add(linear(x, W[pre + 'weight_ih' + s], W[pre + 'bias_ih' + s]), linear(h, W[pre + 'weight_hh' + s], W[pre + 'bias_hh' + s]));
    const H = h.length, gi = new Float32Array(H), gf = new Float32Array(H), gg = new Float32Array(H), go = new Float32Array(H);
    const c2 = new Float32Array(H), h2 = new Float32Array(H);
    for (let k = 0; k < H; k++) {
      gi[k] = sigmoid(z[k]); gf[k] = sigmoid(z[H + k]); gg[k] = Math.tanh(z[2 * H + k]); go[k] = sigmoid(z[3 * H + k]);
      c2[k] = gf[k] * c[k] + gi[k] * gg[k];
      h2[k] = go[k] * Math.tanh(c2[k]);
    }
    return { h: h2, c: c2, i: gi, f: gf, g: gg, o: go };
  }

  /**
   * 多头注意力。q：长度 Tq 的向量数组；kv：长度 Tk 的向量数组；pre：层名前缀（含 q/k/v/o 四个 Linear）；
   * mask(i, j) 返回 true 表示第 i 个查询不能看第 j 个位置（如解码器不能偷看未来）。
   * 返回 { out：Tq 个向量, att：[头][Tq][Tk] 注意力权重 }
   */
  function mha(W, pre, q, kv, heads, mask) {
    const Q = q.map((v) => linear(v, W[pre + 'q.weight'], W[pre + 'q.bias']));
    const K = kv.map((v) => linear(v, W[pre + 'k.weight'], W[pre + 'k.bias']));
    const V = kv.map((v) => linear(v, W[pre + 'v.weight'], W[pre + 'v.bias']));
    const d = Q[0].length, dh = d / heads, scale = 1 / Math.sqrt(dh);
    const mixed = q.map(() => new Float32Array(d)), att = [];
    for (let hh = 0; hh < heads; hh++) {
      const A = [];
      for (let i = 0; i < Q.length; i++) {
        const s = new Float32Array(K.length);
        for (let j = 0; j < K.length; j++) {
          if (mask && mask(i, j)) { s[j] = -Infinity; continue; }
          let t = 0;
          for (let e = hh * dh; e < (hh + 1) * dh; e++) t += Q[i][e] * K[j][e];
          s[j] = t * scale;
        }
        const a = softmax(s);
        A.push(Array.from(a));
        for (let j = 0; j < K.length; j++) {
          if (!a[j]) continue;
          for (let e = hh * dh; e < (hh + 1) * dh; e++) mixed[i][e] += a[j] * V[j][e];
        }
      }
      att.push(A);
    }
    return { out: mixed.map((v) => linear(v, W[pre + 'o.weight'], W[pre + 'o.bias'])), att };
  }

  /* ---------------- ③ 经典案例模型（键 = pytorch/ 下的脚本名） ----------------
   * 每个模型提供 golden(W, ...输入张量) → [输出数组, ...]，与脚本里 site_export['golden'] 的计算一一对应；
   * 其余方法（predict 等）供互动组件使用。 */
  const models = {};
  /** 对一批图片逐张调用 forward，把 key 指定的输出拼成一个数组（黄金样例用） */
  const each = (m, W, X, key, per) => {
    const out = [];
    for (let i = 0; i < X.shape[0]; i++) out.push(...m.forward(W, X.d.subarray(i * per, (i + 1) * per))[key]);
    return out;
  };

  /** 第 5 课：多层感知机 784 → 128 → 64 → 10。输入是 784 个 0~1 的像素（标准化写在模型里） */
  models['05_mlp_mnist'] = {
    forward(W, px) {
      const x = Float32Array.from(px, (v) => (v - 0.1307) / 0.3081);
      const h1 = relu(linear(x, W['fc1.weight'], W['fc1.bias']));
      const h2 = relu(linear(h1, W['fc2.weight'], W['fc2.bias']));
      const logits = linear(h2, W['fc3.weight'], W['fc3.bias']);
      return { logits, probs: softmax(logits), h1, h2 };
    },
    golden(W, X) { return [each(this, W, X, 'logits', 784)]; },
  };

  /** 第 7 课：LeNet-5。conv(5×5, 6) → ReLU → 池化 → conv(5×5, 16) → ReLU → 池化 → 400 → 120 → 84 → 10 */
  models['07_lenet_mnist'] = {
    forward(W, px) {
      const x = { d: Float32Array.from(px, (v) => (v - 0.1307) / 0.3081), c: 1, h: 28, w: 28 };
      const c1 = conv2d(x, W['conv1.weight'], W['conv1.bias'], 1, 2); relu(c1.d);
      const p1 = maxpool(c1, 2);
      const c2 = conv2d(p1, W['conv2.weight'], W['conv2.bias'], 1, 0); relu(c2.d);
      const p2 = maxpool(c2, 2);
      const f1 = relu(linear(p2.d, W['fc1.weight'], W['fc1.bias']));
      const f2 = relu(linear(f1, W['fc2.weight'], W['fc2.bias']));
      const logits = linear(f2, W['fc3.weight'], W['fc3.bias']);
      return { logits, probs: softmax(logits), c1, p1, c2, p2 };
    },
    golden(W, X) { return [each(this, W, X, 'logits', 784)]; },
  };

  /** 第 8 课：CIFAR 版 ResNet-20。输入 3×32×32、0~1 的像素（CHW），按通道标准化写在模型里 */
  models['08_resnet_cifar10'] = {
    MEAN: [0.4914, 0.4822, 0.4465], STD: [0.2470, 0.2435, 0.2616],
    forward(W, chw) {
      let x = { d: Float32Array.from(chw, (v, i) => (v - this.MEAN[i >> 10]) / this.STD[i >> 10]), c: 3, h: 32, w: 32 };
      x = bn2d(conv2d(x, W['conv.weight'], null, 1, 1), W, 'bn'); relu(x.d);
      for (let i = 0; W[`layers.${i}.conv1.weight`]; i++) {
        const p = `layers.${i}.`, down = !!W[p + 'short.0.weight'];
        const o = bn2d(conv2d(x, W[p + 'conv1.weight'], null, down ? 2 : 1, 1), W, p + 'bn1'); relu(o.d);
        const o2 = bn2d(conv2d(o, W[p + 'conv2.weight'], null, 1, 1), W, p + 'bn2');
        const sc = down ? bn2d(conv2d(x, W[p + 'short.0.weight'], null, 2, 0), W, p + 'short.1') : x;   // 捷径：恒等或 1×1 卷积
        add(o2.d, sc.d); relu(o2.d);   // 残差连接：F(x) + x，再 ReLU
        x = o2;
      }
      const logits = linear(gap(x), W['fc.weight'], W['fc.bias']);
      return { logits, probs: softmax(logits) };
    },
    golden(W, X) { return [each(this, W, X, 'logits', 3072)]; },
  };

  /** 第 9、10 课的写诗模型：字表 = 3 个特殊记号 + 2500 个常用字 */
  const vocabOf = (py) => [...py.extras.specials, ...Array.from(py.extras.vocab)];
  /** 逐字跑完一个序列，收集每个时刻的隐状态和最后的得分（黄金样例用） */
  const runSeq = (m, W, ids) => {
    let st = m.init(), hs = [];
    for (const id of ids) { st = m.step(W, Math.round(id), st); hs.push(...st.h); }
    return [hs, st.logits];
  };

  /** 第 9 课：字符级 RNN。每一步：嵌入 → h = tanh(W_ih·x + W_hh·h + b) → 得分（输出层与嵌入共享权重） */
  models['09_rnn_poem'] = {
    init: () => ({ h: new Float32Array(128) }),
    step(W, id, st) {
      const h = rnnStep(W, 'rnn.', embed(W['emb.weight'], id), st.h);
      return { h, logits: linear(h, W['out.weight'], W['out.bias']) };
    },
    golden(W, ids) { return runSeq(this, W, ids.d); },
  };

  /** 第 10 课：字符级 LSTM。状态是 (h, c)；step 同时返回四个门的值，供可视化 */
  models['10_lstm_poem'] = {
    init: () => ({ h: new Float32Array(128), c: new Float32Array(128) }),
    step(W, id, st) {
      const r = lstmStep(W, 'rnn.', embed(W['emb.weight'], id), st.h, st.c);
      return { ...r, logits: linear(r.h, W['out.weight'], W['out.bias']) };
    },
    golden(W, ids) { return runSeq(this, W, ids.d); },
  };

  /** 第 11 课：迷你 Transformer（Pre-LN，2 层编码器 + 2 层解码器，d = 64，4 个头，正弦位置编码） */
  models['11_transformer_dates'] = {
    D: 64, H: 4,
    pe(pos) {
      const d = this.D, v = new Float32Array(d);
      for (let i = 0; i < d; i += 2) { const div = Math.exp(i * (-Math.log(10000) / d)); v[i] = Math.sin(pos * div); v[i + 1] = Math.cos(pos * div); }
      return v;
    },
    embed(W, ids) { return ids.map((id, p) => { const e = embed(W['emb.weight'], id), pe = this.pe(p); for (let k = 0; k < this.D; k++) e[k] = e[k] * Math.sqrt(this.D) + pe[k]; return e; }); },
    ln(W, name, v) { return layernorm(v, W[name + '.weight'], W[name + '.bias']); },
    ffn(W, p, v) { return linear(relu(linear(v, W[p + 'ff.0.weight'], W[p + 'ff.0.bias'])), W[p + 'ff.2.weight'], W[p + 'ff.2.bias']); },
    /** 编码器：返回 memory（最后的 LayerNorm 之后）和每层自注意力权重 [层][头][Ts][Ts] */
    encode(W, src) {
      let xs = this.embed(W, src);
      const atts = [];
      for (let l = 0; W[`enc.${l}.ln1.weight`]; l++) {
        const p = `enc.${l}.`, ys = xs.map((v) => this.ln(W, p + 'ln1', v)), a = mha(W, p + 'att.', ys, ys, this.H);
        xs = xs.map((v, i) => add(Float32Array.from(v), a.out[i]));
        xs = xs.map((v) => add(v, this.ffn(W, p, this.ln(W, p + 'ln2', v))));
        atts.push(a.att);
      }
      return { mem: xs.map((v) => this.ln(W, 'enc_ln', v)), atts };
    },
    /** 解码器（一次算完整个前缀）：返回每个位置的得分、带掩码自注意力与交叉注意力 */
    decode(W, tgtIn, mem) {
      let xs = this.embed(W, tgtIn);
      const selfA = [], cross = [];
      for (let l = 0; W[`dec.${l}.ln1.weight`]; l++) {
        const p = `dec.${l}.`, ys = xs.map((v) => this.ln(W, p + 'ln1', v));
        const a = mha(W, p + 'self_att.', ys, ys, this.H, (i, j) => j > i);   // 不能看后面的位置
        xs = xs.map((v, i) => add(Float32Array.from(v), a.out[i]));
        const c = mha(W, p + 'cross_att.', xs.map((v) => this.ln(W, p + 'ln2', v)), mem, this.H);
        xs = xs.map((v, i) => add(v, c.out[i]));
        xs = xs.map((v) => add(v, this.ffn(W, p, this.ln(W, p + 'ln3', v))));
        selfA.push(a.att); cross.push(c.att);
      }
      return { logits: xs.map((v) => linear(this.ln(W, 'dec_ln', v), W['out.weight'], W['out.bias'])), self: selfA, cross };
    },
    /** 贪心解码 10 步：从 <s>（编号 1）开始，每步取概率最大的字符 */
    translate(W, src) {
      const { mem, atts } = this.encode(W, src), out = [1];
      let r;
      for (let k = 0; k < 10; k++) { r = this.decode(W, out, mem); out.push(argmax(r.logits[r.logits.length - 1])); }
      return { ids: out.slice(1), enc: atts, cross: r.cross, self: r.self, probs: r.logits.map((l) => softmax(l)) };
    },
    golden(W, src, tgt) { return [this.decode(W, Array.from(tgt.d, Math.round), this.encode(W, Array.from(src.d, Math.round)).mem).logits.flatMap((v) => Array.from(v))]; },
  };

  /** 第 12 课：VAE 的编码器（784 → 128 → 2，取均值 mu）和解码器（2 → 128 → 784）；普通 AE 只导出了解码器（前缀 ae_dec.） */
  models['12_vae_mnist'] = {
    encode(W, x) { return linear(relu(linear(x, W['vae.enc.fc.weight'], W['vae.enc.fc.bias'])), W['vae.enc.mu.weight'], W['vae.enc.mu.bias']); },
    decode(W, z, pre = 'vae.dec.') { return sigmoidIn(linear(relu(linear(z, W[pre + 'fc1.weight'], W[pre + 'fc1.bias'])), W[pre + 'fc2.weight'], W[pre + 'fc2.bias'])); },
    golden(W, z, x) {
      const dv = [], mu = [], da = [];
      for (let i = 0; i < z.shape[0]; i++) { const zi = z.d.subarray(2 * i, 2 * i + 2); dv.push(...this.decode(W, zi)); da.push(...this.decode(W, zi, 'ae_dec.')); }
      for (let i = 0; i < x.shape[0]; i++) mu.push(...this.encode(W, x.d.subarray(784 * i, 784 * (i + 1))));
      return [dv, mu, da];
    },
  };

  /** 第 13 课：DCGAN 生成器。噪声 32 → 全连接 → BN → ReLU → 64×7×7 → 转置卷积 → 32×14×14 → 转置卷积 → 1×28×28 → tanh */
  models['13_gan_mnist'] = {
    forward(W, z) {
      const h = linear(z, W['fc.weight'], W['fc.bias']);
      const g = W['bn0.weight'].d, b = W['bn0.bias'].d, m = W['bn0.running_mean'].d, v = W['bn0.running_var'].d;
      for (let k = 0; k < h.length; k++) h[k] = Math.max(0, ((h[k] - m[k]) / Math.sqrt(v[k] + 1e-5)) * g[k] + b[k]);   // BatchNorm1d（推理模式）+ ReLU
      const x = bn2d(convT2d({ d: h, c: 64, h: 7, w: 7 }, W['up1.weight'], W['up1.bias'], 2, 1), W, 'bn1');
      relu(x.d);
      return tanhIn(convT2d(x, W['up2.weight'], W['up2.bias'], 2, 1).d);   // 784 个 −1~1 的像素
    },
    golden(W, Z) { const out = []; for (let i = 0; i < Z.shape[0]; i++) out.push(...this.forward(W, Z.d.subarray(i * 32, (i + 1) * 32))); return [out]; },
  };

  /** 自检：用 JS 前向重算 PyTorch 导出的黄金样例，返回 { err：最大相对误差, n：比较的数值个数 }；没有黄金样例时返回 null */
  function goldenCheck(py) {
    const m = models[py.name];
    if (!py.golden) return null;
    if (!m) throw new Error(`js/nn.js 里没有模型 models['${py.name}']`);
    const W = weights(py);
    const ins = py.golden.in.map(decode), outs = py.golden.out.map(decode);
    const got = m.golden(W, ...ins);
    let err = 0, n = 0;
    outs.forEach((o, k) => {
      const g = got[k];
      if (!g || g.length !== o.d.length) throw new Error(`第 ${k + 1} 个输出长度不一致：JS ${g?.length}，PyTorch ${o.d.length}`);
      for (let i = 0; i < g.length; i++) { err = Math.max(err, Math.abs(g[i] - o.d[i]) / (1 + Math.abs(o.d[i]))); n++; }
    });
    return { err, n };
  }

  /* ---------------- ④ 可训练的小 MLP（浏览器内实时训练） ---------------- */
  /** 可复现的随机数：mulberry32 + Box–Muller 正态分布 */
  function rng(seed = 1) {
    let s = seed >>> 0, spare = null;
    const next = () => {
      s = (s + 0x6d2b79f5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    return {
      next,
      uniform: (a = 0, b = 1) => a + (b - a) * next(),
      int: (n) => Math.floor(next() * n),
      randn() {
        if (spare !== null) { const v = spare; spare = null; return v; }
        let u = 0;
        while (!u) u = next();
        const v = next(), r = Math.sqrt(-2 * Math.log(u));
        spare = r * Math.sin(2 * Math.PI * v);
        return r * Math.cos(2 * Math.PI * v);
      },
    };
  }

  const ACT = {
    linear: { f: (z) => z, df: () => 1 },
    tanh: { f: Math.tanh, df: (z, a) => 1 - a * a },
    relu: { f: (z) => (z > 0 ? z : 0), df: (z) => (z > 0 ? 1 : 0) },
    sigmoid: { f: sigmoid, df: (z, a) => a * (1 - a) },
  };

  /**
   * 小型全连接网络，逐个样本前向 / 反向，梯度累加后由 step() 统一更新。
   * sizes：每层神经元数，如 [2, 8, 8, 1]；act：隐藏层激活；out：输出层激活。
   * 用法：net.forward(x) → 输出；net.backward(dL/d输出) → 返回 dL/d输入；累积若干样本后 net.step({ lr, opt })。
   */
  class MLP {
    constructor(sizes, { act = 'tanh', out = 'linear', seed = 1 } = {}) {
      this.sizes = sizes;
      this.act = act;
      this.out = out;
      const r = rng(seed);
      this.W = [];
      this.b = [];
      for (let l = 0; l < sizes.length - 1; l++) {
        const fi = sizes[l], fo = sizes[l + 1];
        const std = act === 'relu' ? Math.sqrt(2 / fi) : Math.sqrt(1 / fi);   // He / LeCun 初始化
        this.W.push(Float64Array.from({ length: fi * fo }, () => r.randn() * std));
        this.b.push(new Float64Array(fo));
      }
      this.gW = this.W.map((w) => new Float64Array(w.length));
      this.gb = this.b.map((b) => new Float64Array(b.length));
      this.n = 0;
      this.t = 0;
    }
    forward(x) {
      this.zs = [];
      this.as = [Float64Array.from(x)];
      const L = this.W.length;
      for (let l = 0; l < L; l++) {
        const fi = this.sizes[l], fo = this.sizes[l + 1], W = this.W[l], b = this.b[l], a = this.as[l];
        const f = ACT[l === L - 1 ? this.out : this.act].f;
        const z = new Float64Array(fo), y = new Float64Array(fo);
        for (let o = 0; o < fo; o++) {
          let s = b[o];
          for (let i = 0; i < fi; i++) s += W[o * fi + i] * a[i];
          z[o] = s;
          y[o] = f(s);
        }
        this.zs.push(z);
        this.as.push(y);
      }
      return this.as[L];
    }
    backward(dOut) {
      const L = this.W.length;
      let d = Float64Array.from(dOut);
      for (let l = L - 1; l >= 0; l--) {
        const fi = this.sizes[l], fo = this.sizes[l + 1], W = this.W[l], z = this.zs[l], a = this.as[l + 1], prev = this.as[l];
        const df = ACT[l === L - 1 ? this.out : this.act].df;
        for (let o = 0; o < fo; o++) d[o] *= df(z[o], a[o]);
        const gW = this.gW[l], gb = this.gb[l], dp = new Float64Array(fi);
        for (let o = 0; o < fo; o++) {
          gb[o] += d[o];
          for (let i = 0; i < fi; i++) { gW[o * fi + i] += d[o] * prev[i]; dp[i] += W[o * fi + i] * d[o]; }
        }
        d = dp;
      }
      this.n++;
      return d;
    }
    zeroGrad() { this.gW.forEach((g) => g.fill(0)); this.gb.forEach((g) => g.fill(0)); this.n = 0; }
    /** 用累积的平均梯度更新一次参数。opt：'sgd' | 'momentum' | 'adam'；l2：权重衰减系数 */
    step({ lr = 0.03, opt = 'sgd', l2 = 0, beta1 = 0.9, beta2 = 0.999 } = {}) {
      if (!this.n) return;
      this.t++;
      if (!this.m) { this.m = [...this.W, ...this.b].map((p) => new Float64Array(p.length)); this.v = this.m.map((p) => new Float64Array(p.length)); }
      const params = [...this.W, ...this.b], grads = [...this.gW, ...this.gb], nW = this.W.length;
      params.forEach((p, k) => {
        const g = grads[k], m = this.m[k], v = this.v[k];
        for (let i = 0; i < p.length; i++) {
          const gi = g[i] / this.n + (k < nW ? l2 * p[i] : 0);
          if (opt === 'adam') {
            m[i] = beta1 * m[i] + (1 - beta1) * gi;
            v[i] = beta2 * v[i] + (1 - beta2) * gi * gi;
            p[i] -= (lr * (m[i] / (1 - beta1 ** this.t))) / (Math.sqrt(v[i] / (1 - beta2 ** this.t)) + 1e-8);
          } else if (opt === 'momentum') {
            m[i] = beta1 * m[i] + gi;
            p[i] -= lr * m[i];
          } else p[i] -= lr * gi;
        }
      });
      this.zeroGrad();
    }
  }

  /** 梯度检验：反向传播算出的梯度 vs 数值差分，返回最大相对误差（应远小于 1e-5） */
  function gradCheck() {
    let worst = 0;
    for (const [act, out] of [['tanh', 'linear'], ['relu', 'sigmoid']]) {
      const net = new MLP([3, 5, 4, 2], { act, out, seed: 7 });
      const x = [0.3, -0.8, 0.5], y = [0.2, 0.7];
      const loss = () => { const o = net.forward(x); return 0.5 * ((o[0] - y[0]) ** 2 + (o[1] - y[1]) ** 2); };
      const o = net.forward(x);
      net.backward([o[0] - y[0], o[1] - y[1]]);
      for (const [P, G] of [[net.W, net.gW], [net.b, net.gb]]) {
        P.forEach((p, l) => {
          for (let k = 0; k < p.length; k++) {
            const old = p[k];
            p[k] = old + 1e-5; const lp = loss();
            p[k] = old - 1e-5; const lm = loss();
            p[k] = old;
            const num = (lp - lm) / 2e-5, ana = G[l][k];
            if (Math.abs(num) + Math.abs(ana) > 1e-9) worst = Math.max(worst, Math.abs(num - ana) / (Math.abs(num) + Math.abs(ana)));
          }
        });
      }
    }
    return worst;
  }

  /** 动画循环：每帧调用 fn()；组件被移出页面（切换路由）后自动停止。返回 { start(), stop(), running } */
  function loop(el, fn) {
    let id = null;
    const tick = () => {
      if (!el.isConnected) { id = null; return; }
      fn();
      if (id !== null) id = requestAnimationFrame(tick);
    };
    return {
      start() { if (id === null) id = requestAnimationFrame(tick); },
      stop() { if (id !== null) cancelAnimationFrame(id); id = null; },
      get running() { return id !== null; },
    };
  }

  window.NN = {
    decode, weights, linear, relu, tanhIn, sigmoidIn, sigmoid, softmax, argmax, topk, conv2d, convT2d, maxpool, gap, bn2d, add,
    layernorm, embed, rnnStep, lstmStep, mha, models, goldenCheck, rng, MLP, gradCheck, loop, vocabOf,
  };
})();
