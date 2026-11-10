/* =====================================================================
 * 站点引擎：启动加载、路由、课程页（4 步）、测验 / 随堂小问 / 情景模拟、
 * 经典案例（脚本代码 + 真实输出）、排错手册、学习进度。
 * 日常维护改 content/（内容）和 config.js（配置）即可，一般不需要动这个文件。
 * ===================================================================== */
(function () {
  'use strict';

  /* ---------------- 小工具 ---------------- */
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
  const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;
  const mdHtml = (src) => (src ? marked.parse(String(src)) : '');
  const mdInline = (src) => marked.parseInline(String(src ?? ''));
  const debounce = (fn, ms = 300) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
  const fmtSecs = (s) => (s < 60 ? `${s.toFixed(s < 10 ? 1 : 0)} 秒` : `${Math.floor(s / 60)} 分 ${Math.round(s % 60)} 秒`);

  /** md`...`：去掉模板字符串的公共缩进，方便在 JS 里排版 Markdown。行内代码的反引号写成 \` */
  window.md = (strings, ...vals) => {
    const s = strings.reduce((acc, str, i) => acc + (str ?? strings.raw[i]) + (i < vals.length ? vals[i] : ''), '');
    const lines = s.replace(/^\s*\n/, '').replace(/\s+$/, '').split('\n');
    const indent = Math.min(...lines.filter((l) => l.trim()).map((l) => l.match(/^ */)[0].length));
    return lines.map((l) => l.slice(indent)).join('\n');
  };

  /** 动态加载脚本（同一地址只加载一次；用 <script> 而不是 fetch，双击打开 index.html 时也能用） */
  const scripts = {};
  function loadScript(src) {
    return scripts[src] || (scripts[src] = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = src;
      s.onload = resolve;
      s.onerror = () => { delete scripts[src]; s.remove(); reject(new Error(`无法加载 ${src}`)); };
      document.head.appendChild(s);
    }));
  }

  /* ---------------- 注册表 ---------------- */
  /** 课程：content/lessons/<id>.js 里调用 Content.addLesson({...}) */
  const Content = window.Content = { lessons: {}, addLesson(l) { this.lessons[l.id] = l; } };
  /** 互动组件：js/widgets/<id>.js 里写 Widgets['<id>'] = { title, render(el, opts) {...} } */
  window.Widgets = {};
  /** 脚本运行结果：tools/run.py 生成的 assets/py/<名字>.js 写入 PY['<名字>'] */
  window.PY = {};

  async function loadPy(name) {
    if (!PY[name]) await loadScript(`assets/py/${name}.js`).catch(() => {});
    if (!PY[name]) throw new Error(`没有找到 assets/py/${name}.js —— 请先运行 python tools/run.py ${name}`);
    return PY[name];
  }
  /** 占位符 {{key}} → 脚本 summary 里的真实数字（重新训练后自动更新） */
  const fill = (s, py) => (py?.summary ? String(s).replace(/\{\{(\w+)\}\}/g, (m, k) => (k in py.summary ? String(py.summary[k]) : m)) : s);

  /* ---------------- 本地进度（localStorage，失败时静默降级） ---------------- */
  const Store = {
    data: {},
    load() { try { this.data = JSON.parse(localStorage.getItem(SITE.storageKey)) || {}; } catch (e) { this.data = {}; } },
    get(k, d) { return k in this.data ? this.data[k] : d; },
    set(k, v) { this.data[k] = v; try { localStorage.setItem(SITE.storageKey, JSON.stringify(this.data)); } catch (e) { /* 隐私模式等无法保存时忽略 */ } },
  };
  const Progress = {
    of: (id) => Store.get('l:' + id, { steps: [], quiz: 0 }),
    markStep(id, n) { const p = Progress.of(id); if (!p.steps.includes(n)) { p.steps.push(n); Store.set('l:' + id, p); } },
    setQuiz(id, score) {
      const p = Progress.of(id);
      p.quiz = Math.max(p.quiz || 0, score);
      if (score >= SITE.quizPass && !p.steps.includes(4)) p.steps.push(4);
      Store.set('l:' + id, p);
    },
    done: (id) => (Progress.of(id).quiz || 0) >= SITE.quizPass,
  };

  const lessonList = () => SITE.lessons.map((id) => Content.lessons[id]).filter(Boolean);
  const STEPS = ['视频导学', '原理精讲', '经典案例', '闯关练习'];
  const stagesOf = (list) => {
    const stages = [];
    for (const l of list) {
      let s = stages.find((x) => x.name === l.stage);
      if (!s) stages.push((s = { name: l.stage, lessons: [] }));
      s.lessons.push(l);
    }
    return stages;
  };

  /* ---------------- 通用片段 ---------------- */
  const setTitle = (t) => { document.title = t ? `${t} · ${SITE.name}` : `${SITE.name} · 深度学习交互学习站`; };
  const bar = (done, total) => `<div class="card-foot"><div class="progress"><span style="width:${total ? (done / total) * 100 : 0}%"></span></div><span>${done}/${total}</span></div>`;

  function videoHtml(v) {
    const p = v.p || 1;
    return `<figure class="video">
      <div class="frame"><iframe src="https://player.bilibili.com/player.html?bvid=${esc(v.bvid)}&p=${p}&autoplay=0&danmaku=0" title="${esc(v.title)}" loading="lazy" allowfullscreen allow="fullscreen; picture-in-picture"></iframe></div>
      <figcaption><b>${esc(v.title)}</b>${v.note ? ` — ${esc(v.note)}` : ''}<br>
        <span class="muted">UP 主：${esc(v.up)}${v.minutes ? ` · 约 ${v.minutes} 分钟` : ''} · <a href="https://www.bilibili.com/video/${esc(v.bvid)}/?p=${p}" target="_blank" rel="noopener">在 B 站打开</a></span></figcaption>
    </figure>`;
  }

  /** 题干：多行（含代码块）按块级 Markdown 渲染，单行按行内渲染 */
  const qHtml = (s) => (String(s).includes('\n') ? mdHtml(s) : mdInline(s));
  const isNum = (q) => !q.options;   // 没有 options 的题是计算填空，answer 是数字
  const numOk = (q, v) => Number.isFinite(v) && Math.abs(v - q.answer) <= (q.tol ?? 1e-6 + Math.abs(q.answer) * 0.005);
  const parseNum = (s) => Number(String(s).trim().replace(/[，,]/g, '').replace('−', '-'));

  /** 闯关练习：qs = [{ q, options, answer(下标/下标数组), explain }] 或计算填空 [{ q, answer: 数字, tol?, unit?, explain }] */
  function renderQuiz(el, qs, onSubmit) {
    el.innerHTML = qs.map((q, i) => {
      const multi = Array.isArray(q.answer);
      const tag = isNum(q) ? '填空' : multi ? '多选' : '单选';
      const body = isNum(q)
        ? `<label class="num-ans">答案：<input type="text" inputmode="decimal" autocomplete="off" aria-label="第 ${i + 1} 题答案">${q.unit ? ` <span>${esc(q.unit)}</span>` : ''}</label>`
        : `<div class="opts">${q.options.map((o, j) => `<label class="opt"><input type="${multi ? 'checkbox' : 'radio'}" name="q${i}" value="${j}"><span>${mdInline(o)}</span></label>`).join('')}</div>`;
      return `<div class="q" data-i="${i}">
        <div class="q-title"><span class="tag">${tag}</span>${i + 1}. ${qHtml(q.q)}</div>${body}
        <div class="explain" hidden></div>
      </div>`;
    }).join('') + '<div class="row"><button class="btn" type="button">提交答案</button><span class="score" aria-live="polite"></span></div>';
    const btn = $('.row > .btn', el), scoreEl = $('.score', el);
    btn.addEventListener('click', () => {
      if (btn.dataset.state === 'done') { renderQuiz(el, qs, onSubmit); return; }
      const boxes = $$('.q', el);
      const unanswered = boxes.filter((b, i) => (isNum(qs[i]) ? !$('input', b).value.trim() : !$('input:checked', b))).length;
      if (unanswered) { scoreEl.textContent = `还有 ${unanswered} 题没有作答`; return; }
      let right = 0;
      qs.forEach((q, i) => {
        const box = boxes[i];
        let ok;
        if (isNum(q)) {
          const inp = $('input', box);
          ok = numOk(q, parseNum(inp.value));
          inp.disabled = true;
          inp.parentElement.classList.add(ok ? 'correct' : 'wrong');
        } else {
          const picked = $$('input:checked', box).map((x) => +x.value).sort();
          const ans = [].concat(q.answer).sort();
          ok = picked.length === ans.length && picked.every((v, k) => v === ans[k]);
          $$('.opt', box).forEach((lab, j) => {
            lab.classList.toggle('correct', ans.includes(j));
            lab.classList.toggle('wrong', picked.includes(j) && !ans.includes(j));
            $('input', lab).disabled = true;
          });
        }
        if (ok) right++;
        const ex = $('.explain', box);
        ex.hidden = false;
        ex.className = `explain ${ok ? 'ok' : 'err'}`;
        ex.innerHTML = `<p><b>${ok ? '✓ 回答正确' : '✗ 回答错误'}</b>${isNum(q) ? `　正确答案：${q.answer}${q.unit ? ' ' + esc(q.unit) : ''}` : ''}</p>${mdHtml(q.explain)}`;
      });
      const score = right / qs.length;
      scoreEl.textContent = `得分 ${right} / ${qs.length}（${pct(score)}）${score >= SITE.quizPass ? ' 🎉 通过！' : ' 未达标，看看解析再试一次'}`;
      btn.textContent = '重新作答';
      btn.dataset.state = 'done';
      onSubmit?.(score);
    });
  }

  /** 随堂小问：答错可以继续选，答对后显示解析 */
  function renderCheck(el, q) {
    el.className = 'q check';
    el.innerHTML = `<div class="q-title"><span class="tag warn">随堂小问</span> ${qHtml(q.q)}</div>
      <div class="opts">${q.options.map((o, j) => `<button type="button" class="opt" data-j="${j}">${mdInline(o)}</button>`).join('')}</div>
      <div class="explain" hidden></div>`;
    el.addEventListener('click', (e) => {
      const b = e.target.closest('.opt');
      if (!b || el.dataset.done) return;
      const ok = [].concat(q.answer).includes(+b.dataset.j);
      b.classList.add(ok ? 'correct' : 'wrong');
      const ex = $('.explain', el);
      ex.hidden = false;
      ex.className = `explain ${ok ? 'ok' : 'err'}`;
      ex.innerHTML = ok ? `<p><b>✓ 答对了！</b></p>${mdHtml(q.explain)}` : `<p><b>✗ 再想想，</b>${esc(q.hint || '换一个选项试试')}</p>`;
      if (ok) { el.dataset.done = '1'; $$('.opt', el).forEach((o) => o.classList.add('locked')); }
    });
  }

  /** 情景模拟：sc = { title, intro, steps: [{ who, text, options: [{ label, ok, feedback }] }], end } */
  function renderScenario(el, sc) {
    el.className = 'widget scenario';
    el.innerHTML = `<h3>🎭 ${esc(sc.title)}</h3>${sc.intro ? `<div class="intro prose">${mdHtml(sc.intro)}</div>` : ''}<div class="turns"></div>`;
    const turns = $('.turns', el);
    const show = (i) => {
      if (i >= sc.steps.length) { turns.insertAdjacentHTML('beforeend', `<div class="notice ok turn">${mdHtml(sc.end || '🎉 情景完成！')}</div>`); return; }
      const st = sc.steps[i];
      const div = document.createElement('div');
      div.className = 'turn';
      div.innerHTML = `<div class="say"><span class="avatar" aria-hidden="true">${st.who || '🧑‍💻'}</span><div class="bubble">${mdHtml(st.text)}</div></div>
        <div class="opts" style="margin-top:10px">${st.options.map((o, j) => `<button type="button" class="opt" data-j="${j}">${mdInline(o.label)}</button>`).join('')}</div>
        <div class="explain" hidden></div>`;
      turns.appendChild(div);
      div.addEventListener('click', (e) => {
        const b = e.target.closest('.opt');
        if (!b || div.dataset.done) return;
        const o = st.options[+b.dataset.j];
        b.classList.add(o.ok ? 'correct' : 'wrong');
        const ex = $('.explain', div);
        ex.hidden = false;
        ex.className = `explain ${o.ok ? 'ok' : 'err'}`;
        ex.innerHTML = mdHtml(o.feedback);
        if (o.ok) { div.dataset.done = '1'; $$('.opt', div).forEach((x) => x.classList.add('locked')); show(i + 1); }
      });
    };
    show(0);
  }

  /** 互动组件：按需加载 js/widgets/<id>.js；opts 是内容里写的其余字段（以及 py：本课脚本的运行结果） */
  async function mountWidget(el, id, opts = {}) {
    el.className = 'widget';
    el.innerHTML = '<p class="muted">互动组件加载中…</p>';
    try { await loadScript(`js/widgets/${id}.js`); } catch (e) { /* 下面统一提示 */ }
    const w = Widgets[id];
    if (!w) { el.innerHTML = `<div class="notice err">互动组件「${esc(id)}」加载失败：请检查 js/widgets/${esc(id)}.js</div>`; return; }
    el.innerHTML = `${opts.title !== '' ? `<h3>🧪 ${esc(opts.title || w.title)}</h3>` : ''}${opts.intro ? `<div class="intro prose">${mdHtml(opts.intro)}</div>` : ''}<div class="widget-body"></div>`;
    try { await w.render($('.widget-body', el), opts); } catch (e) {
      console.error(e);
      $('.widget-body', el).innerHTML = `<div class="notice err">组件运行出错：${esc(e.message)}</div>`;
    }
  }

  /** Python 代码的轻量高亮：只把注释变暗（注释里是中文讲解，单独显示更好读） */
  function pyHtml(code) {
    return code.split('\n').map((line) => {
      let q = null;
      for (let i = 0; i < line.length; i++) {
        const c = line[i];
        if (q) { if (c === '\\') i++; else if (c === q) q = null; }
        else if (c === '"' || c === "'") q = c;
        else if (c === '#') return esc(line.slice(0, i)) + `<span class="cm">${esc(line.slice(i))}</span>`;
      }
      return esc(line);
    }).join('\n');
  }

  /** 逐段展示脚本：代码 + 这段代码真实运行时的输出（数据来自 tools/run.py） */
  function renderCode(el, py) {
    el.className = 'code-walk';
    if (!py) { el.innerHTML = '<div class="notice warn">本课没有配套脚本。</div>'; return; }
    const shown = py.cells.filter((c) => !c.title.startsWith('导出给网站'));
    el.innerHTML = `
      <div class="code-head"><div><b>📄 ${esc(py.file)}</b><br><span class="muted small">共 ${shown.length} 段 · ${esc(py.meta.date)} 在本机 CPU 上运行（PyTorch ${esc(py.meta.torch)}，总用时 ${fmtSecs(py.meta.secs)}）</span></div>
        <button class="btn sm ghost" type="button" data-act="dl">⬇ 下载脚本</button></div>
      <p class="small muted">点开每一段：上面是代码（注释讲了为什么这样写），下面是它真实运行时打印的输出。下载后可以直接 <code>python ${esc(py.file.split('/').pop())}</code> 运行，也能在 VS Code / PyCharm 里按 <code># %%</code> 分段运行。</p>
      ${shown.map((c, i) => `<details class="cell"${i === (shown[0]?.title === '脚本说明' ? 1 : 0) ? ' open' : ''}><summary><span class="tag">${i + 1}</span> ${esc(c.title)}${c.secs >= 1 ? ` <span class="muted small">· ${fmtSecs(c.secs)}</span>` : ''}</summary>
        <pre class="code">${pyHtml(c.code)}</pre>
        ${c.out ? `<div class="out-label small muted">▼ 运行输出</div><pre class="stdout">${esc(c.out.replace(/\s+$/, ''))}</pre>` : ''}</details>`).join('')}`;
    $('[data-act=dl]', el).addEventListener('click', () => {
      const a = Object.assign(document.createElement('a'), {
        href: URL.createObjectURL(new Blob([py.source], { type: 'text/x-python;charset=utf-8' })),
        download: py.file.split('/').pop(),
      });
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    });
  }

  /** 内容块：Markdown 字符串 / { check } / { widget, ... } / { scenario } / { code: true }（案例脚本的逐段代码） */
  function renderBlocks(container, blocks, ctx = {}) {
    for (const block of blocks || []) {
      const div = document.createElement('div');
      container.appendChild(div);
      if (typeof block === 'string') { div.className = 'prose'; div.innerHTML = mdHtml(fill(block, ctx.py)); }
      else if (block.check) renderCheck(div, block.check);
      else if (block.widget) mountWidget(div, block.widget, { ...block, intro: block.intro && fill(block.intro, ctx.py), note: block.note && fill(block.note, ctx.py), py: ctx.py });
      else if (block.scenario) renderScenario(div, block.scenario);
      else if (block.code) renderCode(div, ctx.py);
      else div.innerHTML = '<div class="notice err">无法识别的内容块（应为 Markdown 文本、check、widget、scenario 或 code）</div>';
    }
  }

  /* ---------------- 页面 ---------------- */
  function nextHref() {
    const l = lessonList().find((x) => !Progress.done(x.id));
    if (!l) return '#/cases';
    return `#/learn/${l.id}/${[1, 2, 3, 4].find((n) => !Progress.of(l.id).steps.includes(n)) || 4}`;
  }

  function pageHome(el) {
    setTitle('');
    const list = lessonList();
    const doneN = list.filter((l) => Progress.done(l.id)).length;
    const steps = list.reduce((s, l) => s + Progress.of(l.id).steps.length, 0);
    const started = steps > 0;
    const featured = list.filter((l) => l.case?.featured);
    el.innerHTML = `
      <section class="hero">
        <h1>${esc(SITE.name)}</h1>
        <p>${esc(SITE.slogan)}。从张量、梯度、感知机讲起，一路到 CNN、RNN、Transformer 和生成模型：每个概念都配一张能动手玩的图，每个模型都配一个在你浏览器里真正运行的经典案例。</p>
        <div class="row"><a class="btn" href="${nextHref()}">${started ? '继续学习' : '开始学习'} →</a><a class="btn ghost" href="#/cases">先玩玩经典案例</a></div>
      </section>
      <h2>学习路线</h2>
      <ol class="path">${stagesOf(list).map((s) => {
        const d = s.lessons.filter((l) => Progress.done(l.id)).length;
        const href = `#/learn/${(s.lessons.find((l) => !Progress.done(l.id)) || s.lessons[0]).id}`;
        return `<li><a href="${href}"><b>${esc(s.name)}</b><span class="muted">${s.lessons.map((l) => esc(l.title)).join(' · ')}</span>${bar(d, s.lessons.length)}</a></li>`;
      }).join('')}</ol>
      ${featured.length ? `<h2>经典案例：打开就能玩</h2>
      <div class="grid">${featured.map((l) => `<a class="card case-card" href="#/learn/${l.id}/3"><div class="icon" aria-hidden="true">${l.case.icon || '🧩'}</div>
        <h3>${esc(l.case.title)}</h3><p class="muted small">${esc(l.case.summary || '')}</p><span class="tag blue">${esc(l.title)}</span></a>`).join('')}</div>` : ''}
      <h2>学习进度</h2>
      <div class="tiles">
        <div class="tile"><div class="label">已完成课程</div><div class="value">${doneN} <small>/ ${list.length}</small></div></div>
        <div class="tile"><div class="label">已完成步骤</div><div class="value">${steps} <small>/ ${list.length * 4}</small></div></div>
      </div>
      <h2>怎么学</h2>
      <div class="grid">
        <div class="card"><div class="icon">🪜</div><h3>每课 4 步</h3><p class="muted">视频导学 → 原理精讲（边读边玩互动图、随堂小问）→ 经典案例 → 闯关练习，每完成一步打一个勾。</p></div>
        <div class="card"><div class="icon">👀</div><h3>先直觉，再公式</h3><p class="muted">每个概念先给直观的图和类比，再给公式（每个符号都解释），最后讲清「为什么要这样设计」。推导放在可折叠的「深入一点」里。</p></div>
        <div class="card"><div class="icon">🧠</div><h3>模型在你的浏览器里跑</h3><p class="muted">案例模型都是用 PyTorch 真实训练出来的，导出后在网页里运行：手写一个数字、拍一张照片、写一首藏头诗，立刻看到结果。</p></div>
        <div class="card"><div class="icon">📄</div><h3>代码真实、可下载</h3><p class="muted">每段 PyTorch 代码都附带它真实运行时的输出，不用自己写代码也能看懂每一步；想动手时下载脚本即可运行。</p></div>
      </div>
      <p class="small muted">学习进度只保存在当前浏览器里。电脑、手机都能打开（手机建议横屏玩互动图）。</p>`;
  }

  function pageLearn(el) {
    setTitle('课程');
    const missing = SITE.lessons.filter((id) => !Content.lessons[id]);
    el.innerHTML = `
      <h1>课程</h1>
      <p class="muted">共 ${lessonList().length} 课，建议按顺序学习。每课 4 步：视频导学 → 原理精讲 → 经典案例 → 闯关练习；闯关正确率达到 ${pct(SITE.quizPass)} 即完成本课。</p>
      ${missing.length ? `<div class="notice warn">config.js 中登记了但没有加载成功的课程文件：${esc(missing.join('、'))}。打开 <a href="#/selftest">站点自检</a> 查看原因。</div>` : ''}
      ${stagesOf(lessonList()).map((s) => `<h2>${esc(s.name)}</h2><div class="grid">${s.lessons.map((l) => {
        const p = Progress.of(l.id), i = lessonList().indexOf(l);
        const state = Progress.done(l.id) ? '<span class="tag ok">✓ 已完成</span>' : p.steps.length ? '<span class="tag blue">学习中</span>' : '';
        return `<a class="card" href="#/learn/${l.id}">
          <div class="row"><span class="icon" aria-hidden="true">${l.icon || '📘'}</span><span class="spacer"></span>${state}</div>
          <h3>${i + 1}. ${esc(l.title)}</h3>
          <p class="muted small">${esc(l.summary)}</p>
          <div class="card-foot"><span>约 ${l.minutes || 40} 分钟</span><div class="progress"><span style="width:${(p.steps.length / 4) * 100}%"></span></div><span>${p.steps.length}/4</span></div>
        </a>`;
      }).join('')}</div>`).join('')}`;
  }

  function pageLesson(el, id, stepStr) {
    const list = lessonList();
    const idx = list.findIndex((l) => l.id === id);
    const l = list[idx];
    if (!l) { pageNotFound(el); return; }
    const step = +stepStr || 1;
    const p = Progress.of(id);
    const base = `#/learn/${id}`;
    setTitle(l.title);
    el.innerHTML = `
      <div class="crumbs"><a href="#/learn">课程</a> / 第 ${idx + 1} 课 · ${esc(l.stage)}</div>
      <h1>${l.icon || ''} ${esc(l.title)}</h1>
      <p class="muted">${esc(l.summary)}</p>
      <nav class="stepper" aria-label="学习步骤">${STEPS.map((s, i) => {
        const n = i + 1, done = p.steps.includes(n);
        return `<a href="${base}/${n}" class="${n === step ? 'current' : ''} ${done ? 'done' : ''}" ${n === step ? 'aria-current="step"' : ''}><span class="num">${done ? '✓' : n}</span>${s}</a>`;
      }).join('')}</nav>
      <section class="step-body"></section>
      <nav class="step-nav"></nav>`;

    const body = $('.step-body', el);
    const next = list[idx + 1];
    const goNext = step < 4 ? `${base}/${step + 1}` : next ? `#/learn/${next.id}` : '#/cases';

    if (step === 1) {
      const kw = l.searchKeyword || l.title;
      body.innerHTML = `
        <h2>学习目标</h2><ul class="goals">${(l.goals || []).map((g) => `<li>${esc(g)}</li>`).join('')}</ul>
        <h2>视频导学</h2>
        <div class="videos">${(l.videos || []).map(videoHtml).join('')}</div>
        <p class="small muted">视频来自 B 站，版权归原作者。播放不了可以点「在 B 站打开」，或<a href="https://search.bilibili.com/all?keyword=${encodeURIComponent(kw)}" target="_blank" rel="noopener">在 B 站搜索更多「${esc(kw)}」视频</a>。不想看视频也可以直接进入下一步。</p>`;
    } else if (step === 2) {
      renderBlocks(body, l.knowledge);
    } else if (step === 3) {
      const c = l.case || {};
      body.innerHTML = `<h2>${c.icon || '🧩'} ${esc(c.title || '经典案例')}</h2>`;
      const show = (py, err) => {
        if (err) body.insertAdjacentHTML('beforeend', `<div class="notice warn">${esc(err.message)}</div>`);
        renderBlocks(body, c.blocks, { py });
        if (py && !(c.blocks || []).some((b) => b && b.code)) renderBlocks(body, [{ code: true }], { py });
        if (c.troubleshoot?.length) {
          body.insertAdjacentHTML('beforeend', `<h2>🩺 排错清单</h2><p class="muted small">自己训练时遇到问题，先对照「现象」找原因。全部课程的清单汇总在 <a href="#/handbook">排错手册</a>。</p>
            <div class="table-wrap"><table class="data trouble"><thead><tr><th>现象</th><th>可能原因</th><th>怎么办</th></tr></thead><tbody>${c.troubleshoot.map((t) => `<tr><td>${mdInline(t.sign)}</td><td>${mdInline(t.cause)}</td><td>${mdInline(t.fix)}</td></tr>`).join('')}</tbody></table></div>`);
        }
      };
      if (c.script) loadPy(c.script).then((py) => show(py), (e) => show(null, e));
      else show(null);
    } else {
      body.innerHTML = `<p class="muted">共 ${l.quiz.length} 题，正确率 ≥ ${pct(SITE.quizPass)} 即完成本课。${p.quiz ? `历史最好成绩：${pct(p.quiz)}` : ''}</p><div class="quiz"></div>`;
      renderQuiz($('.quiz', body), l.quiz, (score) => {
        Progress.setQuiz(id, score);
        if (score >= SITE.quizPass) $('.stepper a:nth-child(4)', el).classList.add('done');
      });
    }

    $('.step-nav', el).innerHTML = `
      ${step > 1 ? `<a class="btn ghost" href="${base}/${step - 1}">← 上一步</a>` : ''}<span class="spacer"></span>
      ${step < 4 ? `<button class="btn" type="button" id="step-done">完成「${STEPS[step - 1]}」，下一步 →</button>`
                 : `<a class="btn ghost" href="${goNext}">${next ? `下一课：${esc(next.title)}` : '去看全部经典案例'} →</a>`}`;
    $('#step-done', el)?.addEventListener('click', () => { Progress.markStep(id, step); location.hash = goNext; });
  }

  function pageCases(el) {
    setTitle('经典案例');
    el.innerHTML = `
      <h1>经典案例</h1>
      <p class="muted">每个模型配一个最经典的任务。模型都在本机用 PyTorch 训练好、导出到网页里运行——点开就能玩，往下翻能看到训练曲线和每一段真实代码。</p>
      ${stagesOf(lessonList().filter((l) => l.case)).map((s) => `<h2>${esc(s.name)}</h2><div class="grid">${s.lessons.map((l) => `
        <a class="card case-card" href="#/learn/${l.id}/3"><div class="icon" aria-hidden="true">${l.case.icon || '🧩'}</div>
          <h3>${esc(l.case.title)}</h3><p class="muted small">${esc(l.case.summary || '')}</p><span class="tag blue">${esc(l.title)}</span></a>`).join('')}</div>`).join('')}`;
  }

  function pageHandbook(el) {
    setTitle('排错手册');
    const ls = lessonList().filter((l) => l.case?.troubleshoot?.length);
    el.innerHTML = `
      <h1>排错手册</h1>
      <p class="muted">训练神经网络时最常见的问题：先看「现象」，再对照「可能原因」和「怎么办」。内容汇总自各课经典案例的排错清单。</p>
      <input class="filter" type="search" placeholder="搜索现象或关键词，例如：loss、过拟合、NaN、梯度" aria-label="搜索排错手册">
      ${ls.map((l) => `<section class="hb"><h2>${l.icon || ''} ${esc(l.title)} <a class="small" href="#/learn/${l.id}/3">去案例 →</a></h2>
        <div class="table-wrap"><table class="data trouble"><thead><tr><th>现象</th><th>可能原因</th><th>怎么办</th></tr></thead>
        <tbody>${l.case.troubleshoot.map((t) => `<tr><td>${mdInline(t.sign)}</td><td>${mdInline(t.cause)}</td><td>${mdInline(t.fix)}</td></tr>`).join('')}</tbody></table></div></section>`).join('')}
      <p class="muted empty" hidden>没有找到匹配的条目，换个关键词试试。</p>`;
    $('.filter', el).addEventListener('input', (e) => {
      const kw = e.target.value.trim().toLowerCase();
      let any = false;
      $$('section.hb', el).forEach((sec) => {
        let n = 0;
        $$('tbody tr', sec).forEach((tr) => { const hit = !kw || tr.textContent.toLowerCase().includes(kw); tr.hidden = !hit; n += hit; });
        sec.hidden = !n;
        any = any || n > 0;
      });
      $('.empty', el).hidden = any;
    });
  }

  function pageNotFound(el) {
    setTitle('页面不存在');
    el.innerHTML = '<h1>页面不存在</h1><p>链接可能已经变化。<a href="#/">回到首页</a></p>';
  }

  /* ---------------- 路由 ---------------- */
  const routes = [
    [/^\/$/, pageHome],
    [/^\/learn$/, pageLearn],
    [/^\/learn\/([\w-]+)(?:\/([1-4]))?$/, pageLesson],
    [/^\/cases$/, pageCases],
    [/^\/handbook$/, pageHandbook],
    [/^\/selftest(?:\/(run))?$/, (el, run) => Selftest.run(el, !!run)],   // #/selftest/run 打开即自动运行
  ];

  function route() {
    const path = location.hash.replace(/^#/, '') || '/';
    const el = $('#app');
    const hit = routes.find(([re]) => re.test(path));
    el.innerHTML = '';
    try {
      if (hit) hit[1](el, ...path.match(hit[0]).slice(1));
      else pageNotFound(el);
    } catch (e) {
      console.error(e);
      el.innerHTML = `<div class="notice err">页面渲染出错：${esc(e.message)}。打开 <a href="#/selftest">站点自检</a> 排查内容问题。</div>`;
    }
    const top = '/' + (path.split('/')[1] || '');
    $$('#nav a').forEach((a) => a.classList.toggle('active', a.getAttribute('href') === '#' + top));
    window.scrollTo(0, 0);
    el.focus({ preventScroll: true });
  }

  async function start() {
    Store.load();
    $('#brand-name').textContent = SITE.name;
    $('#nav').innerHTML = SITE.nav.map(([text, href]) => `<a href="${href}">${esc(text)}</a>`).join('');
    marked.use({ gfm: true });
    await Promise.all(SITE.lessons.map((id) => loadScript(`content/lessons/${id}.js`).catch((e) => console.warn(e.message))));
    window.addEventListener('hashchange', route);
    route();
  }

  window.App = { start, $, $$, esc, pct, mdHtml, mdInline, debounce, loadScript, loadPy, fill, Store, Progress, lessonList, renderQuiz, renderCheck, mountWidget, renderBlocks, STEPS };
})();
