/* =====================================================================
 * 站点自检：#/selftest（点按钮运行）或 #/selftest/run（打开即运行）
 *  1. 内容体检：课程字段、题目答案下标、内容块类型、排错清单
 *  2. 互动组件：能否加载，并在后台用课程里的参数试运行一遍
 *  3. 脚本数据：assets/py/<名字>.js 是否存在、文字占位符是否都有对应数字
 *  4. 黄金样例：网页里的 JS 模型与 PyTorch 对同一输入的输出是否一致
 *  5. JS 训练引擎：反向传播梯度 vs 数值差分
 * 结果同时写入 window.__selftest，便于自动化检查。
 * ===================================================================== */
(function () {
  'use strict';

  function validate() {
    const issues = [];
    const empty = (v) => v == null || v === '' || (Array.isArray(v) && !v.length);
    const need = (obj, fields, where) => fields.forEach((f) => { if (empty(obj?.[f])) issues.push(`${where} 缺少字段 ${f}`); });
    const checkQ = (q, where, allowNum) => {
      if (!q || !q.q) { issues.push(`${where}：缺少题干`); return; }
      if (!q.explain) issues.push(`${where}：缺少解析 explain`);
      if (!q.options) {
        if (!allowNum || typeof q.answer !== 'number') issues.push(`${where}：没有 options 时必须是计算填空（answer 为数字）`);
        return;
      }
      if (!Array.isArray(q.options) || q.options.length < 2) { issues.push(`${where}：选项不足 2 个`); return; }
      const ans = [].concat(q.answer);
      if (!ans.length || ans.some((a) => !Number.isInteger(a) || a < 0 || a >= q.options.length)) issues.push(`${where}：answer 下标越界（选项从 0 开始编号）`);
    };
    // 文字里的尖括号会被当成 HTML 标签（例如写「<pad>」会被浏览器吞掉、甚至打乱后面的页面结构）。
    // 允许的标签之外一律报错；要显示尖括号，请用反引号把它写成代码，如 `<pad>`。
    const ALLOWED = new Set(['details', 'summary', 'div', 'span', 'a', 'b', 'i', 'em', 'strong', 'br', 'sub', 'sup', 'code', 'p', 'ul', 'ol', 'li', 'table', 'tr', 'td', 'th', 'kbd']);
    const tagCheck = (text, where) => {
      if (typeof text !== 'string') return;
      const plain = text.replace(/```[\s\S]*?```/g, '').replace(/`[^`]*`/g, '');
      for (const m of plain.matchAll(/<\/?([a-zA-Z][\w-]*)[^>]*>/g)) {
        if (!ALLOWED.has(m[1].toLowerCase())) issues.push(`${where}：文字里的「${m[0]}」会被当成 HTML 标签吞掉，请用反引号写成 \`${m[0]}\``);
      }
    };
    const qText = (q, where) => { if (!q) return; tagCheck(q.q, where); (q.options || []).forEach((o) => tagCheck(o, where)); tagCheck(q.explain, where); tagCheck(q.hint, where); };
    const uses = [];   // [课程, 组件 id, 参数, 是否在案例里]
    const blocks = (l, list, where, inCase) => (list || []).forEach((b, i) => {
      const w = `${where}第 ${i + 1} 块`;
      if (typeof b === 'string') { tagCheck(b, w); return; }
      if (b?.check) { checkQ(b.check, `${w}（随堂小问）`, false); qText(b.check, `${w}（随堂小问）`); }
      else if (b?.widget) { uses.push([l, b.widget, b, inCase]); tagCheck(b.intro, w); tagCheck(b.note, w); }
      else if (b?.scenario) (b.scenario.steps || []).forEach((s, k) => { if (!s.options?.some((o) => o.ok)) issues.push(`${w} 情景模拟第 ${k + 1} 步没有正确选项（ok: true）`); });
      else if (b?.code) { if (!inCase) issues.push(`${w}：{ code: true } 只能用在经典案例里`); }
      else issues.push(`${w} 无法识别（应为 Markdown 文本、check、widget、scenario 或 code）`);
    });

    SITE.lessons.forEach((id) => { if (!Content.lessons[id]) issues.push(`config.js 登记的课程「${id}」没有加载：检查 content/lessons/${id}.js 是否存在、有无语法错误（F12 打开控制台查看）`); });
    Object.keys(Content.lessons).forEach((id) => { if (!SITE.lessons.includes(id)) issues.push(`课程「${id}」已加载但不在 config.js 的 lessons 里（不会显示）`); });
    for (const l of App.lessonList()) {
      const w = `课程 ${l.id}`;
      need(l, ['title', 'stage', 'summary', 'goals', 'videos', 'knowledge', 'case', 'quiz'], w);
      (l.videos || []).forEach((v, i) => {
        if (!/^BV[0-9A-Za-z]{10}$/.test(v.bvid || '')) issues.push(`${w} 第 ${i + 1} 个视频的 bvid 格式不对`);
        need(v, ['title', 'up'], `${w} 第 ${i + 1} 个视频`);
      });
      blocks(l, l.knowledge, `${w} 原理精讲`, false);
      if (l.case) {
        need(l.case, ['title', 'summary', 'blocks'], `${w} 经典案例`);
        blocks(l, l.case.blocks, `${w} 经典案例`, true);
        (l.case.troubleshoot || []).forEach((t, i) => {
          need(t, ['sign', 'cause', 'fix'], `${w} 排错清单第 ${i + 1} 条`);
          ['sign', 'cause', 'fix'].forEach((k) => tagCheck(t[k], `${w} 排错清单第 ${i + 1} 条`));
        });
      }
      (l.quiz || []).forEach((q, i) => { checkQ(q, `${w} 闯关第 ${i + 1} 题`, true); qText(q, `${w} 闯关第 ${i + 1} 题`); });
    }
    return { issues, uses };
  }

  async function run(el, autoRun) {
    const { $, esc } = App;
    document.title = `站点自检 · ${SITE.name}`;
    el.innerHTML = `
      <h1>站点自检</h1>
      <p class="muted">修改内容、配置或重新训练模型后打开本页：检查课程字段、试运行全部互动组件、核对脚本数据，并用「黄金样例」确认网页里的模型和 PyTorch 输出一致。</p>
      <p class="row"><button class="btn" type="button" data-act="run">开始自检</button><span class="summary muted" aria-live="polite">约需 10–30 秒</span></p>
      <div class="table-wrap"><table class="data selftest"><thead><tr><th>类别</th><th>项目</th><th>结果</th><th>说明</th></tr></thead><tbody></tbody></table></div>`;
    const btn = $('[data-act=run]', el), summary = $('.summary', el), tbody = $('tbody', el);

    const go = async () => {
      btn.disabled = true;
      tbody.innerHTML = '';
      const rows = [];
      const add = (cat, item, status, msg) => {
        rows.push({ cat, item, status, msg });
        const icon = { ok: '✅ 通过', warn: '⚠️ 注意', fail: '❌ 失败' }[status];
        tbody.insertAdjacentHTML('beforeend', `<tr class="${status}"><td>${esc(cat)}</td><td>${esc(item)}</td><td>${icon}</td><td>${esc(msg)}</td></tr>`);
      };

      // 1. 内容体检
      summary.textContent = '内容体检…';
      const { issues, uses } = validate();
      if (issues.length) issues.forEach((m) => add('内容体检', '字段与结构', 'fail', m));
      else add('内容体检', `${App.lessonList().length} 课`, 'ok', '字段、题目答案、内容块类型都正确');

      // 2. 脚本数据（先加载，后面试运行组件要用）
      const pys = {};
      for (const l of App.lessonList()) {
        const name = l.case?.script;
        if (!name) continue;
        summary.textContent = `加载脚本数据 ${name}…`;
        try {
          const py = (pys[l.id] = await App.loadPy(name));
          add('脚本数据', name, py.cells?.length ? 'ok' : 'fail', `${py.cells?.length || 0} 段代码 · ${py.meta?.date || '?'} 运行 · 文件约 ${Math.round(JSON.stringify(py).length / 1024)} KB`);
          const text = (l.case.blocks || []).map((b) => (typeof b === 'string' ? b : `${b?.intro || ''} ${b?.note || ''}`)).join('\n');
          const missing = [...new Set([...text.matchAll(/\{\{(\w+)\}\}/g)].map((m) => m[1]).filter((k) => !(py.summary && k in py.summary)))];
          if (missing.length) add('占位符', name, 'fail', `正文用了 {{${missing.join('}}、{{')}}}，但脚本 summary 里没有`);
        } catch (e) { add('脚本数据', name, 'fail', e.message); }
      }

      // 3. 互动组件：加载 + 后台试运行
      const sandbox = document.createElement('div');
      sandbox.style.cssText = 'position:absolute;left:-10000px;top:0;width:820px;visibility:hidden';
      document.body.appendChild(sandbox);
      const loaded = new Set();
      for (const [l, id, opts, inCase] of uses) {
        summary.textContent = `试运行组件 ${id}…`;
        if (!loaded.has(id)) {
          loaded.add(id);
          try { await App.loadScript(`js/widgets/${id}.js`); } catch (e) { /* 下面统一判断 */ }
          if (!Widgets[id]) { add('互动组件', id, 'fail', `无法加载 js/widgets/${id}.js`); continue; }
        }
        if (!Widgets[id]) continue;
        const box = document.createElement('div');
        sandbox.appendChild(box);
        try {
          await Widgets[id].render(box, { ...opts, py: inCase ? pys[l.id] : undefined });
          add('互动组件', `${id}（${l.id}）`, 'ok', '加载并试运行成功');
        } catch (e) {
          console.error(e);
          add('互动组件', `${id}（${l.id}）`, 'fail', `试运行出错：${e.message}`);
        }
        box.remove();
      }
      sandbox.remove();

      // 4. 黄金样例：JS 模型 vs PyTorch
      for (const py of Object.values(pys)) {
        if (!py.golden) continue;
        summary.textContent = `核对模型 ${py.name}…`;
        await new Promise((r) => setTimeout(r, 0));
        try {
          const t0 = performance.now();
          const g = NN.goldenCheck(py);
          add('黄金样例', py.name, g.err < 1e-3 ? 'ok' : 'fail', `${g.n} 个输出数值，与 PyTorch 的最大相对误差 ${g.err.toExponential(2)}（${Math.round(performance.now() - t0)} ms）`);
        } catch (e) { add('黄金样例', py.name, 'fail', e.message); }
      }

      // 5. JS 训练引擎
      const ge = NN.gradCheck();
      add('JS 训练引擎', '反向传播梯度检验', ge < 1e-6 ? 'ok' : 'fail', `反向传播 vs 数值差分，最大相对误差 ${ge.toExponential(2)}`);

      const failed = rows.filter((r) => r.status === 'fail');
      summary.textContent = failed.length ? `完成：${failed.length} 项有问题，请查看下表` : `完成：全部 ${rows.length} 项通过 ✓`;
      window.__selftest = { done: true, failed, rows };
      btn.disabled = false;
    };
    btn.addEventListener('click', go);
    if (autoRun) go();
  }

  window.Selftest = { run, validate };
})();
