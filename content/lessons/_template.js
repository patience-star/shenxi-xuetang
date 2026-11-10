/* =====================================================================
 * 课程模板 —— 新增一课的步骤：
 *   1. 复制本文件为 content/lessons/<你的id>.js（文件名 = id，只用小写字母、数字、短横线）；
 *   2. 按注释修改内容；配套 PyTorch 脚本放在 pytorch/<编号>_<名字>.py，运行 python tools/run.py <编号>_<名字>；
 *   3. 在 config.js 的 lessons 数组里加上 '<你的id>'（数组顺序 = 学习顺序）；
 *   4. 刷新页面，打开「站点自检」（#/selftest/run）确认没有报错。
 * 本文件以下划线开头，不会被加载，只作为参考。
 *
 * 写 Markdown 的小提示：
 *   - 用 md`...` 包裹，缩进会自动去掉；
 *   - 行内代码的反引号要写成 \`（反斜杠 + 反引号），代码块写成 \`\`\`；
 *   - 需要字面上的反斜杠时写两个 \\；不要出现 ${（会被当成模板变量）；
 *   - 折叠的「深入一点」：<details><summary>深入一点：标题</summary> 空一行 → Markdown 正文 → 空一行 </details>
 * ===================================================================== */
Content.addLesson({
  id: 'my-lesson',                 // 唯一 id，与文件名一致
  title: '课程标题',
  icon: '📘',                      // 一个 emoji
  stage: '预备知识',               // 学习路线里的阶段名；相同 stage 的课归为一组（按 config.js 顺序）
  summary: '一句话介绍（显示在卡片上）',
  minutes: 40,                     // 预计学习时长（分钟）
  goals: ['学习目标 1', '学习目标 2'],

  // 第 1 步「视频导学」：B 站视频。bvid 在视频网址里（BV 开头的 12 位）；多 P 视频用 p 指定第几集
  // 改完运行 python tools/check_videos.py 确认视频有效
  videos: [
    { bvid: 'BV1bx411M7Zx', p: 1, title: '视频标题', up: 'UP 主名字', minutes: 19, note: '一句话说明（可省略）' },
  ],
  searchKeyword: 'B 站搜索关键词',  // 视频失效时的兜底搜索链接

  // 第 2 步「原理精讲」：按顺序排列的内容块
  //   字符串              → Markdown 正文
  //   { check: {...} }    → 随堂小问（答错可以重选，答对显示解析）
  //   { widget: 'id', title?, intro?, ...其余参数 } → 互动组件 js/widgets/<id>.js（其余参数原样传给组件）
  //   { scenario: {...} } → 情景模拟（多轮选择题对话）
  knowledge: [
    md`
      ## 1. 小标题

      正文，支持 **加粗**、表格、列表、> 引用（显示为提示框）。
    `,
    { widget: 'diagram', kind: 'neuron', title: '可选：覆盖组件标题', intro: '可选：组件上方的说明（Markdown）' },
    { check: {
      q: '随堂小问的题干',
      options: ['选项 A', '选项 B', '选项 C'],
      answer: 1,                   // 正确选项的下标，从 0 开始
      explain: '答对后显示的解析（Markdown）',
      hint: '答错时的提示（可省略）',
    } },
  ],

  // 第 3 步「经典案例」
  case: {
    title: '案例标题',
    icon: '✍️',
    summary: '一句话（显示在「经典案例」页的卡片上）',
    featured: true,                // 是否显示在首页「经典案例」区
    script: '05_mlp_mnist',        // 配套脚本 pytorch/05_mlp_mnist.py（运行 tools/run.py 后生成 assets/py/05_mlp_mnist.js）
    // 内容块同上，另外：
    //   { code: true } → 脚本的逐段代码 + 真实输出（不写时自动放在最后）
    //   Markdown 里的 {{key}} → 脚本 site_export['summary'][key] 的值（重新训练后自动更新）
    //   组件参数里会自动带上 py（本课脚本的运行结果），组件可以直接用其中的权重和数据
    blocks: [
      md`## 任务\n\n为什么选这个任务、数据长什么样。`,
      { widget: 'digit-pad', model: '05_mlp_mnist' },
      { code: true },
      md`## 结果解读\n\n测试集准确率 {{test_acc}}。`,
    ],
    // 排错清单（也会汇总到「排错手册」页）：现象 / 可能原因 / 怎么办
    troubleshoot: [
      { sign: '训练 loss 不下降', cause: '学习率太小或太大', fix: '把学习率按 10 倍调整试试' },
    ],
  },

  // 第 4 步「闯关练习」
  //   单选：answer 为下标；多选：answer 为下标数组（如 [0, 2]）
  //   计算填空：不写 options，answer 为数字（默认允许 0.5% 误差，可用 tol 指定绝对误差，unit 显示单位）
  //   读代码题：题干里可以放代码块（题干多行时按完整 Markdown 渲染）
  quiz: [
    { q: '单选题题干', options: ['A', 'B', 'C', 'D'], answer: 0, explain: '解析' },
    { q: '多选题题干（多选）', options: ['A', 'B', 'C', 'D'], answer: [0, 2], explain: '解析' },
    { q: '计算填空：28 × 28 = ？', answer: 784, explain: '解析' },
  ],
});
