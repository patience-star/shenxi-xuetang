# 深析学堂 · 深度学习交互学习站

一个纯静态的深度学习教学网站：从张量、梯度、感知机讲起，一路到 CNN、RNN、Transformer 和生成模型。

- **13 课，5 个阶段**：预备知识（张量与 PyTorch、梯度下降、感知机、反向传播）→ 神经网络基础（多层感知机、训练技巧）→ 卷积网络（CNN 与 LeNet、ResNet）→ 序列模型（RNN、LSTM 与 GRU、Transformer）→ 生成模型（VAE、GAN）。
- **每课 4 步**：视频导学（B 站，已核实为原创上传）→ 原理精讲（边读边玩的互动图 + 随堂小问 + 折叠的推导）→ 经典案例 → 闯关练习（单选 / 多选 / 计算填空 / 读代码题，全部附解析）。
- **约 30 个互动组件**：梯度下降、感知机、计算图、神经网络训练场、优化器赛跑、卷积滑窗、注意力计算……有些在浏览器里实时训练小网络。
- **经典案例的模型在浏览器里运行**：手写数字识别（MLP、LeNet）、CIFAR-10 图像分类（ResNet-20）、预测下一个字（RNN）、藏头诗（LSTM）、日期翻译（Transformer）、隐空间漫游（VAE）、生成手写数字（DCGAN）。这些模型都是用 `pytorch/` 下的脚本在本机真实训练出来、导出后由 `js/nn.js` 运行的。
- **代码与输出都是真实的**：案例页展示的每一段 PyTorch 代码和它的输出，都来自 `tools/run.py` 的一次真实运行；正文里的关键数字用占位符引用，重新训练后自动更新。

---

## 快速开始

**方式一：双击 `index.html`**（推荐 Chrome / Edge），什么都不用装。

**方式二：本地服务器**（修改网站时预览，或者用手机看）：

```bash
python tools/serve.py
```

然后打开 <http://localhost:8010>。终端里还会打印一个局域网地址（如 `http://192.168.1.5:8010`）：**手机连同一个 Wi-Fi，用浏览器打开它即可**（Windows 第一次运行如果弹出防火墙提示，选「允许」）。

> 只有 B 站视频需要联网，其余内容（课程、组件、模型、数据）全部在本地，不依赖任何 CDN。

---

## 目录结构

```
index.html               页面外壳（一般不用改）
config.js                ★ 站点配置：名称、导航、课程列表和顺序、及格线
style.css                样式；顶部 :root 是颜色、字体、圆角等设计变量（含深色模式）
vendor/marked.umd.min.js Markdown 渲染库（已本地化）

js/app.js                引擎：路由、4 步课程页、测验 / 随堂问 / 情景模拟、经典案例（逐段代码）、排错手册、学习进度
js/charts.js             SVG 图表、像素图、热力图、概率条、手写板
js/nn.js                 浏览器里的小型神经网络库：权重解码、推理算子、9 个案例模型的前向、可训练的小 MLP
js/selftest.js           站点自检（#/selftest/run）
js/widgets/<id>.js       互动组件，一个文件一个

content/lessons/<id>.js  ★ 课程，一课一个文件（_template.js 是模板，字段说明都在里面）

pytorch/<编号>_<名字>.py   ★ 教学脚本：每课一个，按 "# %%" 分段，中文注释，可以单独运行
pytorch/data/            数据集（MNIST、CIFAR-10 由脚本自动下载；tang5.txt 是整理好的唐诗语料）
assets/py/<名字>.js       由 tools/run.py 生成：脚本的分段代码 + 真实输出 + 模型权重 + 训练曲线（不要手改）

tools/run.py             运行教学脚本，生成 assets/py/<名字>.js
tools/serve.py           本地预览服务器（不缓存，打印手机访问地址）
tools/check_videos.py    检查 B 站视频是否失效
tools/prep_poems.py      一次性整理唐诗语料（一般不需要再运行）
```

日常维护只需要改带 ★ 的文件。**没有构建步骤**：改完保存、刷新浏览器就能看到效果。

---

## 每次修改后：运行站点自检

打开 `#/selftest`（页脚有「站点自检」链接），或者直接打开 `#/selftest/run` 自动运行。它会：

1. **内容体检**：必填字段、题目答案下标、内容块类型、排错清单是否完整，config 里登记的课程是否都加载成功；
2. **互动组件**：每个被引用的组件能否加载，并用课程里写的参数在后台**试运行**一遍；
3. **脚本数据**：`assets/py/` 下的数据文件是否存在，正文里的 `{{占位符}}` 是否都能在脚本结果里找到；
4. **黄金样例**：网页里的 JS 模型和 PyTorch 对同一组输入的输出是否一致（误差 < 10⁻³）——改了 `js/nn.js` 或重新训练后尤其要看；
5. **JS 训练引擎**：浏览器里那个小 MLP 的反向传播，和数值差分比是否一致。

---

## 常见修改怎么做

| 想做的事 | 改哪里 |
|---|---|
| 改站点名称、标语、导航 | `config.js` 的 `name`、`slogan`、`nav` |
| 调整课程顺序、隐藏某一课 | `config.js` 的 `lessons` 数组：调整顺序，或在行首加 `//` 注释掉 |
| 改某课的讲解、随堂小问 | `content/lessons/<id>.js` 的 `knowledge`（Markdown 字符串和 `{ check }`） |
| 改闯关练习 | 同一文件的 `quiz`：单选 `answer: 下标`；多选 `answer: [下标…]`；计算填空不写 `options`、`answer: 数字`（可加 `tol`、`unit`） |
| 在讲解里插入一个互动组件 | 在 `knowledge` 里加 `{ widget: '组件名', intro: '说明', ...参数 }`；组件名就是 `js/widgets/` 下的文件名 |
| 插入一张结构示意图 | `{ widget: 'diagram', kind: 'mlp' }`；现有 kind：neuron、training-loop、xor-net、mlp、lenet、resblock、rnn、transformer、vae、gan，新增在 `js/widgets/diagram.js` 的 `KINDS` 里 |
| 画一张训练曲线 | 案例里加 `{ widget: 'py-chart', series: [{ path: 'history.test_acc', name: '测试准确率' }], format: 'pct' }`；`path` 指向脚本导出数据里的数组 |
| 改案例文字里的数字 | 不要手写数字：在脚本末尾 `site_export['summary']` 里加一项，正文里写 `{{名字}}` |
| 换视频 | `videos`（bvid 在视频网址里，多 P 视频用 `p` 指定集数），改完运行 `python tools/check_videos.py` |
| 修改排错清单 | 课程文件 `case.troubleshoot`：`{ sign: 现象, cause: 原因, fix: 怎么办 }`，会自动汇总到「排错手册」页 |
| 改训练设置后重新训练 | 改 `pytorch/<脚本>.py`，运行 `python tools/run.py <脚本名>`，然后打开站点自检 |
| 新增一课 | 复制 `content/lessons/_template.js`；写配套脚本并运行 `tools/run.py`；把 id 加进 `config.js` 的 `lessons`；如果案例要在网页里运行模型，在 `js/nn.js` 的 `models` 里加同名的前向函数（含 `golden`） |
| 新增互动组件 | 新建 `js/widgets/<id>.js`，写 `Widgets['<id>'] = { title, render(el, opts) {...} }`；动画循环用 `NN.loop(el, fn)`（离开页面会自动停止）；在课程里引用即可，无需登记 |
| 改主题色、字体、圆角 | `style.css` 顶部 `:root`（深色模式在紧随其后的媒体查询里） |
| 改测验及格线 | `config.js` 的 `quizPass` |

### 写内容时的注意事项

- 正文都是 **Markdown**，用 `` md`...` `` 包裹，缩进会自动去掉。
- 在 `` md`...` `` 里写行内代码，反引号要写成 `` \` ``，代码块写成 `` \`\`\` ``；需要字面上的反斜杠时写两个 `\\`；不要出现 `${`。
- 折叠的「深入一点」用 `<details><summary>标题</summary>`，后面**空一行**再写 Markdown，结尾空一行再写 `</details>`。
- 公式块用 `<div class="formula">…</div>`（等宽字体，过长时可以左右滑动）。

---

## 教学脚本与 `tools/run.py`

`pytorch/` 下每个脚本都是独立可运行的（学习者在网页上点「下载脚本」得到的就是它）：

```bash
cd pytorch
python 05_mlp_mnist.py
```

网站用的数据则由 `tools/run.py` 生成：

```bash
python tools/run.py 05_mlp_mnist      # 运行一个（可以写多个）
python tools/run.py all               # 运行全部（ResNet 最久）
```

它按 `# %%` 把脚本切成若干段逐段执行，记录每段的真实输出；再读取脚本最后一段定义的 `site_export` 字典：

| 键 | 作用 |
|---|---|
| `model` | 要在网页里运行的模型（`nn.Module`，或 `{前缀: 模块}`）。权重先四舍五入到 float16，再导出 |
| `golden` | `(输入, 函数)`：用 float16 权重重新算出的输出，站点自检据此核对 JS 模型 |
| `history` | 训练曲线（列表的字典） |
| `summary` | 正文 `{{占位符}}` 用到的数字和文字 |
| `extras` | 其它给可视化用的数据（张量会自动编码） |

各脚本在本机 CPU 上的大致耗时（22 线程，无 GPU）：01–04 几秒；05、07 半分钟左右；06、12 一两分钟；09 约 3 分钟；13 约 7 分钟；10、11 约 10 分钟；08（ResNet）约 45–55 分钟。**一次只跑一个脚本**：同时跑多个会互相抢 CPU 线程，每个都慢好几倍（页面上显示的「训练用时」也会被拖长）。

---

## 部署到网上

纯静态网站，把整个目录（可以不带 `pytorch/data/` 里的 MNIST、CIFAR 原始数据，但要保留 `tang5.txt`）上传到任意静态托管即可：GitHub Pages、Gitee Pages、Netlify、Vercel、阿里云 OSS、腾讯云 COS 等。学习进度保存在每位学习者自己浏览器的 localStorage 里，不需要后端。进度结构有大改动时，把 `config.js` 的 `storageKey` 改个版本号即可。

---

## 维护脚本

```bash
python tools/serve.py          # 本地预览（禁止缓存，改完刷新即生效；打印手机访问地址）
python tools/run.py <脚本名>    # 运行教学脚本、生成网站数据
python tools/check_videos.py   # 检查所有 B 站视频和分集是否仍然有效
python tools/prep_poems.py     # 重新整理唐诗语料（需要 pip install opencc-python-reimplemented）
```

---

## 已知限制

- 浏览器里不能运行 PyTorch：案例代码以「逐段代码 + 真实输出」展示，想动手可以下载脚本在本机运行；网页里的模型由 `js/nn.js` 重新实现，自检保证它和 PyTorch 输出一致。
- 案例模型都很小（为了能在 CPU 上训练、在手机上运行），效果有上限：写诗模型能写出格式工整的诗，但意思常常不连贯；ResNet-20 只训练了 30 轮。
- 学习进度只保存在当前浏览器里，换设备或清除浏览器数据后不保留。
- 写诗模型约 1 MB、其余模型几百 KB，在手机流量下第一次打开对应案例时需要加载一会儿。

---

## 第三方资源与许可

- [marked](https://github.com/markedjs/marked)（MIT）：Markdown 渲染
- MNIST（LeCun 等）、CIFAR-10（Krizhevsky 等）：经 torchvision 下载的公开数据集
- [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)（MIT）：《全唐诗》语料，经 OpenCC 繁转简后筛选五言诗
- 课程视频来自 B 站，版权归各 UP 主所有，通过 B 站官方播放器嵌入
