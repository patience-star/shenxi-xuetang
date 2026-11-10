"""运行教学脚本，生成网站用的数据文件 assets/py/<名字>.js

用法：
    python tools/run.py 05_mlp_mnist        运行一个（可以写多个；名字可带可不带 .py）
    python tools/run.py all                 运行 pytorch/ 下全部脚本（ResNet 最久，约半小时以上）

做了什么：
  1. 在 pytorch/ 目录下按 "# %%" 把脚本切成若干段，逐段执行（和 VS Code / Jupyter 的分段运行一样，各段共享变量），
     记录每段打印的输出和耗时——网站「经典案例」里展示的代码和输出，就是这一次真实运行的结果；
  2. 读取脚本最后一段定义的 site_export 字典并导出：
       model   : nn.Module（或 {前缀: nn.Module}）。权重先四舍五入到 float16（网页里用的就是这一份），再以 base64 写入
       golden  : (输入, 函数)。用 float16 权重重新计算的「黄金样例」，站点自检据此核对 JS 模型和 PyTorch 是否一致
       history : 训练曲线（列表的字典）
       summary : 正文里 {{占位符}} 用到的数字 / 文字
       extras  : 其它给可视化用的数据（张量 / ndarray 自动编码：uint8 原样，整数 int32，其余 float16）
  3. 写成 assets/py/<名字>.js（网页用 <script> 加载，双击打开 index.html 也能用）。

脚本本身不依赖本工具：学习者下载后可以直接 python xxx.py 运行（site_export 那段只是定义一个字典）。
"""
import base64
import contextlib
import datetime
import io
import json
import os
import pathlib
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parent.parent
PYDIR = ROOT / 'pytorch'
OUT = ROOT / 'assets' / 'py'


def split_cells(src):
    """按 '# %%' 切分，返回 [(标题, 展示用代码, 起始行号, 原始代码)]。第一个标记之前的内容（文件说明）作为「脚本说明」段。"""
    cells, title, buf, start = [], '脚本说明', [], 0
    lines = src.splitlines()
    for i, line in enumerate(lines + ['# %%']):
        if line.startswith('# %%'):
            if any(l.strip() for l in buf):
                raw = '\n'.join(buf)
                cells.append((title, raw.strip('\n'), start, raw))
            title, buf, start = line[4:].strip() or f'第 {len(cells) + 1} 段', [], i + 1
        else:
            buf.append(line)
    return cells


class Tee(io.TextIOBase):
    """同时写到多个流：既记录输出，又在终端里实时显示训练进度。"""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def b64(a):
    return base64.b64encode(a.tobytes()).decode('ascii')


def enc(x, kind=None):
    """张量 / ndarray → {'shape': [...], 'f16' | 'f32' | 'u8' | 'i32': base64}"""
    import numpy as np
    import torch
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    shape = list(a.shape)
    if kind == 'f32':
        return {'shape': shape, 'f32': b64(np.ascontiguousarray(a, dtype='<f4'))}
    if a.dtype == np.uint8:
        return {'shape': shape, 'u8': b64(np.ascontiguousarray(a))}
    if a.dtype.kind in 'iub':
        return {'shape': shape, 'i32': b64(np.ascontiguousarray(a, dtype='<i4'))}
    return {'shape': shape, 'f16': b64(np.ascontiguousarray(a, dtype='<f2'))}


def jsonable(v):
    """history / summary / extras → 能写进 JSON 的值。张量和数组编码成 base64，Python 列表保持列表。"""
    import numpy as np
    import torch
    if isinstance(v, (torch.Tensor, np.ndarray)):
        return enc(v)
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return float(f'{float(v):.6g}')
    if isinstance(v, np.integer):
        return int(v)
    return v


def export_model(model):
    """权重四舍五入到 float16 后导出（原地修改模型，之后算的黄金样例用的就是这份权重）。共享的权重只存一次。"""
    import torch
    mods = model if isinstance(model, dict) else {'': model}
    W, seen = {}, {}
    with torch.no_grad():
        for prefix, m in mods.items():
            m.eval()
            for k, t in m.state_dict(keep_vars=True).items():
                if not t.is_floating_point():   # 如 BatchNorm 的 num_batches_tracked
                    continue
                key = f'{prefix}.{k}' if prefix else k
                ptr = t.data_ptr()
                if ptr in seen:
                    W[key] = {'same': seen[ptr]}
                    continue
                seen[ptr] = key
                t.data.copy_(t.data.half().float())
                W[key] = enc(t.data, 'f16')
    return W


def export_golden(golden):
    import torch
    inputs, fn = golden
    ins = tuple(inputs) if isinstance(inputs, (tuple, list)) else (inputs,)
    with torch.no_grad():
        outs = fn(*ins)
    outs = tuple(outs) if isinstance(outs, (tuple, list)) else (outs,)
    return {'in': [enc(x, 'f32') for x in ins], 'out': [enc(o, 'f32') for o in outs]}


def run(name):
    import torch
    name = name.removesuffix('.py')
    path = PYDIR / f'{name}.py'
    if not path.exists():
        sys.exit(f'找不到 {path}')
    src = path.read_text('utf-8')
    print(f'\n▶ 运行 {path.relative_to(ROOT)}', flush=True)
    os.chdir(PYDIR)
    ns = {'__name__': '__main__', '__file__': str(path)}
    cells, t_all = [], time.time()
    for title, code, start, raw in split_cells(src):
        print(f'── [{len(cells) + 1}] {title}', flush=True)
        buf, t0 = io.StringIO(), time.time()
        try:
            with contextlib.redirect_stdout(Tee(buf, sys.__stdout__)):
                exec(compile('\n' * start + raw, str(path), 'exec'), ns)
        except Exception:
            traceback.print_exc()
            sys.exit(f'\n✗ {name} 在「{title}」这一段出错，没有生成数据文件。')
        cells.append({'title': title, 'code': code, 'out': buf.getvalue(), 'secs': round(time.time() - t0, 1)})

    data = {
        'name': name, 'file': f'pytorch/{name}.py', 'source': src, 'cells': cells,
        'meta': {'date': datetime.date.today().isoformat(), 'torch': torch.__version__.split('+')[0], 'secs': round(time.time() - t_all, 1)},
    }
    exp = ns.get('site_export') or {}
    if 'model' in exp:
        data['weights'] = export_model(exp['model'])
    if 'golden' in exp:
        data['golden'] = export_golden(exp['golden'])
    for k in ('history', 'summary', 'extras'):
        if k in exp:
            data[k] = jsonable(exp[k])
    OUT.mkdir(parents=True, exist_ok=True)
    js = (f'/* 由 tools/run.py 根据 pytorch/{name}.py 的真实运行结果生成，请勿手改。重新生成：python tools/run.py {name} */\n'
          f'PY[{json.dumps(name)}] = {json.dumps(data, ensure_ascii=False, separators=(",", ":"))};\n')
    out = OUT / f'{name}.js'
    out.write_text(js, 'utf-8')
    print(f'✓ 已生成 {out.relative_to(ROOT)}（{len(js.encode()) / 1024:.0f} KB，用时 {data["meta"]["secs"]:.0f} 秒）', flush=True)


def main():
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding='utf-8', errors='replace')
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    names = [p.stem for p in sorted(PYDIR.glob('[0-9]*.py'))] if args == ['all'] else args
    for n in names:
        run(n)


if __name__ == '__main__':
    main()
