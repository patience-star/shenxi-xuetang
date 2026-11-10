"""一次性准备唐诗语料：python tools/prep_poems.py

从 chinese-poetry 项目（MIT 许可，https://github.com/chinese-poetry/chinese-poetry）下载《全唐诗》，
繁体转简体，挑出每句 5 个字的「五言绝句」（4 句）和「五言律诗」（8 句），每首一行写入 pytorch/data/tang5.txt。
第 9、10 课的脚本用它训练写诗模型。整理好的文件已经放在项目里，一般不需要重新运行。

需要：pip install opencc-python-reimplemented（繁简转换）
"""
import json
import pathlib
import re
import sys
import urllib.parse
import urllib.request

from opencc import OpenCC

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / 'pytorch' / 'data' / 'tang5.txt'
BASE = 'https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/' + urllib.parse.quote('全唐诗') + '/poet.tang.{}.json'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    cc = OpenCC('t2s')
    poems, seen = [], set()
    for k in range(0, 58000, 1000):
        with urllib.request.urlopen(urllib.request.Request(BASE.format(k), headers={'User-Agent': 'Mozilla/5.0'}), timeout=60) as r:
            data = json.load(r)
        for p in data:
            text = cc.convert(''.join(p.get('paragraphs', [])))
            lines = [s for s in re.split('[，。]', text) if s]
            if len(lines) not in (4, 8) or any(len(s) != 5 for s in lines):
                continue
            if not re.fullmatch(r'[一-鿿，。]+', text):      # 去掉含缺字（□）、注释括号等的诗
                continue
            poem = ''.join(f'{a}，{b}。' for a, b in zip(lines[::2], lines[1::2]))
            if poem not in seen:
                seen.add(poem)
                poems.append(poem)
        print(f'已处理 poet.tang.{k}.json，累计 {len(poems)} 首', flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(poems) + '\n', 'utf-8')
    print(f'完成：{len(poems)} 首五言诗写入 {OUT.relative_to(ROOT)}（{OUT.stat().st_size / 1e6:.1f} MB）')


if __name__ == '__main__':
    main()
