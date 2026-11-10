"""检查内容里引用的 B 站视频是否仍然有效。

用法：python tools/check_videos.py
做了什么：扫描 content/ 下所有 .js 文件里的 BV 号（及 p 分集号），逐个调用 B 站公开的视频信息接口，
打印标题、UP 主、播放量和分集标题；失效（删除、下架、不可见）的视频会标红并让脚本以非 0 退出码结束。
只用 Python 标准库。
"""
import json
import pathlib
import re
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
API = 'https://api.bilibili.com/x/web-interface/view?bvid={}'
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 'Referer': 'https://www.bilibili.com'}


def find_refs():
    """返回 {bvid: {p, ...}}：同一视频可能引用了不同分集。"""
    refs = {}
    for path in sorted((ROOT / 'content').rglob('*.js')):
        if path.name.startswith('_'):
            continue
        text = path.read_text('utf-8')
        for m in re.finditer(r"bvid:\s*'(BV[0-9A-Za-z]{10})'(?:,\s*p:\s*(\d+))?", text):
            refs.setdefault(m.group(1), set()).add(int(m.group(2) or 1))
    return refs


def main():
    refs = find_refs()
    bad = 0
    for bvid, pages in refs.items():
        try:
            req = urllib.request.Request(API.format(bvid), headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.load(r)
        except Exception as e:  # 网络问题不算视频失效，单独提示
            print(f'⚠️  {bvid}  无法访问接口：{e}')
            continue
        if data.get('code') != 0:
            bad += 1
            print(f'❌ {bvid}  已失效：{data.get("message")}（引用的分集：{sorted(pages)}）')
            continue
        d = data['data']
        print(f'✅ {bvid}  {d["title"]}  · UP 主：{d["owner"]["name"]}  · 播放 {d["stat"]["view"]:,}')
        for p in sorted(pages):
            part = next((x for x in d.get('pages', []) if x['page'] == p), None)
            if part is None:
                bad += 1
                print(f'   ❌ 第 {p} 集不存在（共 {d.get("videos", 1)} 集）')
            elif d.get('videos', 1) > 1:
                print(f'   · 第 {p} 集：{part["part"]}')
        time.sleep(0.3)
    print(f'\n共 {len(refs)} 个视频，{bad} 个问题。')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
