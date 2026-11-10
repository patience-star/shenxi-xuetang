"""本地预览服务器：python tools/serve.py [端口，默认 8010]

和 python -m http.server 一样，只是禁止浏览器缓存——修改文件后刷新页面就能看到效果。
同时打印局域网地址：手机和电脑连同一个 Wi-Fi，用手机浏览器打开那个地址即可
（Windows 第一次运行时如果弹出防火墙提示，选「允许」）。只用 Python 标准库。
"""
import functools
import http.server
import pathlib
import socket
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, '.js': 'text/javascript', '.json': 'application/json'}

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()


def lan_ip():
    """本机在局域网里的地址（不会真的发出数据包）。"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(('8.8.8.8', 80))
            return s.getsockname()[0]
    except OSError:
        return None


if __name__ == '__main__':
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding='utf-8', errors='replace')
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8010
    handler = functools.partial(NoCacheHandler, directory=str(ROOT))
    ip = lan_ip()
    print(f'深析学堂已启动：http://localhost:{port}', flush=True)
    if ip:
        print(f'手机（同一 Wi-Fi）打开：http://{ip}:{port}', flush=True)
    print('按 Ctrl+C 停止', flush=True)
    http.server.ThreadingHTTPServer(('', port), handler).serve_forever()
