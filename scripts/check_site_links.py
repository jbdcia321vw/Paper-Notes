#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 MkDocs 构建产物中的本地链接与静态资源是否存在。"""

import argparse
import html
import re
import sys
import urllib.parse
from pathlib import Path

ATTR_RE = re.compile(r'(?:href|src)="([^"]+)"')


def check_links(site_dir: Path, base_path: str = "/Paper-Notes/", verbose: bool = False) -> list:
    site_dir = site_dir.resolve()
    base_path = "/" + base_path.strip("/") + "/" if base_path.strip("/") else "/"
    missing = []
    checked = 0

    for page in site_dir.rglob("*.html"):
        if page.name == "404.html":
            continue
        text = page.read_text(encoding="utf-8", errors="ignore")
        for raw in ATTR_RE.findall(text):
            url = html.unescape(raw).strip()
            if not url or url.startswith("#"):
                continue
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme or parsed.netloc:
                continue
            path_part = urllib.parse.unquote(parsed.path)
            if not path_part:
                continue
            if path_part.startswith(base_path):
                target = site_dir / path_part[len(base_path):]
            elif path_part.startswith("/"):
                target = site_dir / path_part.lstrip("/")
            else:
                target = page.parent / path_part
            target = target.resolve()
            if path_part.endswith("/"):
                target = target / "index.html"
            checked += 1
            if not target.exists():
                missing.append((str(page.relative_to(site_dir)), url, str(target)))
                if verbose:
                    print(f"[Missing] {page.relative_to(site_dir)} -> {url}")

    print(f"[Links] checked={checked} missing={len(missing)}")
    return missing


def main():
    parser = argparse.ArgumentParser(description="检查 MkDocs site/ 中的本地链接")
    parser.add_argument("--site", default="site", help="MkDocs 输出目录（默认 site）")
    parser.add_argument("--base-path", default="/Paper-Notes/", help="站点 base path（默认 /Paper-Notes/）")
    parser.add_argument("--verbose", action="store_true", help="逐条打印缺失链接")
    args = parser.parse_args()

    site_dir = Path(args.site)
    if not site_dir.exists():
        print(f"[Error] 站点目录不存在: {site_dir}")
        return 1

    missing = check_links(site_dir, args.base_path, args.verbose)
    if missing:
        print("[Error] 发现缺失链接：")
        for page, url, target in missing[:50]:
            print(f"  - {page}: {url} -> {target}")
        if len(missing) > 50:
            print(f"  ... 其余 {len(missing) - 50} 条省略")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
