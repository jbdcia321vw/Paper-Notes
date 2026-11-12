#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_nav_and_index.py
扫描 docs/venues/ 下的所有推荐系统论文 Markdown 笔记，
自动生成：
1. 按推荐系统细分类别聚合的跨会议索引页 (docs/topics/<category>.md)
2. 按顶会/顶刊划分的会议索引页 (docs/venues/<venue>/index.md 及子目录 index.md)
3. 全站总览首页 (docs/index.md)
4. 双维度导航配置文件 (mkdocs.yml)
"""

import re
from pathlib import Path
from collections import defaultdict
from init_clean_recsys_repo import VENUES, CATEGORIES


def parse_frontmatter(md_path: Path) -> dict:
    """解析单篇论文 Markdown 笔记顶部的 YAML Frontmatter 元数据。"""
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    meta = {
        "title": md_path.stem.replace("_", " ").title(),
        "description": "",
        "date": "",
        "venue": md_path.parent.parent.name,
        "categories": [md_path.parent.name],
        "path": md_path,
    }
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not m:
        for line in text.splitlines():
            if line.startswith("# "):
                meta["title"] = line[2:].strip()
                break
        return meta

    fm = m.group(1)
    title_m = re.search(r'^title:\s*["\']?(.*?)["\']?\s*$', fm, re.MULTILINE)
    if title_m:
        meta["title"] = title_m.group(1).replace("[Paper Note] ", "").strip()

    desc_m = re.search(r'^description:\s*["\']?(.*?)["\']?\s*$', fm, re.MULTILINE)
    if desc_m:
        meta["description"] = desc_m.group(1).strip()

    date_m = re.search(r"^date:\s*(.*?)\s*$", fm, re.MULTILINE)
    if date_m:
        meta["date"] = date_m.group(1).strip()

    venue_m = re.search(r"^venue:\s*(.*?)\s*$", fm, re.MULTILINE)
    if venue_m:
        meta["venue"] = venue_m.group(1).strip()

    cats = re.findall(r"^\s*-\s*([a-z0-9_]+)\s*$", fm, re.MULTILINE)
    valid_cats = [c for c in cats if c in CATEGORIES]
    if valid_cats:
        meta["categories"] = valid_cats

    return meta


def collect_all_notes(docs_dir: Path):
    """遍历 docs/venues/ 收集所有论文解读笔记。"""
    venues_dir = docs_dir / "venues"
    notes = []
    if not venues_dir.exists():
        return notes

    for md_file in sorted(venues_dir.rglob("*.md")):
        if md_file.name == "index.md":
            continue
        meta = parse_frontmatter(md_file)
        meta["rel_from_docs"] = md_file.relative_to(docs_dir).as_posix()
        notes.append(meta)
    return notes


def build_topic_pages(docs_dir: Path, notes: list):
    """生成 docs/topics/ 下的 14 大推荐系统细分方向跨会议聚合页。"""
    topics_dir = docs_dir / "topics"
    topics_dir.mkdir(parents=True, exist_ok=True)

    cat_map = defaultdict(list)
    for n in notes:
        for c in n["categories"]:
            if c in CATEGORIES:
                cat_map[c].append(n)

    topic_index_lines = [
        "# 按推荐系统细分类别浏览 (14 大核心方向)",
        "",
        "本页面按**推荐系统研究子方向**跨会议、跨期刊聚合所有论文解读笔记。无论论文发表于 SIGIR、RecSys、KDD、WWW、ACM MM 还是 TOIS、TKDE，均可在此按技术脉络集中检索。",
        "",
        "| 推荐系统细分方向 | 英文名称 | 核心研究范畴 | 收录篇数 |",
        "| :--- | :--- | :--- | :---: |",
    ]

    for cat_slug, cinfo in CATEGORIES.items():
        items = cat_map.get(cat_slug, [])
        topic_index_lines.append(
            f"| [**{cinfo['title']}**]({cat_slug}.md) | {cinfo['en_title']} | {cinfo['desc']} | **{len(items)}** |"
        )

        page_lines = [
            f"# {cinfo['title']} ({cinfo['en_title']})",
            "",
            f"> **研究范畴**：{cinfo['desc']}",
            "",
            f"当前方向共收录 **{len(items)}** 篇顶会/顶刊论文解读：",
            "",
        ]
        if not items:
            page_lines.append("*暂无论文笔记，运行 `python scripts/generate_recsys_notes.py` 后将自动归档至此。*")
        else:
            page_lines.extend([
                "| 会议/期刊 | 论文标题 | 核心速览 / 描述 | 日期 |",
                "| :---: | :--- | :--- | :---: |",
            ])
            for item in items:
                rel_link = "../" + item["rel_from_docs"]
                desc = item["description"] or "点击查看 5 分钟核心思想精读"
                date_str = item["date"] or "-"
                page_lines.append(
                    f"| `{item['venue']}` | [**{item['title']}**]({rel_link}) | {desc} | {date_str} |"
                )

        (topics_dir / f"{cat_slug}.md").write_text("\n".join(page_lines) + "\n", encoding="utf-8")

    (topics_dir / "index.md").write_text("\n".join(topic_index_lines) + "\n", encoding="utf-8")


def build_venue_pages(docs_dir: Path, notes: list, default_year: str = "2026"):
    """生成 docs/venues/ 下的各顶会/顶刊索引页。"""
    venues_dir = docs_dir / "venues"
    venues_dir.mkdir(parents=True, exist_ok=True)

    venue_map = defaultdict(lambda: defaultdict(list))
    for n in notes:
        v_folder = n["path"].parent.parent.name
        c_folder = n["path"].parent.name
        venue_map[v_folder][c_folder].append(n)

    venue_root_lines = [
        "# 按顶会与顶刊浏览 (22 个推荐系统核心会议与期刊)",
        "",
        "| 会议/期刊代码 | 全称 | 级别与定位 | 收录篇数 |",
        "| :--- | :--- | :--- | :---: |",
    ]

    for venue_code, vinfo in VENUES.items():
        v_folder_name = f"{venue_code}{default_year}"
        v_dir = venues_dir / v_folder_name
        v_dir.mkdir(parents=True, exist_ok=True)

        total_v = sum(len(lst) for lst in venue_map[v_folder_name].values())
        venue_root_lines.append(
            f"| [**{vinfo['name']} ({default_year})**]({v_folder_name}/index.md) | {vinfo['full_name']} | {vinfo['tier']} | **{total_v}** |"
        )

        v_lines = [
            f"# {vinfo['name']} {default_year} 推荐系统论文解读汇总",
            "",
            f"- **英文全称**: {vinfo['full_name']}",
            f"- **级别定位**: {vinfo['tier']}",
            f"- **当前收录总数**: **{total_v}** 篇",
            "",
            "## 按推荐系统细分方向分布",
            "",
            "| 细分方向 | 核心范畴 | 笔记数 |",
            "| :--- | :--- | :---: |",
        ]

        for cat_slug, cinfo in CATEGORIES.items():
            c_dir = v_dir / cat_slug
            c_dir.mkdir(parents=True, exist_ok=True)
            c_items = venue_map[v_folder_name].get(cat_slug, [])
            v_lines.append(
                f"| [**{cinfo['title']}**]({cat_slug}/index.md) | {cinfo['desc']} | **{len(c_items)}** |"
            )

            c_lines = [
                f"# [{vinfo['name']} {default_year}] {cinfo['title']}",
                "",
                f"> **方向说明**：{cinfo['desc']}",
                "",
            ]
            if not c_items:
                c_lines.append("*当前会议在该细分方向下暂无论文笔记。*")
            else:
                for item in c_items:
                    c_lines.append(f"- [**{item['title']}**]({item['path'].name}) — {item['description']}")

            (c_dir / "index.md").write_text("\n".join(c_lines) + "\n", encoding="utf-8")

        (v_dir / "index.md").write_text("\n".join(v_lines) + "\n", encoding="utf-8")

    (venues_dir / "index.md").write_text("\n".join(venue_root_lines) + "\n", encoding="utf-8")


def build_home_and_mkdocs_yml(repo_root: Path, notes: list, github_user: str = "jbdcia321vw", default_year: str = "2026"):
    """生成 docs/index.md 首页与轻量级双维度 mkdocs.yml。"""
    docs_dir = repo_root / "docs"
    cat_counts = defaultdict(int)
    for n in notes:
        for c in n["categories"]:
            cat_counts[c] += 1

    home_lines = [
        "# RecSys Paper Notes · 推荐系统顶会顶刊论文 5 分钟精读库",
        "",
        f"欢迎访问 **RecSys Paper Notes**！本知识库基于 [zhaoyang97/Paper-Notes](https://github.com/zhaoyang97/Paper-Notes) 架构重构，专注于**推荐系统与信息检索领域**，支持**按顶会/顶刊**与**按推荐系统 14 大细分方向**双维度检索。",
        "",
        f"- **当前收录推荐系统论文解读**: **{len(notes)}** 篇",
        f"- **覆盖核心顶会与顶刊**: **{len(VENUES)}** 个（SIGIR / RecSys / KDD / WWW / WSDM / CIKM / ACM MM / TOIS / TKDE / TORS 等）",
        f"- **覆盖推荐系统细分方向**: **{len(CATEGORIES)}** 大类（生成式推荐 / 大模型推荐 / 多模态推荐 / 序列推荐 / 协同过滤 / 图推荐 / 社交推荐 / 冷启动与跨域等）",
        "",
        "## 一、按推荐系统细分类别快速检索",
        "",
        "| 推荐系统细分方向 | 核心研究问题 | 笔记数 |",
        "| :--- | :--- | :---: |",
    ]

    for cat_slug, cinfo in CATEGORIES.items():
        home_lines.append(
            f"| [**{cinfo['title']}**](topics/{cat_slug}.md) | {cinfo['desc']} | **{cat_counts[cat_slug]}** |"
        )

    home_lines.extend([
        "",
        "## 二、按核心会议与期刊快速检索",
        "",
        "| 类别 | 代表性会议 / 期刊入口 |",
        "| :--- | :--- |",
        f"| **IR 与推荐旗舰会议** | [SIGIR](venues/SIGIR{default_year}/index.md) · [RecSys](venues/RecSys{default_year}/index.md) · [WSDM](venues/WSDM{default_year}/index.md) · [CIKM](venues/CIKM{default_year}/index.md) · [ECIR](venues/ECIR{default_year}/index.md) · [SIGIR-AP](venues/SIGIR-AP{default_year}/index.md) |",
        f"| **数据挖掘与多媒体顶会** | [KDD](venues/KDD{default_year}/index.md) · [WWW](venues/WWW{default_year}/index.md) · [ACM MM](venues/ACM_MM{default_year}/index.md) · [ICDE](venues/ICDE{default_year}/index.md) · [ICDM](venues/ICDM{default_year}/index.md) · [DASFAA](venues/DASFAA{default_year}/index.md) |",
        f"| **综合 AI/NLP/ML 顶会** | [AAAI](venues/AAAI{default_year}/index.md) · [IJCAI](venues/IJCAI{default_year}/index.md) · [ACL/EMNLP](venues/ACL_EMNLP{default_year}/index.md) · [ICLR/NeurIPS/ICML](venues/ICLR_NeurIPS_ICML{default_year}/index.md) |",
        f"| **信息系统与数据工程顶刊** | [ACM TOIS](venues/TOIS{default_year}/index.md) · [IEEE TKDE](venues/TKDE{default_year}/index.md) · [ACM TORS](venues/TORS{default_year}/index.md) · [IEEE TMM](venues/TMM{default_year}/index.md) · [ACM TKDD](venues/TKDD{default_year}/index.md) · [ACM TWEB](venues/TWEB{default_year}/index.md) |",
    ])

    (docs_dir / "index.md").write_text("\n".join(home_lines) + "\n", encoding="utf-8")

    # 构建 mkdocs.yml
    topic_nav_items = ["      - 全部推荐细分方向总览: topics/index.md"]
    for cat_slug, cinfo in CATEGORIES.items():
        topic_nav_items.append(f"      - {cinfo['title']}: topics/{cat_slug}.md")

    conf_nav_items = ["      - 全部顶会与顶刊总览: venues/index.md"]
    for venue_code, vinfo in VENUES.items():
        v_folder_name = f"{venue_code}{default_year}"
        conf_nav_items.append(f"      - {vinfo['name']}: venues/{v_folder_name}/index.md")

    mkdocs_content = f"""site_name: RecSys Paper Notes · 推荐系统论文精读库
site_description: 推荐系统与信息检索顶会顶刊论文解读（SIGIR / RecSys / KDD / WWW / ACM MM / TOIS / TKDE），每篇 5 分钟读懂核心思想
site_url: https://{github_user}.github.io/Paper-Notes/
repo_url: https://github.com/{github_user}/Paper-Notes
repo_name: {github_user}/Paper-Notes

theme:
  name: material
  language: zh
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.top
    - search.suggest
    - search.highlight
    - content.code.copy
  palette:
    - scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: 切换至深色模式
    - scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: 切换至浅色模式

plugins:
  - search:
      lang:
        - zh
        - en

markdown_extensions:
  - admonition
  - tables
  - attr_list
  - pymdownx.arithmatex:
      generic: true
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format

extra_javascript:
  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js

nav:
  - 首页总览: index.md
  - 按推荐细分类别浏览 (14大类):
{chr(10).join(topic_nav_items)}
  - 按顶会与顶刊浏览 (22个会刊):
{chr(10).join(conf_nav_items)}
"""
    (repo_root / "mkdocs.yml").write_text(mkdocs_content, encoding="utf-8")
    print(f"[Build] 已生成双维度索引页与 mkdocs.yml（共索引 {len(notes)} 篇论文笔记）。")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    all_notes = collect_all_notes(root / "docs")
    build_topic_pages(root / "docs", all_notes)
    build_venue_pages(root / "docs", all_notes, default_year="2026")
    build_home_and_mkdocs_yml(root, all_notes, github_user="jbdcia321vw", default_year="2026")
