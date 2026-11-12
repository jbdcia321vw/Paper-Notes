#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_nav_and_index.py
扫描 docs/venues/<VENUE>/<YEAR>/<category>/ 下的推荐系统论文笔记，自动生成：

1. 按推荐系统 16 大细分类别聚合的跨会议索引页 (docs/topics/<category>.md)
2. 按会议年份聚合的索引页 (docs/years/<year>.md)
3. 按顶会/顶刊 + 年份 + 分类划分的索引页
   (docs/venues/<VENUE>/index.md, docs/venues/<VENUE>/<YEAR>/index.md, .../<category>/index.md)
4. 全站总览首页 (docs/index.md)
5. 双维度 + 年份导航配置文件 (mkdocs.yml)
"""

import re
from pathlib import Path
from collections import defaultdict

from init_clean_recsys_repo import VENUES, CATEGORIES, SUPPORTED_YEARS

CAT_ICONS = {
    "collaborative_filtering": "🤝",
    "graph_rec": "🕸️",
    "self_supervised_denoising": "🔄",
    "sequential_rec": "⏱️",
    "generative_rec": "✨",
    "llm_rec": "🧠",
    "agent_conversational_rec": "🗣️",
    "multimodal_rec": "🧩",
    "social_group_rec": "👥",
    "cold_start_rec": "❄️",
    "cross_domain_multi_scenario": "🌐",
    "poi_spatiotemporal_rec": "📍",
    "ctr_feature_interaction": "🎯",
    "multi_task_reranking": "⚖️",
    "rl_bandit_rec": "🎮",
    "debias_fairness_industrial": "🛡️",
}

PORTAL_STYLE = """<style>
.recsys-hero {
  text-align: center;
  padding: 2.6rem 1.5rem 2.2rem;
  margin-bottom: 2rem;
  border-radius: 16px;
  background: linear-gradient(135deg, rgba(63,81,181,0.10) 0%, rgba(33,150,243,0.06) 50%, rgba(156,39,176,0.08) 100%);
  border: 1px solid rgba(63,81,181,0.16);
}
.recsys-hero h1 {
  font-size: 2.2rem;
  font-weight: 800;
  margin: 0 0 0.6rem 0;
  letter-spacing: -0.02em;
}
.recsys-hero p {
  font-size: 1.05rem;
  opacity: 0.82;
  max-width: 740px;
  margin: 0 auto 1.4rem;
}
.recsys-stats {
  display: flex;
  justify-content: center;
  gap: 1rem;
  flex-wrap: wrap;
}
.recsys-stat-pill {
  padding: 0.45rem 1.1rem;
  border-radius: 999px;
  background: var(--md-default-bg-color);
  border: 1px solid rgba(128,128,128,0.2);
  font-size: 0.9rem;
  font-weight: 600;
  box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}
.recsys-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 1rem;
  margin: 1.2rem 0 2.2rem;
}
.recsys-card {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: 1.15rem 1.25rem;
  border-radius: 12px;
  background: var(--md-default-bg-color);
  border: 1px solid rgba(128,128,128,0.18);
  text-decoration: none !important;
  color: var(--md-default-fg-color) !important;
  transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}
.recsys-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 8px 24px rgba(0,0,0,0.08);
  border-color: var(--md-primary-fg-color);
}
.recsys-card-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.45rem;
}
.recsys-card-title {
  font-size: 1.02rem;
  font-weight: 700;
}
.recsys-badge {
  font-size: 0.75rem;
  padding: 0.15rem 0.55rem;
  border-radius: 999px;
  background: rgba(63,81,181,0.12);
  color: var(--md-primary-fg-color);
  font-weight: 600;
  white-space: nowrap;
}
.recsys-card-desc {
  font-size: 0.84rem;
  opacity: 0.75;
  line-height: 1.45;
}
</style>"""


def _field(frontmatter: str, name: str) -> str:
    m = re.search(rf'^{name}:\s*["\']?(.*?)["\']?\s*$', frontmatter, re.MULTILINE)
    return m.group(1).strip() if m else ""


def parse_frontmatter(md_path: Path) -> dict:
    """解析单篇论文 Markdown 笔记顶部的 YAML Frontmatter 元数据。"""
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    meta = {
        "title": md_path.stem.replace("_", " ").title(),
        "description": "",
        "date": "",
        "venue": "",
        "year": "",
        "paper_id": "",
        "arxiv_id": "",
        "paper_url": "",
        "code_url": "",
        "categories": [],
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
    meta["title"] = _field(fm, "title").replace("[Paper Note] ", "").strip()
    meta["description"] = _field(fm, "description")
    meta["date"] = _field(fm, "date")
    meta["venue"] = _field(fm, "venue")
    meta["year"] = _field(fm, "year")
    meta["paper_id"] = _field(fm, "paper_id")
    meta["arxiv_id"] = _field(fm, "arxiv_id")
    meta["paper_url"] = _field(fm, "paper_url")
    meta["code_url"] = _field(fm, "code_url")

    cats = re.findall(r"^\s*-\s*([a-z0-9_]+)\s*$", fm, re.MULTILINE)
    inline = re.search(r"^categories:\s*\[(.*?)\]", fm, re.MULTILINE)
    if inline:
        cats += [c.strip().strip("'\"") for c in inline.group(1).split(",")]
    valid_cats = []
    for c in cats:
        if c in CATEGORIES and c not in valid_cats:
            valid_cats.append(c)
    meta["categories"] = valid_cats
    return meta


def md_cell(value: str) -> str:
    """转义 Markdown 表格单元格中的竖线与换行。"""
    return (value or "").replace("|", "\\|").replace("\n", " ").strip()


def format_resource_links(item: dict) -> str:
    """生成论文原文与开源代码的快捷点击链接标签。"""
    links = []
    if item.get("paper_url"):
        links.append(f"[📄 原文]({item['paper_url']})")
    if item.get("code_url"):
        links.append(f"[💻 代码]({item['code_url']})")
    return " · ".join(links) if links else "-"


def parse_note_path(md_file: Path, docs_dir: Path):
    """解析 venues/<VENUE>/<YEAR>/<category>/<slug>.md 路径。"""
    try:
        rel = md_file.relative_to(docs_dir)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) != 5 or parts[0] != "venues":
        return None
    venue, year, category = parts[1], parts[2], parts[3]
    if not re.fullmatch(r"\d{4}", year):
        return None
    if venue not in VENUES or category not in CATEGORIES:
        return None
    return venue, year, category


def collect_all_notes(docs_dir: Path):
    """遍历 docs/venues/<VENUE>/<YEAR>/<category>/ 收集所有论文解读笔记。"""
    venues_dir = docs_dir / "venues"
    notes = []
    if not venues_dir.exists():
        return notes

    for md_file in sorted(venues_dir.rglob("*.md")):
        if md_file.name == "index.md":
            continue
        parsed = parse_note_path(md_file, docs_dir)
        if not parsed:
            print(f"[Warn] 跳过不符合 venues/<VENUE>/<YEAR>/<category>/ 结构的文件: {md_file}")
            continue
        venue, year, category = parsed
        meta = parse_frontmatter(md_file)
        meta["venue"] = venue
        meta["year"] = year
        meta["category"] = category
        if category not in meta["categories"]:
            meta["categories"].insert(0, category)
        meta["rel_from_docs"] = md_file.relative_to(docs_dir).as_posix()
        notes.append(meta)
    return notes


def discover_years(docs_dir: Path, notes: list) -> list:
    """从配置、笔记与目录中发现所有年份，按倒序返回。"""
    years = set(SUPPORTED_YEARS)
    for n in notes:
        if n.get("year"):
            years.add(n["year"])
    venues_dir = docs_dir / "venues"
    if venues_dir.exists():
        for venue_dir in venues_dir.iterdir():
            if not venue_dir.is_dir():
                continue
            for child in venue_dir.iterdir():
                if child.is_dir() and re.fullmatch(r"\d{4}", child.name):
                    years.add(child.name)
    return sorted(years, reverse=True)


def build_topic_pages(docs_dir: Path, notes: list):
    """生成 docs/topics/ 下的 16 大推荐系统细分方向跨会议聚合页。"""
    topics_dir = docs_dir / "topics"
    topics_dir.mkdir(parents=True, exist_ok=True)

    cat_map = defaultdict(list)
    for n in notes:
        for c in n["categories"]:
            if c in CATEGORIES:
                cat_map[c].append(n)

    cards_html = ['<div class="recsys-grid">']
    for cat_slug, cinfo in CATEGORIES.items():
        items = cat_map.get(cat_slug, [])
        icon = CAT_ICONS.get(cat_slug, "📌")
        cards_html.append(
            f'  <a class="recsys-card" href="{cat_slug}/">'
            f'    <div class="recsys-card-head">'
            f'      <span class="recsys-card-title">{icon} {cinfo["title"]}</span>'
            f'      <span class="recsys-badge">{len(items)} 篇</span>'
            f'    </div>'
            f'    <div class="recsys-card-desc">{cinfo["desc"]}</div>'
            f'  </a>'
        )
    cards_html.append("</div>")

    topic_index_lines = [
        "---",
        "hide:",
        "  - toc",
        "---",
        "",
        PORTAL_STYLE,
        "",
        f"# 🔬 按推荐系统细分类别浏览 ({len(CATEGORIES)} 大正交方向)",
        "",
        "按**推荐系统研究子方向**跨会议、跨年份聚合所有论文解读笔记。每篇均附带精读笔记、**论文原文直达链接**与**开源代码仓库链接（如有）**。",
        "",
        "\n".join(cards_html),
    ]

    for cat_slug, cinfo in CATEGORIES.items():
        items = cat_map.get(cat_slug, [])
        icon = CAT_ICONS.get(cat_slug, "📌")
        page_lines = [
            f"# {icon} {cinfo['title']} ({cinfo['en_title']})",
            "",
            f"> **研究范畴**：{cinfo['desc']}",
            "",
            f"当前方向共收录 **{len(items)}** 篇顶会/顶刊论文解读：",
            "",
        ]
        if not items:
            page_lines.append("*暂无论文笔记。*")
        else:
            page_lines.extend([
                "| 会议/期刊 | 年份 | 论文标题 (点击查看 5 分钟精读) | 原文与开源代码 | 核心速览 |",
                "| :---: | :---: | :--- | :---: | :--- |",
            ])
            for item in sorted(items, key=lambda x: (x["venue"], x["year"], x["title"])):
                rel_link = "../" + item["rel_from_docs"]
                desc = md_cell(item["description"] or "点击查看核心思想与实验协议精读")
                res_links = format_resource_links(item)
                page_lines.append(
                    f"| `{item['venue']}` | {item['year']} | [**{md_cell(item['title'])}**]({rel_link}) | {res_links} | {desc} |"
                )

        (topics_dir / f"{cat_slug}.md").write_text("\n".join(page_lines) + "\n", encoding="utf-8")

    (topics_dir / "index.md").write_text("\n".join(topic_index_lines) + "\n", encoding="utf-8")


def build_year_pages(docs_dir: Path, notes: list, years: list):
    """生成 docs/years/<year>.md 跨会刊年份聚合页。"""
    years_dir = docs_dir / "years"
    years_dir.mkdir(parents=True, exist_ok=True)

    year_map = defaultdict(lambda: defaultdict(int))
    for n in notes:
        year_map[n["year"]][n["venue"]] += 1

    cards_html = ['<div class="recsys-grid">']
    for year in years:
        total = sum(year_map[year].values())
        cards_html.append(
            f'  <a class="recsys-card" href="{year}/">'
            f'    <div class="recsys-card-head">'
            f'      <span class="recsys-card-title">🗓️ {year} 年</span>'
            f'      <span class="recsys-badge">{total} 篇</span>'
            f'    </div>'
            f'    <div class="recsys-card-desc">按会议/期刊浏览 {year} 年收录的推荐系统论文解读</div>'
            f'  </a>'
        )
    cards_html.append("</div>")

    index_lines = [
        "---",
        "hide:",
        "  - toc",
        "---",
        "",
        PORTAL_STYLE,
        "",
        "# 🗓️ 按会议年份浏览",
        "",
        "按论文发表年份跨会刊聚合。每个年份页面可继续进入对应会议/期刊的年度目录。",
        "",
        "\n".join(cards_html),
    ]
    (years_dir / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    for year in years:
        total = sum(year_map[year].values())
        venue_cards = ['<div class="recsys-grid">']
        for venue_code, vinfo in VENUES.items():
            count = year_map[year].get(venue_code, 0)
            icon = "🏛️" if vinfo["type"] == "conference" else "📖"
            venue_cards.append(
                f'  <a class="recsys-card" href="../venues/{venue_code}/{year}/">'
                f'    <div class="recsys-card-head">'
                f'      <span class="recsys-card-title">{icon} {vinfo["name"]}</span>'
                f'      <span class="recsys-badge">{count} 篇</span>'
                f'    </div>'
                f'    <div class="recsys-card-desc">{vinfo["tier"]} · {vinfo["full_name"]}</div>'
                f'  </a>'
            )
        venue_cards.append("</div>")
        lines = [
            "---",
            "hide:",
            "  - toc",
            "---",
            "",
            PORTAL_STYLE,
            "",
            f"# 🗓️ {year} 年推荐系统论文解读",
            "",
            f"当前年份共收录 **{total}** 篇论文解读。",
            "",
            "\n".join(venue_cards),
        ]
        (years_dir / f"{year}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_venue_pages(docs_dir: Path, notes: list, years: list):
    """生成 venues/index.md、venues/<VENUE>/index.md、venues/<VENUE>/<YEAR>/index.md 与分类页。"""
    venues_dir = docs_dir / "venues"
    venues_dir.mkdir(parents=True, exist_ok=True)

    venue_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for n in notes:
        venue_map[n["venue"]][n["year"]][n["category"]].append(n)

    venue_cards = ['<div class="recsys-grid">']
    for venue_code, vinfo in VENUES.items():
        total_v = sum(
            len(items)
            for year_items in venue_map[venue_code].values()
            for items in year_items.values()
        )
        icon = "🏛️" if vinfo["type"] == "conference" else "📖"
        venue_cards.append(
            f'  <a class="recsys-card" href="{venue_code}/">'
            f'    <div class="recsys-card-head">'
            f'      <span class="recsys-card-title">{icon} {vinfo["name"]}</span>'
            f'      <span class="recsys-badge">{total_v} 篇</span>'
            f'    </div>'
            f'    <div class="recsys-card-desc">{vinfo["tier"]} · {vinfo["full_name"]}</div>'
            f'  </a>'
        )
    venue_cards.append("</div>")

    venue_index_lines = [
        "---",
        "hide:",
        "  - toc",
        "---",
        "",
        PORTAL_STYLE,
        "",
        "# 🏛️ 按顶会与顶刊浏览 (22 个推荐系统核心会议与期刊)",
        "",
        "\n".join(venue_cards),
    ]
    (venues_dir / "index.md").write_text("\n".join(venue_index_lines) + "\n", encoding="utf-8")

    for venue_code, vinfo in VENUES.items():
        venue_dir = venues_dir / venue_code
        venue_dir.mkdir(parents=True, exist_ok=True)
        icon = "🏛️" if vinfo["type"] == "conference" else "📖"

        year_cards = ['<div class="recsys-grid">']
        for year in years:
            total_y = sum(len(items) for items in venue_map[venue_code][year].values())
            year_cards.append(
                f'  <a class="recsys-card" href="{year}/">'
                f'    <div class="recsys-card-head">'
                f'      <span class="recsys-card-title">🗓️ {year} 年</span>'
                f'      <span class="recsys-badge">{total_y} 篇</span>'
                f'    </div>'
                f'    <div class="recsys-card-desc">{vinfo["name"]} {year} 年收录的推荐系统论文解读</div>'
                f'  </a>'
            )
        year_cards.append("</div>")

        venue_lines = [
            "---",
            "hide:",
            "  - toc",
            "---",
            "",
            PORTAL_STYLE,
            "",
            f"# {icon} {vinfo['name']} 推荐系统论文解读汇总",
            "",
            f"- **英文全称**: {vinfo['full_name']}",
            f"- **级别定位**: {vinfo['tier']}",
            "",
            "## 按年份浏览",
            "",
            "\n".join(year_cards),
        ]
        (venue_dir / "index.md").write_text("\n".join(venue_lines) + "\n", encoding="utf-8")

        for year in years:
            year_dir = venue_dir / year
            year_dir.mkdir(parents=True, exist_ok=True)
            total_y = sum(len(items) for items in venue_map[venue_code][year].values())

            cat_cards = ['<div class="recsys-grid">']
            for cat_slug, cinfo in CATEGORIES.items():
                c_dir = year_dir / cat_slug
                c_dir.mkdir(parents=True, exist_ok=True)
                c_items = venue_map[venue_code][year].get(cat_slug, [])
                c_icon = CAT_ICONS.get(cat_slug, "📌")
                cat_cards.append(
                    f'  <a class="recsys-card" href="{cat_slug}/">'
                    f'    <div class="recsys-card-head">'
                    f'      <span class="recsys-card-title">{c_icon} {cinfo["title"]}</span>'
                    f'      <span class="recsys-badge">{len(c_items)} 篇</span>'
                    f'    </div>'
                    f'    <div class="recsys-card-desc">{cinfo["desc"]}</div>'
                    f'  </a>'
                )
            cat_cards.append("</div>")

            year_lines = [
                "---",
                "hide:",
                "  - toc",
                "---",
                "",
                PORTAL_STYLE,
                "",
                f"# {icon} {vinfo['name']} {year} 推荐系统论文解读",
                "",
                f"- **英文全称**: {vinfo['full_name']}",
                f"- **级别定位**: {vinfo['tier']} · 当前年份共收录 **{total_y}** 篇笔记",
                "",
                "## 按推荐系统细分方向浏览",
                "",
                "\n".join(cat_cards),
            ]
            (year_dir / "index.md").write_text("\n".join(year_lines) + "\n", encoding="utf-8")

            for cat_slug, cinfo in CATEGORIES.items():
                c_dir = year_dir / cat_slug
                c_items = venue_map[venue_code][year].get(cat_slug, [])
                c_icon = CAT_ICONS.get(cat_slug, "📌")
                c_lines = [
                    f"# [{vinfo['name']} {year}] {c_icon} {cinfo['title']}",
                    "",
                    f"> **方向说明**：{cinfo['desc']}",
                    "",
                ]
                if not c_items:
                    c_lines.append("*当前会议在该年份、该细分方向下暂无论文笔记。*")
                else:
                    c_lines.extend([
                        "| 论文标题 (点击查看 5 分钟精读) | 原文与开源代码 | 核心速览 |",
                        "| :--- | :---: | :--- |",
                    ])
                    for item in sorted(c_items, key=lambda x: x["title"]):
                        res_links = format_resource_links(item)
                        c_lines.append(
                            f"| [**{md_cell(item['title'])}**]({item['path'].name}) | {res_links} | {md_cell(item['description'])} |"
                        )
                (c_dir / "index.md").write_text("\n".join(c_lines) + "\n", encoding="utf-8")


def build_home_and_mkdocs_yml(repo_root: Path, notes: list, years: list, github_user: str = "jbdcia321vw"):
    """生成全宽门户卡片式 docs/index.md 与按会刊/年份分组的 mkdocs.yml。"""
    docs_dir = repo_root / "docs"
    cat_counts = defaultdict(int)
    venue_counts = defaultdict(int)
    year_counts = defaultdict(int)
    for n in notes:
        for c in n["categories"]:
            cat_counts[c] += 1
        venue_counts[n["venue"]] += 1
        year_counts[n["year"]] += 1

    topic_cards = ['<div class="recsys-grid">']
    for cat_slug, cinfo in CATEGORIES.items():
        icon = CAT_ICONS.get(cat_slug, "📌")
        topic_cards.append(
            f'  <a class="recsys-card" href="topics/{cat_slug}/">'
            f'    <div class="recsys-card-head">'
            f'      <span class="recsys-card-title">{icon} {cinfo["title"]}</span>'
            f'      <span class="recsys-badge">{cat_counts[cat_slug]} 篇</span>'
            f'    </div>'
            f'    <div class="recsys-card-desc">{cinfo["desc"]}</div>'
            f'  </a>'
        )
    topic_cards.append("</div>")

    year_cards = ['<div class="recsys-grid">']
    for year in years:
        year_cards.append(
            f'  <a class="recsys-card" href="years/{year}/">'
            f'    <div class="recsys-card-head">'
            f'      <span class="recsys-card-title">🗓️ {year} 年</span>'
            f'      <span class="recsys-badge">{year_counts[year]} 篇</span>'
            f'    </div>'
            f'    <div class="recsys-card-desc">按会议/期刊浏览 {year} 年收录的论文解读</div>'
            f'  </a>'
        )
    year_cards.append("</div>")

    venue_cards = ['<div class="recsys-grid">']
    for venue_code, vinfo in VENUES.items():
        icon = "🏛️" if vinfo["type"] == "conference" else "📖"
        venue_cards.append(
            f'  <a class="recsys-card" href="venues/{venue_code}/">'
            f'    <div class="recsys-card-head">'
            f'      <span class="recsys-card-title">{icon} {vinfo["name"]}</span>'
            f'      <span class="recsys-badge">{venue_counts[venue_code]} 篇</span>'
            f'    </div>'
            f'    <div class="recsys-card-desc">{vinfo["tier"]} · {vinfo["full_name"]}</div>'
            f'  </a>'
        )
    venue_cards.append("</div>")

    home_lines = [
        "---",
        "hide:",
        "  - navigation",
        "  - toc",
        "---",
        "",
        PORTAL_STYLE,
        "",
        '<div class="recsys-hero">',
        '  <h1>📚 RecSys Paper Notes</h1>',
        f'  <p>5 分钟读懂一篇推荐系统与信息检索顶会/顶刊论文 · 支持「{len(CATEGORIES)} 大正交细分方向 × {len(VENUES)} 个核心会刊 × 年份」三维检索</p>',
        '  <div class="recsys-stats">',
        f'    <span class="recsys-stat-pill">📖 已收录 {len(notes)} 篇精读笔记</span>',
        f'    <span class="recsys-stat-pill">🔬 {len(CATEGORIES)} 大推荐系统细分方向</span>',
        f'    <span class="recsys-stat-pill">🏛️ {len(VENUES)} 个顶会与顶级期刊</span>',
        f'    <span class="recsys-stat-pill">🗓️ {len(years)} 个年份</span>',
        '  </div>',
        '</div>',
        "",
        f"## 🔬 按推荐系统细分类别浏览 ({len(CATEGORIES)} 大正交方向)",
        "",
        "\n".join(topic_cards),
        "",
        "## 🗓️ 按会议年份浏览",
        "",
        "\n".join(year_cards),
        "",
        "## 🏛️ 按顶会与顶刊浏览 (SIGIR / RecSys / KDD / WWW / ACM MM / TOIS / TKDE)",
        "",
        "\n".join(venue_cards),
    ]
    (docs_dir / "index.md").write_text("\n".join(home_lines) + "\n", encoding="utf-8")

    has_overrides = (repo_root / "overrides").exists()
    custom_dir_line = "  custom_dir: overrides\n" if has_overrides else ""
    logo_line = "  logo: assets/papernotes-logo-horizontal.svg\n" if (docs_dir / "assets" / "papernotes-logo-horizontal.svg").exists() else ""
    favicon_line = "  favicon: assets/favicon.svg\n" if (docs_dir / "assets" / "favicon.svg").exists() else ""

    extra_css_block = ""
    if (docs_dir / "stylesheets" / "extra.css").exists():
        extra_css_block = "extra_css:\n  - stylesheets/extra.css\n"

    js_list = []
    for js_rel in ["javascripts/mathjax.js", "javascripts/search-page.js", "javascripts/title-link.js"]:
        if (docs_dir / js_rel).exists():
            js_list.append(f"  - {js_rel}")
    js_list.append("  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js")
    extra_js_block = "extra_javascript:\n" + "\n".join(js_list) + "\n"

    hooks_list = []
    for h_rel in ["hooks/set_build_date.py", "hooks/tags_order.py"]:
        if (repo_root / h_rel).exists():
            hooks_list.append(f"  - {h_rel}")
    hooks_block = ("hooks:\n" + "\n".join(hooks_list) + "\n") if hooks_list else ""

    topic_nav_items = ["      - 全部推荐细分方向总览: topics/index.md"]
    for cat_slug, cinfo in CATEGORIES.items():
        icon = CAT_ICONS.get(cat_slug, "📌")
        topic_nav_items.append(f'      - "{icon} {cinfo["title"]}": topics/{cat_slug}.md')

    year_nav_items = ["      - 全部年份总览: years/index.md"]
    for year in years:
        year_nav_items.append(f'      - "{year} 年": years/{year}.md')

    conf_nav_items = ["      - 全部顶会与顶刊总览: venues/index.md"]
    for venue_code, vinfo in VENUES.items():
        icon = "🏛️" if vinfo["type"] == "conference" else "📖"
        conf_nav_items.append(f'      - "{icon} {vinfo["name"]}":')
        conf_nav_items.append(f"          - 会刊总览: venues/{venue_code}/index.md")
        for year in years:
            conf_nav_items.append(f'          - "{year}": venues/{venue_code}/{year}/index.md')

    mkdocs_content = f"""site_name: RecSys Paper Notes · 推荐系统论文精读库
site_description: 推荐系统与信息检索顶会顶刊论文解读（SIGIR / RecSys / KDD / WWW / ACM MM / TOIS / TKDE），按会刊与年份分类，每篇 5 分钟读懂核心思想
site_url: https://{github_user}.github.io/Paper-Notes/
repo_url: https://github.com/{github_user}/Paper-Notes
repo_name: {github_user}/Paper-Notes

theme:
  name: material
{custom_dir_line}{logo_line}{favicon_line}  language: zh
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.top
    - navigation.indexes
    - search.suggest
    - search.highlight
    - content.code.copy
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: 切换至深色模式
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
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
  - tags

{hooks_block}{extra_css_block}{extra_js_block}
markdown_extensions:
  - admonition
  - tables
  - attr_list
  - md_in_html
  - pymdownx.details
  - pymdownx.arithmatex:
      generic: true
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format

nav:
  - 🏠 首页总览: index.md
  - 🔬 按推荐细分类别 ({len(CATEGORIES)}大类):
{chr(10).join(topic_nav_items)}
  - 🗓️ 按会议年份浏览:
{chr(10).join(year_nav_items)}
  - 🏛️ 按顶会与顶刊 ({len(VENUES)}个会刊):
{chr(10).join(conf_nav_items)}
"""
    (repo_root / "mkdocs.yml").write_text(mkdocs_content, encoding="utf-8")
    print(f"[Build] 已生成卡片式首页、分类/年份/会刊索引与 mkdocs.yml（共索引 {len(notes)} 篇笔记，{len(years)} 个年份）。")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    docs_dir = root / "docs"
    all_notes = collect_all_notes(docs_dir)
    all_years = discover_years(docs_dir, all_notes)
    build_topic_pages(docs_dir, all_notes)
    build_year_pages(docs_dir, all_notes, all_years)
    build_venue_pages(docs_dir, all_notes, all_years)
    build_home_and_mkdocs_yml(root, all_notes, all_years)
