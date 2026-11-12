#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_recsys_notes.py
推荐系统论文自动抓取、14 大细分类别自动归类与 5 分钟深度解读生成工具。
支持两种输入模式：
1. 从本地 PDF 文件夹批量解析 (--pdf-dir ./pdfs --venue SIGIR2026)
2. 从 arXiv API 按推荐系统关键词自动检索最新顶会/预印本论文 (--arxiv-query "recommender system" --max-papers 10)
支持所有兼容 OpenAI SDK 格式的大模型接口（DeepSeek-V3 / Gemini 2.5 Flash / GPT-4o-mini 等）。
"""

import os
import re
import json
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import argparse
from pathlib import Path
from datetime import datetime
from init_clean_recsys_repo import VENUES, CATEGORIES

CATEGORY_LIST_STR = "\n".join(
    [f"- `{slug}`: {info['title']} ({info['en_title']}) — {info['desc']}" for slug, info in CATEGORIES.items()]
)

SYSTEM_PROMPT = f"""你是一位资深的推荐系统（Recommender Systems）与信息检索（Information Retrieval）领域研究员。
请对输入的学术论文进行深度拆解，输出符合 Paper-Notes 规范的中文 Markdown 论文解读（5分钟读懂核心思想）。

首先，请从以下 14 个推荐系统细分方向中，为该论文选择 **1 个主分类 (primary_category)** 和 **最多 2 个交叉分类 (secondary_categories)**：
{CATEGORY_LIST_STR}

你的输出必须严格包含顶部的 YAML Frontmatter（不要用代码块包裹整个输出），格式如下：
---
title: "[Paper Note] 论文英文原名"
description: "[会议或期刊名][主分类中文名] 一句话概括核心方法与关键收益（80字以内）"
tags:
  - 会议或期刊名
  - 主分类中文名
  - 关键技术词1
  - 关键技术词2
date: {datetime.now().strftime('%Y-%m-%d')}
venue: 会议目录名(如SIGIR2026)
categories:
  - 主分类英文slug(必须严格属于上述14个slug之一)
  - 次分类英文slug(可选)
---

# 论文英文原名 (中文译名)

- **会议/期刊**: 会议或期刊名
- **研究方向**: 主分类中文名 / 次分类中文名
- **论文链接**: 论文 URL 或 arXiv 链接

## TL;DR (核心速览)
用 150 字以内精炼概括：针对推荐系统的什么痛点（如冷启动、长序列兴趣漂移、语义ID碰撞、曝光偏差等）？提出了什么核心架构或损失函数？在哪些基准数据集或线上 A/B 测试中取得了什么提升？

## Background & Motivation (背景与动机)
- **领域背景**: 当前该推荐子方向的主流范式是什么？
- **现有局限 (Limitations of Prior Work)**: 现有方法在表征能力、训练效率、稀疏性或工业落地中存在哪 2-3 个具体矛盾？
- **核心洞察 (Key Insight)**: 作者观察到了什么关键现象或理论性质，从而引出本文的解法？

## Method (方法架构)
### 1. 整体流水线 (Overall Architecture)
详细说明用户特征、物品多模态/ID特征、历史交互序列如何输入模型，经过哪些模块，最终如何计算推荐得分或生成候选物品。并附上一段合法的 Mermaid 流程图：

```mermaid
flowchart TD
    A["输入层: 用户历史序列 / 物品特征"] --> B["核心编码模块"]
    B --> C["预测/生成层"]
```

### 2. 核心模块设计 (Key Designs)
分 2-3 点拆解核心创新模块（包含关键数学公式、损失函数设计 $\\mathcal{{L}}$ 及其背后的物理意义）。

## Key Experimental Results (核心实验结果)
提取论文中的主实验对比表（Recall@K, NDCG@K, AUC, LogLoss 等）与消融实验结论，使用 Markdown 表格呈现。

## Highlights & Limitations (亮点、启发与局限)
- **亮点与可迁移性**: 哪些设计技巧可以直接迁移到其他推荐任务或工业场景？
- **局限性与未来方向**: 计算开销、超参敏感度或潜在假设限制。

## Rating (四维评分)
- **创新性 (Novelty)**: ⭐⭐⭐⭐
- **实验扎实度 (Experiments)**: ⭐⭐⭐⭐
- **工业落地潜力 (Industrial Value)**: ⭐⭐⭐⭐
"""


def extract_text_from_pdf(pdf_path: Path, max_chars: int = 45000) -> str:
    """使用 pypdf 提取本地 PDF 文本。"""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        pages_text = []
        for page in reader.pages:
            pages_text.append(page.extract_text() or "")
        full_text = "\n".join(pages_text)
        return full_text[:max_chars]
    except Exception as e:
        print(f"[Warn] 读取 PDF 失败 {pdf_path}: {e}")
        return ""


def fetch_arxiv_papers(query: str, max_results: int = 5):
    """通过 arXiv API 检索推荐系统相关论文摘要与元数据。"""
    encoded_q = urllib.parse.quote(f"all:{query} AND (cat:cs.IR OR cat:cs.AI OR cat:cs.LG)")
    url = f"http://export.arxiv.org/api/query?search_query={encoded_q}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    req = urllib.request.Request(url, headers={"User-Agent": "RecSys-Paper-Notes/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        xml_data = resp.read()

    root = ET.fromstring(xml_data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.find("atom:title", ns).text or "").strip().replace("\n", " ")
        summary = (entry.find("atom:summary", ns).text or "").strip()
        link = (entry.find("atom:id", ns).text or "").strip()
        published = (entry.find("atom:published", ns).text or "")[:10]
        comment_el = entry.find("{http://arxiv.org/schemas/atom}comment")
        comment = comment_el.text.strip() if comment_el is not None and comment_el.text else ""
        papers.append({
            "title": title,
            "content": f"Title: {title}\nPublished: {published}\nComment/Venue: {comment}\URL: {link}\n\nAbstract:\n{summary}",
            "comment": comment,
        })
    return papers


def call_llm_api(paper_text: str, venue: str, api_key: str, base_url: str, model: str) -> str:
    """调用兼容 OpenAI 格式的大模型接口生成结构化 Markdown 笔记。"""
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"目标所属会议/期刊目录标识：`{venue}`\n\n请解读以下推荐系统论文内容：\n\n{paper_text}"},
        ],
        "temperature": 0.2,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        res_json = json.loads(resp.read().decode("utf-8"))
    return res_json["choices"][0]["message"]["content"].strip()


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title.lower()).strip("_")
    return slug[:75] or f"paper_{int(time.time())}"


def extract_primary_category(md_content: str) -> str:
    m = re.search(r"categories:\s*\n\s*-\s*([a-z0-9_]+)", md_content)
    if m and m.group(1) in CATEGORIES:
        return m.group(1)
    return "collaborative_filtering"


def main():
    parser = argparse.ArgumentParser(description="批量生成推荐系统论文解读并自动按会议与14大类别归档")
    parser.add_argument("--venue", default="SIGIR2026", help="目标会议/期刊目录名，如 SIGIR2026, RecSys2026, TOIS2026, TKDE2026")
    parser.add_argument("--pdf-dir", default="", help="本地 PDF 论文文件夹路径")
    parser.add_argument("--arxiv-query", default="", help="arXiv 搜索关键词，如 'generative recommendation' 或 'multimodal recommendation'")
    parser.add_argument("--max-papers", type=int, default=5, help="单次最大处理篇数")
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", ""), help="大模型 API Key (或设置环境变量 LLM_API_KEY)")
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1"), help="API Base URL")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "deepseek-chat"), help="模型名称，如 deepseek-chat, gemini-2.5-flash, gpt-4o-mini")
    args = parser.parse_args()

    if not args.api_key:
        print("[Error] 请提供 --api-key 或设置环境变量 LLM_API_KEY")
        return

    repo_root = Path(__file__).resolve().parent.parent
    docs_dir = repo_root / "docs"

    tasks = []
    if args.pdf_dir:
        pdf_folder = Path(args.pdf_dir)
        for p in sorted(pdf_folder.glob("*.pdf"))[: args.max_papers]:
            txt = extract_text_from_pdf(p)
            if txt:
                tasks.append({"title": p.stem, "content": txt})
    elif args.arxiv_query:
        tasks = fetch_arxiv_papers(args.arxiv_query, max_results=args.max_papers)
    else:
        print("[Info] 请指定 --pdf-dir 或 --arxiv-query。")
        return

    for idx, item in enumerate(tasks, 1):
        print(f"[{idx}/{len(tasks)}] 正在生成论文解读: {item['title']} ...")
        md_note = call_llm_api(item["content"], args.venue, args.api_key, args.base_url, args.model)
        primary_cat = extract_primary_category(md_note)
        target_dir = docs_dir / "venues" / args.venue / primary_cat
        target_dir.mkdir(parents=True, exist_ok=True)
        out_file = target_dir / f"{slugify(item['title'])}.md"
        out_file.write_text(md_note + "\n", encoding="utf-8")
        print(f"  -> 已归档至: {out_file.relative_to(repo_root)} (主类别: {primary_cat})")

    # 自动重建双维度索引与 mkdocs.yml
    from build_nav_and_index import collect_all_notes, build_topic_pages, build_venue_pages, build_home_and_mkdocs_yml
    all_notes = collect_all_notes(docs_dir)
    build_topic_pages(docs_dir, all_notes)
    build_venue_pages(docs_dir, all_notes)
    build_home_and_mkdocs_yml(repo_root, all_notes)


if __name__ == "__main__":
    main()
