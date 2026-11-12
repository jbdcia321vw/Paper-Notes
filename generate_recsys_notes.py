#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_recsys_notes.py (Pro 全文精读与自动归档版)
支持三种填充模式：
  1. 本地 PDF 文件夹批量全文精读：
     python3 scripts/generate_recsys_notes.py --pdf-dir ./my_pdfs --venue SIGIR2026
  2. 按关键词/会议从 arXiv 自动检索 + 自动下载 PDF 全文精读（含公式与实验表格）：
     python3 scripts/generate_recsys_notes.py --venue SIGIR2026 --arxiv-query "generative recommendation" --max-papers 5
  3. 一键遍历 14 大推荐系统细分方向各抓取 N 篇代表作填充全站 (--seed-all-categories)：
     python3 scripts/generate_recsys_notes.py --seed-all-categories --max-papers 2 --venue RecSys2026

默认支持断点续传（已生成的论文自动跳过，不重复消耗 Token），支持多线程并发 (--workers 3)，
生成完毕后自动调用 build_nav_and_index.py 更新全站卡片计数与 mkdocs.yml。
"""

import os
import re
import sys
import json
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 确保能导入同目录下的配置与构建器
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from init_clean_recsys_repo import VENUES, CATEGORIES

# 14 大推荐系统细分类别的代表性检索词（用于 --seed-all-categories 一键填充全库）
CATEGORY_SEED_QUERIES = {
    "collaborative_filtering": "collaborative filtering representation learning recommendation",
    "sequential_rec": "sequential recommendation session-based recommendation transformer",
    "generative_rec": "generative recommendation semantic ID RQ-VAE diffusion recommendation",
    "llm_agent_rec": "large language model recommendation LLM4Rec agent recommendation",
    "multimodal_rec": "multimodal recommendation visual acoustic graph recommendation",
    "graph_rec": "graph neural network recommendation knowledge graph LightGCN",
    "social_rec": "social recommendation group recommendation trust network",
    "cold_start_cross_domain": "cold-start recommendation cross-domain recommendation multi-scenario",
    "ctr_multi_task_ranking": "click-through rate prediction CTR multi-task recommendation ranking",
    "conversational_explainable": "conversational recommender system explainable recommendation",
    "rl_bandit_rec": "reinforcement learning recommendation contextual bandit long-term engagement",
    "debias_fairness_trust": "debiasing recommendation popularity bias causal recommendation fairness",
    "poi_spatiotemporal_rec": "next POI recommendation point-of-interest spatio-temporal trajectory",
    "industrial_system_eval": "industrial recommender system large-scale retrieval ranking A/B testing",
}

CATEGORY_LIST_STR = "\n".join(
    [f"- `{slug}`: {info['title']} ({info['en_title']}) — {info['desc']}" for slug, info in CATEGORIES.items()]
)

SYSTEM_PROMPT = f"""你是一位资深的推荐系统（Recommender Systems）与信息检索（Information Retrieval）领域研究员。
请对输入的学术论文全文进行深度拆解，输出符合 Paper-Notes 规范的中文 Markdown 论文解读（5分钟读懂核心思想）。

首先，请从以下 14 个推荐系统细分方向中，为该论文选择 **1 个最核心的主分类 (primary_category)** 和 **1-2 个交叉分类 (secondary_categories)**：
{CATEGORY_LIST_STR}

你的输出必须直接以 YAML Frontmatter 开头（严禁在最外层包裹 ```markdown 代码块），严格遵循以下模板：
---
title: "[Paper Note] 论文英文原名"
description: "[会议或期刊名][主分类中文名] 一句话概括核心方法与关键收益（80字以内）"
tags:
  - 会议或期刊名
  - 主分类中文名
  - 核心方法词1
  - 核心方法词2
date: {datetime.now().strftime('%Y-%m-%d')}
venue: 会议目录名(如SIGIR2026)
categories:
  - 主分类英文slug(必须严格属于上述14个slug之一)
  - 次分类英文slug(可选，严格属于上述14个slug之一)
---

# 论文英文原名

- **会议/期刊**: 会议或期刊名
- **研究方向**: 主分类中文名 / 次分类中文名
- **论文链接**: 论文 URL 或 arXiv 链接

## TL;DR (核心速览)

用 150 字以内精炼概括：针对推荐系统的什么核心痛点（如冷启动、长序列兴趣漂移、语义ID碰撞、曝光偏差、多模态特征噪声等）？提出了什么核心架构或损失函数？在哪些基准数据集或线上 A/B 测试中取得了多少提升？

## Background & Motivation (背景与动机)

- **领域背景**: 当前该推荐子方向的主流范式是什么？
- **现有局限 (Limitations of Prior Work)**: 现有方法在表征能力、训练效率、稀疏性或工业落地中存在哪 2-3 个具体矛盾？
- **核心挑战 (Key Challenge)**: 为什么直接结合或简单改进无法解决上述矛盾？
- **核心洞察 (Key Insight)**: 作者观察到了什么关键现象或理论性质，从而引出本文的解法？

## Method (方法架构)

### 1. 整体流水线 (Overall Architecture)

详细说明用户特征、物品多模态/ID特征、历史交互序列如何输入模型，经过哪些模块，最终如何计算推荐得分或自回归生成候选物品。并附上一段合法的 Mermaid 架构流程图：

```mermaid
flowchart TD
    A["输入层: 用户历史序列 / 物品特征"] --> B["核心编码与对齐模块"]
    B --> C["预测打分 / 生成式解码层"]
```

### 2. 核心模块设计 (Key Designs)

分 2-3 点深入拆解本文的核心创新模块（必须包含关键数学公式、损失函数设计 $\\mathcal{{L}}$ 及其背后的设计直觉）。

## Key Experimental Results (核心实验结果)

提取论文中的**真实实验对比数据**（如 Recall@K, NDCG@K, Hit@K, AUC, LogLoss 等）与**消融实验 (Ablation Study)** 结果，使用标准 Markdown 表格呈现（如果原文中包含具体数值，请务必准确提取真实数值，切勿编造）。

## Highlights & Limitations (亮点、启发与局限)

- **核心亮点与可迁移性**: 哪些设计技巧可以直接迁移到其他推荐任务或工业场景？
- **局限性与未来方向**: 计算开销、超参敏感度或潜在假设限制。

## Rating (四维评分)

- **创新性 (Novelty)**: ⭐⭐⭐⭐
- **实验扎实度 (Experiments)**: ⭐⭐⭐⭐
- **写作清晰度 (Writing)**: ⭐⭐⭐⭐
- **工业落地价值 (Value)**: ⭐⭐⭐⭐
"""


def extract_text_from_pdf(pdf_path: Path, max_chars: int = 48000) -> str:
    """使用 pypdf 提取 PDF 论文全文文本（包含公式上下文与实验数据表）。"""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        pages_text = []
        for page in reader.pages:
            pages_text.append(page.extract_text() or "")
        full_text = "\n".join(pages_text)
        return full_text[:max_chars]
    except Exception as e:
        print(f"  [Warn] 解析 PDF 失败 ({pdf_path.name}): {e}")
        return ""


def infer_venue_from_comment(comment: str, default_venue: str) -> str:
    """如果 arXiv comment 中明确提到了 SIGIR/RecSys/KDD/WWW/ACM MM/TOIS/TKDE 等顶会，自动匹配对应会刊目录。"""
    if not comment:
        return default_venue
    c_upper = comment.upper()
    mapping = [
        ("SIGIR-AP", "SIGIR-AP2026"),
        ("SIGIR", "SIGIR2026"),
        ("RECSYS", "RecSys2026"),
        ("WSDM", "WSDM2026"),
        ("CIKM", "CIKM2026"),
        ("ECIR", "ECIR2026"),
        ("KDD", "KDD2026"),
        ("WWW", "WWW2026"),
        ("THEWEBCONF", "WWW2026"),
        ("ACM MM", "ACM_MM2026"),
        ("MULTIMEDIA", "ACM_MM2026"),
        ("ICDE", "ICDE2026"),
        ("ICDM", "ICDM2026"),
        ("DASFAA", "DASFAA2026"),
        ("TOIS", "TOIS2026"),
        ("TKDE", "TKDE2026"),
        ("TORS", "TORS2026"),
        ("TMM", "TMM2026"),
        ("TKDD", "TKDD2026"),
        ("TWEB", "TWEB2026"),
        ("AAAI", "AAAI2026"),
        ("IJCAI", "IJCAI2026"),
        ("ACL", "ACL_EMNLP2026"),
        ("EMNLP", "ACL_EMNLP2026"),
        ("ICLR", "ICLR_NeurIPS_ICML2026"),
        ("NEURIPS", "ICLR_NeurIPS_ICML2026"),
        ("ICML", "ICLR_NeurIPS_ICML2026"),
    ]
    for kw, v_dir in mapping:
        if kw in c_upper:
            return v_dir
    return default_venue


def fetch_arxiv_papers(query: str, max_results: int = 5, cache_dir: Path = None, download_pdf: bool = True):
    """通过 arXiv API 检索推荐系统论文，并自动下载 PDF 提取全文（以便提取真实实验表格和公式）。"""
    encoded_q = urllib.parse.quote(f"all:({query}) AND (cat:cs.IR OR cat:cs.AI OR cat:cs.LG)")
    url = f"http://export.arxiv.org/api/query?search_query={encoded_q}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    req = urllib.request.Request(url, headers={"User-Agent": "RecSys-Paper-Notes/2.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        xml_data = resp.read()

    root = ET.fromstring(xml_data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for entry in root.findall("atom:entry", ns):
        title = re.sub(r"\s+", " ", (entry.find("atom:title", ns).text or "").strip())
        summary = (entry.find("atom:summary", ns).text or "").strip()
        abs_link = (entry.find("atom:id", ns).text or "").strip()
        arxiv_id = abs_link.rstrip("/").split("/")[-1]
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        published = (entry.find("atom:published", ns).text or "")[:10]
        comment_el = entry.find("{http://arxiv.org/schemas/atom}comment")
        comment = comment_el.text.strip() if comment_el is not None and comment_el.text else ""

        full_body = ""
        if download_pdf and cache_dir:
            pdf_file = cache_dir / f"{re.sub(r'[^a-zA-Z0-9._-]', '_', arxiv_id)}.pdf"
            if not pdf_file.exists():
                try:
                    pdf_req = urllib.request.Request(pdf_url, headers={"User-Agent": "RecSys-Paper-Notes/2.0"})
                    with urllib.request.urlopen(pdf_req, timeout=45) as p_resp:
                        pdf_file.write_bytes(p_resp.read())
                except Exception as e:
                    print(f"  [Warn] 下载 arXiv PDF 失败 ({arxiv_id})，将回退使用摘要: {e}")
            if pdf_file.exists():
                full_body = extract_text_from_pdf(pdf_file)

        content_str = (
            f"Title: {title}\n"
            f"Published Date: {published}\n"
            f"arXiv URL: {abs_link}\n"
            f"Author Comment / Venue Note: {comment}\n\n"
            f"Abstract:\n{summary}\n\n"
        )
        if full_body:
            content_str += f"Full Paper Text Extracted from PDF:\n{full_body}"

        papers.append({
            "title": title,
            "content": content_str,
            "comment": comment,
            "url": abs_link,
        })
    return papers


def call_llm_api(paper_text: str, venue: str, api_key: str, base_url: str, model: str, forced_cat: str = "") -> str:
    """调用兼容 OpenAI 格式的大模型接口生成结构化 Markdown 笔记。"""
    endpoint = base_url.rstrip("/") + "/chat/completions"
    cat_hint = f"\n优先归入推荐系统分类标识：`{forced_cat}`" if forced_cat else ""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"目标所属会议/期刊目录标识：`{venue}`{cat_hint}\n\n请深入解读以下推荐系统论文：\n\n{paper_text}",
            },
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
    with urllib.request.urlopen(req, timeout=180) as resp:
        res_json = json.loads(resp.read().decode("utf-8"))
    raw = res_json["choices"][0]["message"]["content"].strip()
    # 移除可能包裹在最外层的 ```markdown 围栏
    raw = re.sub(r"^```(?:markdown|yaml)?\s*\n", "", raw)
    raw = re.sub(r"\n```\s*$", "", raw)
    return raw.strip()


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title.lower()).strip("_")
    return slug[:75] or f"paper_{int(time.time())}"


def extract_primary_category(md_content: str, fallback: str = "collaborative_filtering") -> str:
    m = re.search(r"categories:\s*\n\s*-\s*([a-z0-9_]+)", md_content)
    if m and m.group(1) in CATEGORIES:
        return m.group(1)
    return fallback


def process_single_paper(item: dict, default_venue: str, docs_dir: Path, args) -> str:
    slug = slugify(item["title"])
    venue = infer_venue_from_comment(item.get("comment", ""), default_venue)
    forced_cat = item.get("forced_cat", "")

    # 检查是否已存在同名文件（断点续传）
    existing = list((docs_dir / "venues").rglob(f"{slug}.md"))
    if existing and not args.overwrite:
        return f"[Skip] 已存在跳过: {existing[0].relative_to(docs_dir.parent)}"

    md_note = call_llm_api(item["content"], venue, args.api_key, args.base_url, args.model, forced_cat=forced_cat)
    primary_cat = extract_primary_category(md_note, fallback=forced_cat or "collaborative_filtering")
    target_dir = docs_dir / "venues" / venue / primary_cat
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / f"{slug}.md"
    out_file.write_text(md_note + "\n", encoding="utf-8")
    return f"[OK] {item['title'][:55]}... -> venues/{venue}/{primary_cat}/{slug}.md"


def main():
    parser = argparse.ArgumentParser(description="批量生成推荐系统论文全文精读笔记并自动归档至双维度目录")
    parser.add_argument("--venue", default="SIGIR2026", help="默认目标会议/期刊目录名，如 SIGIR2026, RecSys2026, TOIS2026, TKDE2026")
    parser.add_argument("--pdf-dir", default="", help="本地 PDF 论文文件夹路径")
    parser.add_argument("--arxiv-query", default="", help="arXiv 搜索关键词，如 'generative recommendation' 或 'SIGIR recommendation'")
    parser.add_argument("--seed-all-categories", action="store_true", help="一键遍历 14 大推荐系统细分方向自动从 arXiv 拉取代表作填充全库")
    parser.add_argument("--max-papers", type=int, default=3, help="每个查询或文件夹的最大处理篇数（默认 3）")
    parser.add_argument("--workers", type=int, default=2, help="并发调用 LLM 的线程数（默认 2）")
    parser.add_argument("--no-pdf-download", action="store_true", help="arXiv 模式下仅使用摘要，不下载 PDF 全文")
    parser.add_argument("--overwrite", action="store_true", help="强制覆盖已存在的同名笔记")
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", ""), help="大模型 API Key (或设置环境变量 LLM_API_KEY)")
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1"), help="API Base URL")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "deepseek-chat"), help="模型名称，如 deepseek-chat, gemini-2.5-flash, gpt-4o-mini")
    args = parser.parse_args()

    if not args.api_key:
        print("[Error] 未检测到 API Key！请传入 --api-key 或在终端设置环境变量：$env:LLM_API_KEY='sk-...'")
        return

    repo_root = SCRIPT_DIR.parent
    docs_dir = repo_root / "docs"
    cache_dir = repo_root / ".cache_pdfs"

    tasks = []
    if args.pdf_dir:
        pdf_folder = Path(args.pdf_dir)
        for p in sorted(pdf_folder.glob("*.pdf"))[: args.max_papers]:
            print(f"[PDF] 正在提取本地 PDF 全文: {p.name} ...")
            txt = extract_text_from_pdf(p)
            if txt:
                tasks.append({"title": p.stem, "content": f"Title: {p.stem}\n\n{txt}", "comment": ""})
    elif args.seed_all_categories:
        print(f"[Seed] 正在遍历 14 大推荐系统细分方向，每个方向从 arXiv 获取 {args.max_papers} 篇论文...")
        for cat_slug, q in CATEGORY_SEED_QUERIES.items():
            print(f"  -> 检索方向 [{cat_slug}]: {q}")
            fetched = fetch_arxiv_papers(q, max_results=args.max_papers, cache_dir=cache_dir, download_pdf=not args.no_pdf_download)
            for item in fetched:
                item["forced_cat"] = cat_slug
                tasks.append(item)
            time.sleep(1)
    elif args.arxiv_query:
        print(f"[arXiv] 正在检索 '{args.arxiv_query}' (上限 {args.max_papers} 篇)...")
        tasks = fetch_arxiv_papers(args.arxiv_query, max_results=args.max_papers, cache_dir=cache_dir, download_pdf=not args.no_pdf_download)
    else:
        print("[Info] 请指定以下任一参数开始填充论文：\n  1. --seed-all-categories (一键为 14 大推荐方向自动填充代表性论文)\n  2. --arxiv-query \"关键词\"\n  3. --pdf-dir ./本地PDF目录")
        return

    print(f"\n[Start] 共收集 {len(tasks)} 篇待处理论文，启用 {args.workers} 个并发线程调用 {args.model} ...\n")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(process_single_paper, item, args.venue, docs_dir, args): item for item in tasks}
        for idx, fut in enumerate(as_completed(future_map), 1):
            item = future_map[fut]
            try:
                msg = fut.result()
                print(f"[{idx}/{len(tasks)}] {msg}")
            except Exception as e:
                print(f"[{idx}/{len(tasks)}] [Error] 处理失败 ({item['title'][:45]}): {e}")

    print("\n[Index] 正在重新构建双维度导航索引与卡片计数...")
    from build_nav_and_index import collect_all_notes, build_topic_pages, build_venue_pages, build_home_and_mkdocs_yml
    all_notes = collect_all_notes(docs_dir)
    build_topic_pages(docs_dir, all_notes)
    build_venue_pages(docs_dir, all_notes)
    build_home_and_mkdocs_yml(repo_root, all_notes)
    print(f"[Done] 全库现已收录 {len(all_notes)} 篇推荐系统论文解读！")


if __name__ == "__main__":
    main()