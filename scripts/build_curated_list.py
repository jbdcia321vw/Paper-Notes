#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_curated_list.py
从 arXiv 检索明确标注为 SIGIR 2026 / RecSys 2026 主会录用的论文，自动过滤 workshop /
challenge / 技术报告，并用 LLM 根据标题与摘要判断推荐相关性、选择 16 大方向主分类，
生成可人工复核的策展 CSV。未配置 API Key 时自动回退到关键词启发式。

用法：
  python scripts/build_curated_list.py --venue SIGIR --year 2026 --max-results 300
  python scripts/build_curated_list.py --venue RecSys --year 2026 --max-results 150
  python scripts/build_curated_list.py --venue SIGIR --year 2026 --no-llm
"""

import csv
import re
import sys
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from init_clean_recsys_repo import CATEGORIES
from generate_recsys_notes import (
    _fetch_arxiv_entries,
    resolve_api_key,
    resolve_base_url,
    resolve_model,
)
from paper_classifier import DEFAULT_CACHE_PATH, classify_items

CATEGORY_KEYWORDS = {
    "collaborative_filtering": ["collaborative filtering", "matrix factorization", "two-tower", "two tower", "item-item", "implicit feedback", "embedding retrieval"],
    "graph_rec": ["graph neural", "gnn", "knowledge graph", "lightgcn", "heterogeneous graph", "graph convolution", "graph-based"],
    "self_supervised_denoising": ["self-supervised", "contrastive learning", "denoising", "noisy feedback", "data augmentation", "implicit feedback"],
    "sequential_rec": ["sequential recommendation", "session-based", "next-item", "sequence modeling", "transformer", "mamba", "long sequence"],
    "generative_rec": ["generative recommendation", "semantic id", "rq-vae", "diffusion recommendation", "autoregressive", "generative retrieval", "tokenized"],
    "llm_rec": ["large language model", "llm", "instruction tuning", "in-context learning", "prompt", "foundation model"],
    "agent_conversational_rec": ["conversational recommendation", "dialogue", "dialogue system", "user simulator", "agent", "multi-agent"],
    "multimodal_rec": ["multimodal", "multi-modal", "visual", "image", "video", "audio", "clip", "vision-language"],
    "social_group_rec": ["social recommendation", "group recommendation", "trust", "social network", "influence"],
    "cold_start_rec": ["cold-start", "cold start", "long-tail", "long tail", "zero-shot", "few-shot"],
    "cross_domain_multi_scenario": ["cross-domain", "cross domain", "multi-domain", "multi-scenario", "transfer learning", "domain adaptation"],
    "poi_spatiotemporal_rec": ["poi", "point-of-interest", "spatio-temporal", "spatiotemporal", "trajectory", "check-in", "location recommendation"],
    "ctr_feature_interaction": ["click-through rate", "ctr", "cvr", "feature interaction", "advertising", "conversion rate", "sponsored"],
    "multi_task_reranking": ["multi-task", "multi task", "re-ranking", "reranking", "listwise", "pareto", "cascade ranking"],
    "rl_bandit_rec": ["reinforcement learning", "bandit", "exploration", "offline rl", "policy learning", "reward"],
    "debias_fairness_industrial": ["bias", "fairness", "debias", "popularity bias", "causal", "privacy", "robustness", "trustworthy"],
}

VENUE_PATTERNS = {
    "SIGIR": {
        "positive": [r"\bsigir[\s-]?2026\b", r"\bsigir\s*'26\b"],
        "negative": [
            "workshop", "challenge", "tutorial", "technical report", "demo",
            "sigir-ap", "reNeuir", "ikat", "evalmg", "agentsearch", "ecommerce challenge",
        ],
    },
    "RecSys": {
        "positive": [r"\brecsys[\s-]?2026\b", r"\brecsys\s*'26\b", r"\bacm recsys 2026\b"],
        "negative": [
            "workshop", "challenge", "tutorial", "technical report", "demo",
            "cars", "usrw", "genaiecommerce", "daquamrec", "oars", "recsys challenge",
        ],
    },
}


def is_main_conference(comment: str, venue: str) -> bool:
    if not comment:
        return False
    c = comment.lower()
    patterns = VENUE_PATTERNS[venue]
    if not any(re.search(p, c, re.IGNORECASE) for p in patterns["positive"]):
        return False
    return not any(neg.lower() in c for neg in patterns["negative"])


RECOMMENDATION_TERMS = [
    "recommend", "recommender", "recommendation", "personaliz", "user modeling",
    "user behavior", "click-through", "click through", "ctr", "cvr", "conversion",
    "collaborative filtering", "cold-start", "cold start", "sequential recommendation",
    "session-based", "item recommendation", "basket", "purchase", "sponsored",
    "advertising", "user interest", "user preference", "item ranking", "candidate ranking",
]

IR_ONLY_HINTS = ["ad-hoc search", "passage retrieval", "document retrieval", "rag poisoning", "dataset search"]
SHORT_TOKENS = {"poi", "ctr", "cvr", "gnn", "llm", "clip", "rag", "rl", "kg"}


def _contains(text: str, keyword: str) -> bool:
    if keyword in SHORT_TOKENS:
        return re.search(rf"\b{re.escape(keyword)}\b", text) is not None
    return keyword in text


def is_recommendation_relevant(title: str, abstract: str) -> bool:
    text = f"{title} {abstract}".lower()
    if any(hint in text for hint in IR_ONLY_HINTS):
        return False
    return any(_contains(text, term) for term in RECOMMENDATION_TERMS)


def classify(title: str, abstract: str) -> str:
    text = f"{title} {abstract}".lower()
    title_l = title.lower()

    ordered_rules = [
        ("agent_conversational_rec", ["conversational recommend", "dialogue", "user simulator", "multi-agent"]),
        ("poi_spatiotemporal_rec", ["poi", "point-of-interest", "spatio-temporal", "spatiotemporal", "trajectory", "check-in", "location recommendation"]),
        ("ctr_feature_interaction", ["click-through rate", "click through rate", "ctr", "cvr", "feature interaction", "sponsored", "advertising", "conversion rate"]),
        ("cross_domain_multi_scenario", ["cross-domain", "cross domain", "multi-domain", "multi-scenario", "domain adaptation"]),
        ("social_group_rec", ["social recommendation", "group recommendation", "social network", "trust"]),
        ("cold_start_rec", ["cold-start", "cold start", "long-tail", "long tail", "zero-shot", "few-shot"]),
        ("multimodal_rec", ["multimodal", "multi-modal", "visual", "image", "video", "audio", "clip"]),
        ("generative_rec", ["generative recommendation", "semantic id", "rq-vae", "diffusion recommendation", "generative retrieval", "autoregressive"]),
        ("sequential_rec", ["sequential recommendation", "session-based", "next-item", "long sequence", "sequence modeling"]),
        ("graph_rec", ["graph neural", "gnn", "knowledge graph", "lightgcn", "heterogeneous graph", "graph-based"]),
        ("self_supervised_denoising", ["self-supervised", "contrastive learning", "denoising", "noisy feedback"]),
        ("rl_bandit_rec", ["reinforcement learning", "bandit", "offline rl", "policy learning"]),
        ("debias_fairness_industrial", ["fairness", "debias", "popularity bias", "privacy", "causal"]),
        ("multi_task_reranking", ["multi-task", "multi task", "re-ranking", "reranking", "listwise", "pareto"]),
        ("llm_rec", ["large language model", "llm", "instruction tuning", "in-context learning"]),
        ("collaborative_filtering", ["collaborative filtering", "matrix factorization", "two-tower", "two tower", "item-item"]),
    ]
    for cat, keywords in ordered_rules:
        if any(_contains(title_l, kw) for kw in keywords):
            return cat
    for cat, keywords in ordered_rules:
        if any(_contains(text, kw) for kw in keywords):
            return cat
    return "collaborative_filtering"


def main():
    parser = argparse.ArgumentParser(description="从 arXiv 构建 SIGIR/RecSys 2026 策展名单")
    parser.add_argument("--venue", default="SIGIR", choices=["SIGIR", "RecSys"])
    parser.add_argument("--year", default="2026")
    parser.add_argument("--max-results", type=int, default=300)
    parser.add_argument("--no-relevance-filter", action="store_true", help="保留所有主会论文（默认仅保留推荐相关论文）")
    parser.add_argument("--out", default="", help="输出 CSV 路径；默认 data/papers/<VENUE><YEAR>.csv")
    parser.add_argument("--no-llm", action="store_true", help="禁用 LLM，仅用关键词启发式分类")
    parser.add_argument("--llm-api-key", default="", help="LLM API Key（或 LLM_API_KEY / DEEPSEEK_API_KEY）")
    parser.add_argument("--llm-base-url", default="", help="LLM API Base URL")
    parser.add_argument("--llm-model", default="", help="LLM 模型名称")
    parser.add_argument("--llm-workers", type=int, default=4, help="LLM 并发线程数（默认 4）")
    parser.add_argument("--llm-retries", type=int, default=2, help="LLM 单篇失败重试次数（默认 2）")
    parser.add_argument("--llm-min-confidence", type=float, default=0.6, help="LLM 排除非推荐论文所需的最低置信度")
    parser.add_argument("--llm-cache", default=str(DEFAULT_CACHE_PATH), help="LLM 分类缓存路径")
    parser.add_argument("--refresh-llm-cache", action="store_true", help="忽略本地 LLM 分类缓存")
    args = parser.parse_args()

    query = f'all:"{args.venue} {args.year}"'
    print(f"[Query] {query} (max_results={args.max_results})")
    papers = _fetch_arxiv_entries(
        {"search_query": query, "max_results": args.max_results, "sortBy": "submittedDate", "sortOrder": "descending"},
        cache_dir=None,
        download_pdf=False,
    )

    candidates = []
    seen = set()
    for p in papers:
        if not is_main_conference(p.get("comment", ""), args.venue):
            continue
        arxiv_id = p.get("arxiv_id", "")
        if not arxiv_id or arxiv_id in seen:
            continue
        seen.add(arxiv_id)
        abstract = p.get("content", "").split("Abstract:\n", 1)[-1]
        if not args.no_relevance_filter and not is_recommendation_relevant(p["title"], abstract):
            continue
        candidates.append((p, abstract))

    llm_results = {}
    llm_removed = 0
    if not args.no_llm:
        api_key = resolve_api_key(args.llm_api_key)
        if not api_key:
            print("[LLM] 未检测到 API Key（LLM_API_KEY / DEEPSEEK_API_KEY），回退到关键词分类。")
        else:
            base_url = resolve_base_url(args.llm_base_url)
            model = resolve_model(args.llm_model)
            print(f"[LLM] 使用 {model} 对 {len(candidates)} 篇候选论文做相关性判断与分类 ...")
            items = [
                {"arxiv_id": p["arxiv_id"], "title": p["title"], "abstract": abstract}
                for p, abstract in candidates
            ]
            llm_results = classify_items(
                items,
                api_key=api_key,
                base_url=base_url,
                model=model,
                workers=args.llm_workers,
                retries=args.llm_retries,
                cache_path=Path(args.llm_cache),
                refresh=args.refresh_llm_cache,
            )
            print(f"[LLM] 完成 {len(llm_results)} 篇。")

    rows = []
    for p, abstract in candidates:
        arxiv_id = p["arxiv_id"]
        result = llm_results.get(arxiv_id)
        if result and result.get("source") != "error":
            if (
                not args.no_relevance_filter
                and result.get("is_recommendation") is False
                and float(result.get("confidence", 0.0)) >= args.llm_min_confidence
            ):
                llm_removed += 1
                print(f"  [LLM 排除] {p['title'][:85]} | {result.get('reason', '')}")
                continue
            category = result.get("category", "")
            if category not in CATEGORIES:
                category = classify(p["title"], abstract)
        else:
            category = classify(p["title"], abstract)
        rows.append({
            "title": p["title"],
            "arxiv_id": arxiv_id,
            "paper_url": f"https://arxiv.org/abs/{arxiv_id}",
            "code_url": p.get("code_url", ""),
            "category": category,
            "venue": args.venue,
            "year": args.year,
        })

    out_path = Path(args.out) if args.out else SCRIPT_DIR.parent / "data" / "papers" / f"{args.venue}{args.year}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "arxiv_id", "paper_url", "code_url", "category", "venue", "year"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Done] 主会候选 {len(rows)} 篇（LLM 排除 {llm_removed} 篇）-> {out_path}")
    for row in rows[:15]:
        print(f"  - [{row['category']}] {row['title'][:90]} ({row['arxiv_id']})")
    if len(rows) > 15:
        print(f"  ... 其余 {len(rows) - 15} 篇见 CSV")


if __name__ == "__main__":
    main()
