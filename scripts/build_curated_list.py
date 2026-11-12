#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_curated_list.py (v3.0 全量 22 会刊支持 + DBLP/arXiv 双通道自动发现 + 增量合并版)
支持从 arXiv Comment 与 DBLP 官方录用清单自动发现全部 22 个推荐系统相关顶会/顶刊论文，
自动过滤 workshop / challenge / tutorial / demo，并用 LLM 判断推荐系统相关性与 16 大方向分类，
支持增量合并 (--incremental，默认开启，保留已有人工修正的 CSV 行，仅对新发现论文调用 LLM)。

用法：
  python scripts/build_curated_list.py --venue SIGIR --year 2026 --max-results 300
  python scripts/build_curated_list.py --venue KDD --year 2026 --use-dblp
  python scripts/build_curated_list.py --venue TOIS --year 2026 --use-dblp
"""

import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from init_clean_recsys_repo import VENUES, CATEGORIES
from generate_recsys_notes import (
    _fetch_arxiv_entries,
    print_usage_summary,
    resolve_api_key,
    resolve_base_url,
    resolve_model,
)
from paper_classifier import DEFAULT_CACHE_PATH, classify_items
from run_logging import RunLog, format_duration

COMMON_NEGATIVE_HINTS = [
    "workshop", "challenge", "tutorial", "technical report", "demo", "doctoral consortium",
    "companion", "poster", "short paper", "extended abstract", "keynote",
]

# 支持全部 22 个顶会与顶刊的 arXiv 检索词、正则匹配与 DBLP 检索映射
VENUE_CONFIGS = {
    "SIGIR": {"queries": ['"SIGIR {year}"'], "pos": [r"\bsigir[\s'-]?{year}\b", r"\bsigir\s*'{yy}\b"], "neg": ["sigir-ap", "reneuir", "ikat", "evalmg", "agentsearch"], "dblp": "SIGIR"},
    "RecSys": {"queries": ['"RecSys {year}"'], "pos": [r"\brecsys[\s'-]?{year}\b", r"\brecsys\s*'{yy}\b"], "neg": ["cars", "usrw", "genaiecommerce", "daquamrec", "oars"], "dblp": "RecSys"},
    "WSDM": {"queries": ['"WSDM {year}"'], "pos": [r"\bwsdm[\s'-]?{year}\b", r"\bwsdm\s*'{yy}\b"], "neg": [], "dblp": "WSDM"},
    "CIKM": {"queries": ['"CIKM {year}"'], "pos": [r"\bcikm[\s'-]?{year}\b", r"\bcikm\s*'{yy}\b"], "neg": [], "dblp": "CIKM"},
    "ECIR": {"queries": ['"ECIR {year}"'], "pos": [r"\becir[\s'-]?{year}\b", r"\becir\s*'{yy}\b"], "neg": [], "dblp": "ECIR"},
    "SIGIR-AP": {"queries": ['"SIGIR-AP {year}"', '"SIGIR AP {year}"'], "pos": [r"\bsigir[\s-]*ap[\s'-]?{year}\b"], "neg": [], "dblp": "SIGIR-AP"},
    "KDD": {"queries": ['"KDD {year}"', '"SIGKDD {year}"'], "pos": [r"\b(?:sig)?kdd[\s'-]?{year}\b", r"\bkdd\s*'{yy}\b"], "neg": ["kdd cup"], "dblp": "KDD"},
    "WWW": {"queries": ['"WWW {year}"', '"TheWebConf {year}"', '"WebConf {year}"'], "pos": [r"\b(?:www|thewebconf|webconf)[\s'-]?{year}\b", r"\bwww\s*'{yy}\b"], "neg": [], "dblp": "WWW"},
    "ACM_MM": {"queries": ['"ACM MM {year}"', '"ACM Multimedia {year}"'], "pos": [r"\bacm\s*(?:mm|multimedia)[\s'-]?{year}\b", r"\bmm\s*'{yy}\b"], "neg": [], "dblp": "ACM Multimedia"},
    "ICDE": {"queries": ['"ICDE {year}"'], "pos": [r"\bicde[\s'-]?{year}\b"], "neg": [], "dblp": "ICDE"},
    "ICDM": {"queries": ['"ICDM {year}"'], "pos": [r"\bicdm[\s'-]?{year}\b"], "neg": [], "dblp": "ICDM"},
    "DASFAA": {"queries": ['"DASFAA {year}"'], "pos": [r"\bdasfaa[\s'-]?{year}\b"], "neg": [], "dblp": "DASFAA"},
    "AAAI": {"queries": ['"AAAI {year}" AND (recommend OR recommender OR retrieval)'], "pos": [r"\baaai[\s'-]?{year}\b", r"\baaai\s*'{yy}\b"], "neg": [], "dblp": "AAAI"},
    "IJCAI": {"queries": ['"IJCAI {year}" AND (recommend OR recommender OR retrieval)'], "pos": [r"\bijcai[\s'-]?{year}\b", r"\bijcai\s*'{yy}\b"], "neg": [], "dblp": "IJCAI"},
    "ACL_EMNLP": {"queries": ['"ACL {year}" AND recommend', '"EMNLP {year}" AND recommend'], "pos": [r"\b(?:acl|emnlp)[\s'-]?{year}\b"], "neg": ["findings"], "dblp": "ACL"},
    "ICLR_NeurIPS_ICML": {"queries": ['"ICLR {year}" AND recommend', '"NeurIPS {year}" AND recommend', '"ICML {year}" AND recommend'], "pos": [r"\b(?:iclr|neurips|nips|icml)[\s'-]?{year}\b"], "neg": [], "dblp": "ICLR"},
    "TOIS": {"queries": ['"TOIS" AND recommend', '"Transactions on Information Systems" AND recommend'], "pos": [r"\btois\b", r"transactions on information systems"], "neg": [], "dblp": "ACM Trans. Inf. Syst."},
    "TKDE": {"queries": ['"TKDE" AND recommend', '"Transactions on Knowledge and Data Engineering" AND recommend'], "pos": [r"\btkde\b", r"transactions on knowledge and data engineering"], "neg": [], "dblp": "IEEE Trans. Knowl. Data Eng."},
    "TORS": {"queries": ['"TORS" AND recommend', '"Transactions on Recommender Systems"'], "pos": [r"\btors\b", r"transactions on recommender systems"], "neg": [], "dblp": "ACM Trans. Recomm. Syst."},
    "TMM": {"queries": ['"TMM" AND recommend', '"Transactions on Multimedia" AND recommend'], "pos": [r"\btmm\b", r"transactions on multimedia"], "neg": [], "dblp": "IEEE Trans. Multimedia"},
    "TKDD": {"queries": ['"TKDD" AND recommend', '"Transactions on Knowledge Discovery from Data" AND recommend'], "pos": [r"\btkdd\b", r"transactions on knowledge discovery from data"], "neg": [], "dblp": "ACM Trans. Knowl. Discov. Data"},
    "TWEB": {"queries": ['"TWEB" AND recommend', '"Transactions on the Web" AND recommend'], "pos": [r"\btweb\b", r"transactions on the web"], "neg": [], "dblp": "ACM Trans. Web"},
}


def is_main_conference(comment: str, venue: str, year: str, published_date: str = "") -> bool:
    if not comment:
        return False
    c = comment.lower()
    cfg = VENUE_CONFIGS.get(venue)
    if not cfg:
        return venue.lower() in c
    yy = year[-2:] if len(year) == 4 else year
    pos_list = [p.format(year=year, yy=yy) for p in cfg["pos"]]
    if not any(re.search(p, c, re.IGNORECASE) for p in pos_list):
        return False
    # 期刊如果 comment 未写年份，用 arXiv 发布年份辅助核验
    if venue in {"TOIS", "TKDE", "TORS", "TMM", "TKDD", "TWEB"} and year not in c:
        if published_date and not published_date.startswith(year):
            return False
    neg_list = COMMON_NEGATIVE_HINTS + cfg.get("neg", [])
    return not any(neg.lower() in c for neg in neg_list)


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


def discover_via_dblp_and_resolve_arxiv(venue: str, year: str, max_hits: int = 120) -> list:
    """从 DBLP API 获取该会刊官方发表的推荐相关论文标题，并自动去 arXiv 反查匹配的预印本。
    解决很多作者被 SIGIR/KDD/WWW/TOIS/TKDE 录用后未更新 arXiv comment 导致漏检的问题。
    """
    cfg = VENUE_CONFIGS.get(venue, {})
    dblp_venue = cfg.get("dblp", venue)
    q = f"{dblp_venue} {year} recommend"
    url = f"https://dblp.org/search/publ/api?q={urllib.parse.quote(q)}&h={max_hits}&format=json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "RecSys-Paper-Notes/3.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        hits = data.get("result", {}).get("hits", {}).get("hit", [])
    except Exception as e:
        print(f"  [Warn] DBLP 检索跳过 ({venue} {year}): {e}")
        return []

    resolved_papers = []
    for h in hits:
        info = h.get("info", {})
        title = re.sub(r"\.$", "", (info.get("title") or "").strip())
        if not title or len(title) < 15:
            continue
        # 在 arXiv 按精确标题查询
        clean_t = re.sub(r"[^a-zA-Z0-9\s]", " ", title).strip()
        try:
            entries = _fetch_arxiv_entries(
                {"search_query": f'ti:"{clean_t}"', "max_results": 2},
                cache_dir=None,
                download_pdf=False,
            )
            for ep in entries:
                norm_a = re.sub(r"[^a-z0-9]", "", title.lower())
                norm_b = re.sub(r"[^a-z0-9]", "", ep["title"].lower())
                if norm_a and norm_a == norm_b:
                    ep["comment"] = ep.get("comment") or f"Accepted at {venue} {year} (DBLP matched)"
                    resolved_papers.append(ep)
                    break
            time.sleep(0.35)
        except Exception:
            continue
    return resolved_papers


def main():
    parser = argparse.ArgumentParser(description="从 arXiv + DBLP 自动构建 22 个推荐系统会刊策展名单")
    parser.add_argument("--venue", default="SIGIR", choices=sorted(VENUES.keys()), help="目标会刊 key")
    parser.add_argument("--year", default="2026")
    parser.add_argument("--max-results", type=int, default=300)
    parser.add_argument("--use-dblp", action="store_true", help="同时查询 DBLP 官方录用列表并反查 arXiv 预印本（召回率更高）")
    parser.add_argument("--no-incremental", action="store_true", help="不合并已有 CSV，从零重建名单（默认开启增量合并）")
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
    run_start = time.monotonic()
    log = RunLog("build-list")
    log.write(
        f"START venue={args.venue} year={args.year} max_results={args.max_results} "
        f"use_dblp={args.use_dblp} no_llm={args.no_llm} workers={args.llm_workers}"
    )

    out_path = Path(args.out) if args.out else SCRIPT_DIR.parent / "data" / "papers" / f"{args.venue}{args.year}.csv"
    existing_rows_by_id = {}
    if out_path.exists() and not args.no_incremental:
        with out_path.open("r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                aid = re.sub(r"v\d+$", "", (r.get("arxiv_id") or "").strip())
                if aid:
                    existing_rows_by_id[aid] = r
        print(f"[Incremental] 已加载现有 CSV 中的 {len(existing_rows_by_id)} 篇论文记录。")

    cfg = VENUE_CONFIGS.get(args.venue, {"queries": [f'"{args.venue} {args.year}"']})
    papers = []
    for q_tmpl in cfg["queries"]:
        q_str = f"all:{q_tmpl.format(year=args.year)}"
        print(f"[Query] {q_str} (max_results={args.max_results})")
        log.write(f"QUERY {q_str} max_results={args.max_results}")
        papers.extend(
            _fetch_arxiv_entries(
                {"search_query": q_str, "max_results": args.max_results, "sortBy": "submittedDate", "sortOrder": "descending"},
                cache_dir=None,
                download_pdf=False,
            )
        )

    if args.use_dblp:
        print(f"[DBLP] 正在检索 DBLP 官方目录并反查 arXiv 预印本 ({args.venue} {args.year}) ...")
        dblp_found = discover_via_dblp_and_resolve_arxiv(args.venue, args.year)
        print(f"[DBLP] 额外匹配到 {len(dblp_found)} 篇 arXiv 预印本。")
        papers.extend(dblp_found)

    candidates = []
    seen = set(existing_rows_by_id.keys())
    for p in papers:
        pub_date = ""
        m_pub = re.search(r"Published Date:\s*(\d{4}-\d{2}-\d{2})", p.get("content", ""))
        if m_pub:
            pub_date = m_pub.group(1)
        if not is_main_conference(p.get("comment", ""), args.venue, args.year, pub_date):
            continue
        arxiv_id = re.sub(r"v\d+$", "", p.get("arxiv_id", "").strip())
        if not arxiv_id or arxiv_id in seen:
            continue
        seen.add(arxiv_id)
        p["arxiv_id"] = arxiv_id
        abstract = p.get("content", "").split("Abstract:\n", 1)[-1]
        if not args.no_relevance_filter and not is_recommendation_relevant(p["title"], abstract):
            continue
        candidates.append((p, abstract))

    llm_results = {}
    llm_removed = 0
    if not args.no_llm and candidates:
        api_key = resolve_api_key(args.llm_api_key)
        if not api_key:
            print("[LLM] 未检测到 API Key（LLM_API_KEY / DEEPSEEK_API_KEY），回退到关键词分类。")
        else:
            base_url = resolve_base_url(args.llm_base_url)
            model = resolve_model(args.llm_model)
            print(f"[LLM] 使用 {model} 对 {len(candidates)} 篇新发现候选论文做相关性判断与分类 ...")
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
                log=log,
            )
            print(f"[LLM] 完成 {len(llm_results)} 篇。")

    new_rows = []
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
        new_rows.append({
            "title": p["title"],
            "arxiv_id": arxiv_id,
            "paper_url": f"https://arxiv.org/abs/{arxiv_id}",
            "code_url": p.get("code_url", ""),
            "category": category,
            "venue": args.venue,
            "year": args.year,
        })

    final_rows = list(existing_rows_by_id.values()) + new_rows
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "arxiv_id", "paper_url", "code_url", "category", "venue", "year"])
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"[Done] 策展名单共 {len(final_rows)} 篇（本次新增 {len(new_rows)} 篇，LLM 排除 {llm_removed} 篇）-> {out_path}")
    for row in new_rows[:15]:
        print(f"  + [{row['category']}] {row['title'][:90]} ({row['arxiv_id']})")
    elapsed = time.monotonic() - run_start
    print_usage_summary(
        resolve_model(args.llm_model),
        processed=len(candidates),
        success=len(new_rows),
        elapsed_seconds=elapsed,
        log=log,
    )
    print(f"[Time] 候选名单总耗时 {format_duration(elapsed)}")
    print(f"[Log] 详细日志: {log.path}")
    log.write(f"TOTAL elapsed={format_duration(elapsed)} new_candidates={len(candidates)} total_rows={len(final_rows)}")
    log.close()


if __name__ == "__main__":
    main()