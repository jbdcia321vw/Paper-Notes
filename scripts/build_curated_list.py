#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
build_curated_list.py (v3.12 arXiv 防 429 限流版：跳过已收录标题 + 本地标题反查缓存 + 3.1s 礼貌间隔)
核心特性与修复点：
  1. [P0-4 修复] CSV 增量合并同时兼容 arXiv 行与非 arXiv (DOI/URL/标题) 行，绝不丢失非 arXiv 记录。
  2. [P0-5 修复] 仅当单会刊运行时才响应 `--out`，多会刊 (`--venues` / `--all-venues`) 强制按 `<VENUE><YEAR>.csv` 独立落盘。
  3. [P0-6 修复] 严格保证 `<VENUE><YEAR>.csv` 仅收录 `--year` 当年论文，彻底移除跨年份混写逻辑。
  4. [P0-7 修复] 22 会刊主循环具备异常隔离，单个会刊网络/API 异常不会中断后续会刊执行。
  5. [P1-1 修复] OpenAlex 支持分页抓取、HTTP 429 指数退避重试、空来源严格过滤（防误收），反查上限提升至 150 篇。
  6. [P1-2 修复] 修复 DBLP 单条命中时 `hit` 返回 `dict` 而非 `list` 导致静默丢结果的 Bug。
  7. [P1-3 修复] 合并会刊 `ACL_EMNLP`（覆盖 ACL / EMNLP / NAACL）与 `ICLR_NeurIPS_ICML`（覆盖 ICLR / NeurIPS / ICML）在 DBLP 与 OpenAlex 中遍历全部子会议别名。
  8. [P1-4 修复] 期刊无年份 comment 判定时校验初版提交年份 `published_date >= year - 1`，严防早年旧论文因修订更新被误判为当年新论文。

用法：
  python scripts/build_curated_list.py --venue ACM_MM --year 2026 --use-dblp
  python scripts/build_curated_list.py --all-venues --year 2026 --use-dblp
"""

import argparse
import csv
import hashlib
import http.cookiejar
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from init_clean_recsys_repo import VENUES, CATEGORIES
from generate_recsys_notes import (
    _fetch_arxiv_entries,
    fetch_arxiv_by_ids,
    print_usage_summary,
    resolve_api_key,
    resolve_base_url,
    resolve_model,
    slugify,
)
from paper_classifier import DEFAULT_CACHE_PATH, classify_items
from run_logging import RunLog, format_duration

COMMON_NEGATIVE_HINTS = [
    "workshop", "challenge", "tutorial", "technical report", "demo",
    "doctoral consortium", "companion", "extended abstract", "keynote",
    "submitted to", "under review", "in submission", "preprint submitted",
    "submitted for", "currently under review", "withdrawn", "rejected",
    "not accepted", "anonymous submission", "double-blind",
]

DBLP_MIRRORS = [
    "https://dblp.org",
    "https://dblp.uni-trier.de",
    "https://dblp.dagstuhl.de",
]

REC_ARXIV_FILTER = "(all:recommendation OR all:recommender OR all:recommend OR all:collaborative OR all:personaliz OR all:sequential)"

# DBLP 官方记录 key 前缀映射（100% 精准区分 22 个主会与顶刊，杜绝同名 Workshop 或其他会议串台）
DBLP_KEY_PREFIXES = {
    "SIGIR": ("conf/sigir/",),
    "RecSys": ("conf/recsys/",),
    "WSDM": ("conf/wsdm/",),
    "CIKM": ("conf/cikm/",),
    "ECIR": ("conf/ecir/",),
    "SIGIR-AP": ("conf/sigir-ap/",),
    "KDD": ("conf/kdd/",),
    "WWW": ("conf/www/",),
    "ACM_MM": ("conf/mm/",),
    "ICDE": ("conf/icde/",),
    "ICDM": ("conf/icdm/",),
    "DASFAA": ("conf/dasfaa/",),
    "AAAI": ("conf/aaai/",),
    "IJCAI": ("conf/ijcai/",),
    "ACL_EMNLP": ("conf/acl/", "conf/emnlp/", "conf/naacl/"),
    "ICLR_NeurIPS_ICML": ("conf/iclr/", "conf/nips/", "conf/neurips/", "conf/icml/"),
    "TOIS": ("journals/tois/",),
    "TKDE": ("journals/tkde/",),
    "TORS": ("journals/tors/",),
    "TMM": ("journals/tmm/",),
    "TKDD": ("journals/tkdd/",),
    "TWEB": ("journals/tweb/",),
}

JOURNAL_VENUES = {"TOIS", "TKDE", "TORS", "TMM", "TKDD", "TWEB"}

# 22 个推荐系统相关顶会与顶刊的高召回检索配置（支持多别名列表）
VENUE_CONFIGS = {
    "SIGIR": {
        "queries": [
            'all:"SIGIR {year}"',
            f'(co:SIGIR OR jr:SIGIR) AND {REC_ARXIV_FILTER}',
        ],
        "tokens": [r"\bsigir\b"],
        "neg": ["sigir-ap", "sigir ap", "reneuir", "ikat", "evalmg", "agentsearch"],
        "dblp": ["SIGIR"],
        "openalex": ["SIGIR"],
    },
    "RecSys": {
        "queries": [
            'all:"RecSys {year}"',
            'co:RecSys OR jr:RecSys',
        ],
        "tokens": [r"\brecsys\b", r"recommender systems conference"],
        "neg": ["cars", "usrw", "genaiecommerce", "daquamrec", "oars", "recsys challenge"],
        "dblp": ["RecSys"],
        "openalex": ["RecSys", "ACM Conference on Recommender Systems"],
    },
    "WSDM": {
        "queries": [
            'all:"WSDM {year}"',
            f'(co:WSDM OR jr:WSDM) AND {REC_ARXIV_FILTER}',
        ],
        "tokens": [r"\bwsdm\b", r"web search and data mining"],
        "neg": ["wsdm cup"],
        "dblp": ["WSDM"],
        "openalex": ["WSDM", "Web Search and Data Mining"],
    },
    "CIKM": {
        "queries": [
            'all:"CIKM {year}"',
            f'(co:CIKM OR jr:CIKM) AND {REC_ARXIV_FILTER}',
        ],
        "tokens": [r"\bcikm\b", r"information and knowledge management"],
        "neg": [],
        "dblp": ["CIKM"],
        "openalex": ["CIKM", "Information and Knowledge Management"],
    },
    "ECIR": {
        "queries": [
            'all:"ECIR {year}"',
            f'(co:ECIR OR jr:ECIR) AND {REC_ARXIV_FILTER}',
            f'all:"European Conference on Information Retrieval" AND {REC_ARXIV_FILTER}',
        ],
        "tokens": [r"\becir\b", r"european conference on information retrieval"],
        "neg": [],
        "dblp": ["ECIR"],
        "openalex": ["ECIR", "European Conference on Information Retrieval"],
    },
    "SIGIR-AP": {
        "queries": [
            'all:"SIGIR-AP"',
            'all:"SIGIR AP"',
            'co:"SIGIR-AP" OR co:"SIGIR AP"',
        ],
        "tokens": [r"\bsigir[\s-]*ap\b", r"sigir asia pacific"],
        "neg": [],
        "dblp": ["SIGIR-AP"],
        "openalex": ["SIGIR-AP"],
    },
    "KDD": {
        "queries": [
            'all:"KDD {year}"',
            'all:"SIGKDD {year}"',
            f'(co:KDD OR co:SIGKDD OR jr:KDD) AND {REC_ARXIV_FILTER}',
        ],
        "tokens": [r"\b(?:sig)?kdd\b", r"knowledge discovery and data mining"],
        "neg": ["kdd cup"],
        "dblp": ["KDD"],
        "openalex": ["KDD", "SIGKDD", "Knowledge Discovery and Data Mining"],
    },
    "WWW": {
        "queries": [
            'all:"WWW {year}"',
            'all:"TheWebConf {year}"',
            'all:"WebConf {year}"',
            f'(co:WWW OR co:WebConf OR co:TheWebConf OR co:"Web Conference") AND {REC_ARXIV_FILTER}',
        ],
        "tokens": [r"\b(?:www|thewebconf|webconf)\b", r"the web conference", r"world wide web conference"],
        "neg": [],
        "dblp": ["WWW"],
        "openalex": ["The Web Conference", "WWW", "World Wide Web Conference"],
    },
    "ACM_MM": {
        "queries": [
            'all:"ACM MM {year}"',
            'all:"ACM Multimedia {year}"',
            f'(co:"ACM MM" OR co:"ACM Multimedia" OR co:Multimedia) AND {REC_ARXIV_FILTER}',
        ],
        "tokens": [r"\bacm\s*(?:mm|multimedia)\b", r"\bmm\s*['-]?\s*(?:20\d{2}|\d{2})\b", r"acm international conference on multimedia"],
        "neg": [],
        "dblp": ["ACM Multimedia"],
        "openalex": ["ACM Multimedia", "ACM International Conference on Multimedia"],
    },
    "ICDE": {
        "queries": [
            'all:"ICDE {year}"',
            f'(co:ICDE OR jr:ICDE) AND {REC_ARXIV_FILTER}',
            f'all:"Data Engineering" AND co:ICDE AND {REC_ARXIV_FILTER}',
        ],
        "tokens": [r"\bicde\b", r"international conference on data engineering"],
        "neg": [],
        "dblp": ["ICDE"],
        "openalex": ["ICDE", "International Conference on Data Engineering"],
    },
    "ICDM": {
        "queries": [
            'all:"ICDM {year}"',
            f'(co:ICDM OR jr:ICDM) AND {REC_ARXIV_FILTER}',
            f'all:"Data Mining" AND co:ICDM AND {REC_ARXIV_FILTER}',
        ],
        "tokens": [r"\bicdm\b", r"international conference on data mining"],
        "neg": [],
        "dblp": ["ICDM"],
        "openalex": ["ICDM", "IEEE International Conference on Data Mining"],
    },
    "DASFAA": {
        "queries": [
            'all:"DASFAA {year}"',
            f'(co:DASFAA OR jr:DASFAA) AND {REC_ARXIV_FILTER}',
            'co:DASFAA OR jr:DASFAA',
        ],
        "tokens": [r"\bdasfaa\b", r"database systems for advanced applications"],
        "neg": [],
        "dblp": ["DASFAA"],
        "openalex": ["DASFAA", "Database Systems for Advanced Applications"],
    },
    "AAAI": {
        "queries": [
            f'all:"AAAI {{year}}" AND {REC_ARXIV_FILTER}',
            f'(co:AAAI OR jr:AAAI) AND {REC_ARXIV_FILTER}',
        ],
        "tokens": [r"\baaai\b"],
        "neg": [],
        "dblp": ["AAAI"],
        "openalex": ["AAAI", "AAAI Conference on Artificial Intelligence"],
    },
    "IJCAI": {
        "queries": [
            f'all:"IJCAI {{year}}" AND {REC_ARXIV_FILTER}',
            f'(co:IJCAI OR jr:IJCAI) AND {REC_ARXIV_FILTER}',
        ],
        "tokens": [r"\bijcai\b"],
        "neg": [],
        "dblp": ["IJCAI"],
        "openalex": ["IJCAI", "International Joint Conference on Artificial Intelligence"],
    },
    "ACL_EMNLP": {
        "queries": [
            f'(co:ACL OR co:EMNLP OR co:NAACL OR jr:ACL OR jr:EMNLP) AND {REC_ARXIV_FILTER}',
            f'all:"ACL {{year}}" AND {REC_ARXIV_FILTER}',
            f'all:"EMNLP {{year}}" AND {REC_ARXIV_FILTER}',
        ],
        "tokens": [r"\b(?:acl|emnlp|naacl)\b", r"association for computational linguistics", r"empirical methods in natural language processing"],
        "neg": [],
        "dblp": ["ACL", "EMNLP", "NAACL"],
        "openalex": ["ACL", "EMNLP", "NAACL", "Annual Meeting of the Association for Computational Linguistics", "Empirical Methods in Natural Language Processing"],
    },
    "ICLR_NeurIPS_ICML": {
        "queries": [
            f'(co:ICLR OR co:NeurIPS OR co:ICML OR jr:ICLR OR jr:NeurIPS OR jr:ICML) AND {REC_ARXIV_FILTER}',
            f'all:"ICLR {{year}}" AND {REC_ARXIV_FILTER}',
            f'all:"NeurIPS {{year}}" AND {REC_ARXIV_FILTER}',
            f'all:"ICML {{year}}" AND {REC_ARXIV_FILTER}',
        ],
        "tokens": [r"\b(?:iclr|neurips|nips|icml)\b", r"learning representations", r"neural information processing systems", r"international conference on machine learning"],
        "neg": [],
        "dblp": ["ICLR", "NeurIPS", "ICML"],
        "openalex": ["ICLR", "NeurIPS", "ICML", "International Conference on Learning Representations", "Neural Information Processing Systems", "International Conference on Machine Learning"],
    },
    "TOIS": {
        "queries": [
            f'(co:TOIS OR jr:TOIS OR all:"Transactions on Information Systems") AND {REC_ARXIV_FILTER}',
            'co:TOIS OR jr:TOIS',
        ],
        "tokens": [r"\btois\b", r"transactions on information systems", r"acm trans\.?\s*inf\.?\s*syst"],
        "neg": [],
        "dblp": ["ACM Trans. Inf. Syst."],
        "openalex": ["ACM Transactions on Information Systems"],
    },
    "TKDE": {
        "queries": [
            f'(co:TKDE OR jr:TKDE OR all:"Transactions on Knowledge and Data Engineering") AND {REC_ARXIV_FILTER}',
            'co:TKDE OR jr:TKDE',
        ],
        "tokens": [r"\btkde\b", r"transactions on knowledge and data engineering", r"ieee trans\.?\s*knowl\.?\s*data\s*eng"],
        "neg": [],
        "dblp": ["IEEE Trans. Knowl. Data Eng."],
        "openalex": ["IEEE Transactions on Knowledge and Data Engineering"],
    },
    "TORS": {
        "queries": [
            'co:TORS OR jr:TORS OR all:"Transactions on Recommender Systems"',
            'all:"ACM TORS"',
        ],
        "tokens": [r"\btors\b", r"transactions on recommender systems", r"acm trans\.?\s*recomm\.?\s*syst"],
        "neg": [],
        "dblp": ["ACM Trans. Recomm. Syst."],
        "openalex": ["ACM Transactions on Recommender Systems"],
    },
    "TMM": {
        "queries": [
            f'(co:TMM OR jr:TMM OR all:"Transactions on Multimedia") AND {REC_ARXIV_FILTER}',
        ],
        "tokens": [r"\btmm\b", r"transactions on multimedia", r"ieee trans\.?\s*multimedia"],
        "neg": [],
        "dblp": ["IEEE Trans. Multimedia"],
        "openalex": ["IEEE Transactions on Multimedia"],
    },
    "TKDD": {
        "queries": [
            f'(co:TKDD OR jr:TKDD OR all:"Transactions on Knowledge Discovery from Data") AND {REC_ARXIV_FILTER}',
            'co:TKDD OR jr:TKDD',
        ],
        "tokens": [r"\btkdd\b", r"transactions on knowledge discovery from data", r"acm trans\.?\s*knowl\.?\s*discov\.?\s*data"],
        "neg": [],
        "dblp": ["ACM Trans. Knowl. Discov. Data"],
        "openalex": ["ACM Transactions on Knowledge Discovery from Data"],
    },
    "TWEB": {
        "queries": [
            f'(co:TWEB OR jr:TWEB OR all:"Transactions on the Web") AND {REC_ARXIV_FILTER}',
            'co:TWEB OR jr:TWEB',
        ],
        "tokens": [r"\btweb\b", r"transactions on the web", r"acm trans\.?\s*web"],
        "neg": [],
        "dblp": ["ACM Trans. Web"],
        "openalex": ["ACM Transactions on the Web"],
    },
}


def is_target_venue_and_year(
    comment: str,
    venue: str,
    target_year: str,
    published_date: str = "",
    updated_date: str = "",
) -> bool:
    """严格检查 comment/journal_ref 是否属于指定主会/顶刊及目标年份 target_year。"""
    if not comment:
        return False
    c = comment.lower()
    cfg = VENUE_CONFIGS.get(venue)
    if not cfg:
        return venue.lower() in c and target_year in c

    neg_list = COMMON_NEGATIVE_HINTS + cfg.get("neg", [])
    if any(neg.lower() in c for neg in neg_list):
        return False

    tokens = cfg.get("tokens", [])
    if not any(re.search(pat, c, re.IGNORECASE) for pat in tokens):
        return False

    yy = target_year[-2:] if len(target_year) == 4 else target_year
    # 1. 检查 comment 中是否明确带有目标年份（如 2026 / '26 / -26）
    if re.search(rf"\b{target_year}\b", c) or re.search(rf"(?:['\-\s]){yy}\b", c):
        return True

    # 2. 若 comment 中明确写有其他年份（如 2023/2024/2025），直接拒绝
    explicit_years_in_c = re.findall(r"\b(20\d{2})\b", c)
    if explicit_years_in_c:
        return False

    # 3. P1-4 修复：如果是期刊（TOIS/TKDE/TORS/TMM/TKDD/TWEB）且 comment 未写年份，
    # 要求 published_date 或 updated_date 为 target_year，且首次提交年份 published_date >= target_year - 1（严防旧论文修订误判为今年新发）
    if venue in JOURNAL_VENUES:
        upd_y = (updated_date or "")[:4]
        pub_y = (published_date or "")[:4]
        if pub_y == target_year:
            return True
        if upd_y == target_year and pub_y.isdigit() and target_year.isdigit():
            if int(pub_y) >= int(target_year) - 1:
                return True

    return False


RECOMMENDATION_TERMS = [
    "recommend", "recommender", "recommendation", "personaliz", "user modeling",
    "user behavior", "click-through", "click through", "ctr", "cvr", "conversion",
    "collaborative filtering", "cold-start", "cold start", "sequential recommendation",
    "session-based", "item recommendation", "basket", "purchase", "sponsored",
    "advertising", "user interest", "user preference", "item ranking", "candidate ranking",
    "next-item", "point-of-interest", "matrix factorization", "two-tower",
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


def batch_resolve_titles_on_arxiv(
    titles: list,
    venue: str,
    year: str,
    batch_size: int = 12,
    ee_map: dict = None,
    skip_norm_titles: set = None,
) -> list:
    """将从 DBLP/OpenAlex 获得的官方录用论文标题按 12 篇一批在 arXiv 上反查预印本：
    1. 自动跳过已在当前 CSV 中的标题（避免重复请求 arXiv 触发 429）；
    2. 使用本地持久化缓存 `data/dblp_cache/arxiv_title_resolve_cache.json`，查过的标题永不重复请求 arXiv；
    3. 严格遵守 arXiv API 官方 3 秒 1 次的礼貌间隔规范。"""
    if not titles:
        return []
    skip_norm_titles = skip_norm_titles or set()
    cache_dir = SCRIPT_DIR.parent / "data" / "dblp_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    resolve_cache_file = cache_dir / "arxiv_title_resolve_cache.json"
    resolve_cache = {}
    if resolve_cache_file.exists():
        try:
            resolve_cache = json.loads(resolve_cache_file.read_text(encoding="utf-8"))
        except Exception:
            resolve_cache = {}

    resolved = []
    seen_ids = set()
    norm_map = {}
    clean_queries = []

    for raw_t in titles:
        norm_t = re.sub(r"[^a-z0-9]", "", raw_t.lower())
        if len(norm_t) < 12 or norm_t in norm_map or norm_t in skip_norm_titles:
            continue
        if norm_t in resolve_cache:
            cached_ep = resolve_cache[norm_t]
            if isinstance(cached_ep, dict) and cached_ep.get("arxiv_id"):
                aid = cached_ep["arxiv_id"]
                if aid not in seen_ids:
                    seen_ids.add(aid)
                    ep_copy = dict(cached_ep)
                    ep_copy["comment"] = f"{ep_copy.get('comment', '')} | Accepted at {venue} {year} (DBLP Official Verified)".strip(" |")
                    ep_copy["trusted_venue_match"] = True
                    if ee_map and norm_t in ee_map and ee_map[norm_t]:
                        ep_copy["official_ee"] = ee_map[norm_t]
                    resolved.append(ep_copy)
            continue
        norm_map[norm_t] = raw_t
        clean_t = re.sub(r"[^a-zA-Z0-9\s]", " ", raw_t)
        clean_t = re.sub(r"\s+", " ", clean_t).strip()
        if clean_t:
            clean_queries.append((norm_t, clean_t))

    if not clean_queries:
        return resolved

    print(f"  [arXiv Resolve] 共 {len(clean_queries)} 篇未缓存新标题待反查（每批 {batch_size} 篇，遵守 arXiv 3s 礼貌间隔）...")
    cache_dirty = False
    for start in range(0, len(clean_queries), batch_size):
        chunk = clean_queries[start : start + batch_size]
        or_clauses = " OR ".join(f'ti:"{item[1][:90]}"' for item in chunk)
        matched_norms_in_chunk = set()
        try:
            entries = _fetch_arxiv_entries(
                {"search_query": or_clauses, "max_results": len(chunk) * 2},
                cache_dir=None,
                download_pdf=False,
            )
            for ep in entries:
                aid = re.sub(r"v\d+$", "", ep.get("arxiv_id", "").strip())
                if not aid:
                    continue
                norm_ep = re.sub(r"[^a-z0-9]", "", (ep.get("title") or "").lower())
                if not norm_ep:
                    continue
                matched_key = norm_ep if norm_ep in norm_map else None
                if not matched_key:
                    for target_norm in norm_map:
                        if len(target_norm) >= 25 and (target_norm in norm_ep or norm_ep in target_norm):
                            matched_key = target_norm
                            break
                if matched_key:
                    matched_norms_in_chunk.add(matched_key)
                    resolve_cache[matched_key] = {
                        "title": ep.get("title", ""),
                        "content": ep.get("content", ""),
                        "comment": ep.get("comment", ""),
                        "published_date": ep.get("published_date", ""),
                        "updated_date": ep.get("updated_date", ""),
                        "paper_url": ep.get("paper_url", ""),
                        "code_url": ep.get("code_url", ""),
                        "arxiv_id": aid,
                    }
                    cache_dirty = True
                    if aid not in seen_ids:
                        seen_ids.add(aid)
                        orig_c = ep.get("comment") or ""
                        ep["comment"] = f"{orig_c} | Accepted at {venue} {year} (DBLP Official Verified)".strip(" |")
                        ep["trusted_venue_match"] = True
                        if ee_map and matched_key in ee_map and ee_map[matched_key]:
                            ep["official_ee"] = ee_map[matched_key]
                        resolved.append(ep)
            # 对本批次确认未在 arXiv 找到预印本的标题记录为 None，下次不再重复查询
            for norm_k, _ in chunk:
                if norm_k not in matched_norms_in_chunk and norm_k not in resolve_cache:
                    resolve_cache[norm_k] = None
                    cache_dirty = True
            time.sleep(3.1)
        except Exception as e:
            print(f"  [Warn] 批量反查 arXiv 跳过当前批次: {e}")
            time.sleep(5.0)
            continue

    if cache_dirty:
        try:
            resolve_cache_file.write_text(json.dumps(resolve_cache, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
    return resolved


# 全局 CookieJar，用于持久化保存通过 Anubis PoW 验证后的 Cookie
_COOKIE_JAR = http.cookiejar.CookieJar()

# 22 个会刊在 DBLP RDF Knowledge Graph (https://sparql.dblp.org/sparql) 中的官方 Stream URI
DBLP_SPARQL_STREAMS = {
    "SIGIR": ["https://dblp.org/streams/conf/sigir"],
    "RecSys": ["https://dblp.org/streams/conf/recsys"],
    "WSDM": ["https://dblp.org/streams/conf/wsdm"],
    "CIKM": ["https://dblp.org/streams/conf/cikm"],
    "ECIR": ["https://dblp.org/streams/conf/ecir"],
    "SIGIR-AP": ["https://dblp.org/streams/conf/sigir-ap"],
    "KDD": ["https://dblp.org/streams/conf/kdd"],
    "WWW": ["https://dblp.org/streams/conf/www"],
    "ACM_MM": ["https://dblp.org/streams/conf/mm"],
    "ICDE": ["https://dblp.org/streams/conf/icde"],
    "ICDM": ["https://dblp.org/streams/conf/icdm"],
    "DASFAA": ["https://dblp.org/streams/conf/dasfaa"],
    "AAAI": ["https://dblp.org/streams/conf/aaai"],
    "IJCAI": ["https://dblp.org/streams/conf/ijcai"],
    "ACL_EMNLP": [
        "https://dblp.org/streams/conf/acl",
        "https://dblp.org/streams/conf/emnlp",
        "https://dblp.org/streams/conf/naacl",
    ],
    "ICLR_NeurIPS_ICML": [
        "https://dblp.org/streams/conf/iclr",
        "https://dblp.org/streams/conf/nips",
        "https://dblp.org/streams/conf/icml",
    ],
    "TOIS": ["https://dblp.org/streams/journals/tois"],
    "TKDE": ["https://dblp.org/streams/journals/tkde"],
    "TORS": ["https://dblp.org/streams/journals/tors"],
    "TMM": ["https://dblp.org/streams/journals/tmm"],
    "TKDD": ["https://dblp.org/streams/journals/tkdd"],
    "TWEB": ["https://dblp.org/streams/journals/tweb"],
}


def _build_http_opener(proxy_url: str = ""):
    handlers = [urllib.request.HTTPCookieProcessor(_COOKIE_JAR)]
    if proxy_url:
        handlers.append(urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url}))
    return urllib.request.build_opener(*handlers)


def _solve_anubis_pow_if_challenged(html_body: str, target_url: str, opener) -> str:
    """如果返回了 Anubis `<title>Making sure you're not a bot!</title>` 验证页，
    自动解析页面内的 JSON 挑战参数并用 hashlib.sha256 在毫秒级算出 nonce 获取放行 Cookie。
    """
    if "Making sure you" not in html_body and "anubis" not in html_body.lower():
        return html_body

    m = re.search(
        r"""<script[^>]+id=["']anubis_challenge["'][^>]*>(.*?)</script>""",
        html_body,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return html_body

    try:
        payload = json.loads(m.group(1).strip())
        chal = payload.get("challenge") if isinstance(payload.get("challenge"), dict) else payload
        chal_id = chal.get("id") or payload.get("id") or ""
        rand_data = chal.get("randomData") or chal.get("data") or payload.get("randomData") or ""
        rules = payload.get("rules") if isinstance(payload.get("rules"), dict) else chal
        difficulty = int(rules.get("difficulty") or chal.get("difficulty") or 4)
        if not rand_data:
            return html_body

        prefix_zeros = "0" * difficulty
        nonce = 0
        digest_hex = ""
        while nonce < 5_000_000:
            candidate = f"{rand_data}{nonce}".encode("utf-8")
            h = hashlib.sha256(candidate).hexdigest()
            if h.startswith(prefix_zeros):
                digest_hex = h
                break
            nonce += 1

        if not digest_hex:
            return html_body

        parsed = urllib.parse.urlparse(target_url)
        base_origin = f"{parsed.scheme}://{parsed.netloc}"
        pass_params = {
            "response": digest_hex,
            "nonce": str(nonce),
            "redir": target_url,
            "elapsedTime": "45",
        }
        if chal_id:
            pass_params["id"] = chal_id
        pass_url = f"{base_origin}/.within.website/x/cmd/anubis/api/pass-challenge?" + urllib.parse.urlencode(pass_params)
        req = urllib.request.Request(
            pass_url,
            headers={
                "User-Agent": "curl/8.7.1",
                "Accept": "application/json, text/html, */*",
                "Referer": target_url,
            },
        )
        with opener.open(req, timeout=20) as resp:
            return resp.read().decode("utf-8", errors="ignore").strip()
    except Exception:
        return html_body


def _open_url_with_proxy_fallback(url: str, timeout: int = 25, accept: str = "application/json, text/html, */*") -> str:
    """使用非 Mozilla 的 `curl/8.7.1` User-Agent（避开 Anubis 针对 Mozilla 浏览器的默认拦截规则），
    支持直连与本地代理 127.0.0.1:7897 自动切换，并在遇到 Anubis PoW 时自动计算 SHA-256 签名过检。
    """
    headers = {
        "User-Agent": "curl/8.7.1",
        "Accept": accept,
    }
    last_err = None
    for proxy_url in ("", "http://127.0.0.1:7897", "http://127.0.0.1:7890"):
        opener = _build_http_opener(proxy_url)
        req = urllib.request.Request(url, headers=headers)
        try:
            with opener.open(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="ignore").strip()
            if "Making sure you" in body or "anubis_challenge" in body:
                body = _solve_anubis_pow_if_challenged(body, url, opener)
            return body
        except urllib.error.HTTPError:
            raise
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    return ""


def _fetch_from_dblp_sparql(venue: str, year: str, verbose: bool = False) -> tuple:
    """通道 1（首选，无 Anubis 拦截、无 1000 条截断）：
    通过 DBLP 官方 QLever SPARQL 知识图谱接口 (https://sparql.dblp.org/sparql)
    直接按 `dblp:publishedInStream` 与 `dblp:yearOfPublication` 精确拉取整本会议/期刊目录。
    """
    streams = DBLP_SPARQL_STREAMS.get(venue, [])
    if not streams:
        return [], {}
    pure_rec_venue = venue in {"RecSys", "TORS"}
    titles = []
    ee_map = {}
    seen_t = set()

    for stream_uri in streams:
        sparql_query = (
            "PREFIX dblp: <https://dblp.org/rdf/schema#>\n"
            "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
            "SELECT ?paper ?title ?ee WHERE {\n"
            f"  ?paper dblp:publishedInStream <{stream_uri}> .\n"
            f'  ?paper dblp:yearOfPublication "{year}"^^xsd:gYear .\n'
            "  ?paper dblp:title ?title .\n"
            "  OPTIONAL { ?paper dblp:primaryDocumentPage ?ee . }\n"
            "}"
        )
        url = "https://sparql.dblp.org/sparql?" + urllib.parse.urlencode({"query": sparql_query})
        try:
            raw = _open_url_with_proxy_fallback(
                url,
                timeout=30,
                accept="application/sparql-results+json, application/json",
            )
            if not raw.startswith("{"):
                if verbose:
                    print(f"    [SPARQL Debug] sparql.dblp.org 返回非 JSON: {raw[:80]}")
                continue
            data = json.loads(raw)
            bindings = data.get("results", {}).get("bindings", [])
            for b in bindings:
                raw_t = (b.get("title", {}).get("value") or "").strip()
                raw_t = re.sub(r"\.$", "", raw_t).strip()
                if not raw_t or len(raw_t) < 15 or "Proceedings of" in raw_t:
                    continue
                t_low = raw_t.lower()
                if any(neg in t_low for neg in ("workshop", "tutorial", "doctoral", "symposium", "keynote", "recsys challenge")):
                    continue
                if pure_rec_venue or is_recommendation_relevant(raw_t, ""):
                    if raw_t.lower() not in seen_t:
                        seen_t.add(raw_t.lower())
                        titles.append(raw_t)
                        ee_str = (b.get("ee", {}).get("value") or "").strip()
                        norm_k = re.sub(r"[^a-z0-9]", "", raw_t.lower())
                        if ee_str and norm_k not in ee_map:
                            ee_map[norm_k] = ee_str
        except Exception as e:
            if verbose:
                print(f"    [SPARQL Warn] {stream_uri} ({year}) 查询异常: {e}")
        time.sleep(2.5)

    return titles, ee_map


def _fetch_from_crossref(venue: str, year: str, verbose: bool = False) -> tuple:
    """通道 3（官方 DOI 注册中心兜底，永无 Anubis 拦截、支持每秒 50 次请求）：
    当 DBLP 主站被防火墙拦截时，从 Crossref REST API (https://api.crossref.org/works) 获取官方出版目录。
    """
    cfg = VENUE_CONFIGS.get(venue, {})
    oa_aliases = cfg.get("openalex", [venue])
    if isinstance(oa_aliases, str):
        oa_aliases = [oa_aliases]
    pure_rec_venue = venue in {"RecSys", "TORS"}
    titles = []
    ee_map = {}
    seen_t = set()

    for container_name in oa_aliases[:2]:
        params = {
            "query.container-title": container_name,
            "filter": f"from-pub-date:{year}-01-01,until-pub-date:{year}-12-31",
            "select": "DOI,title,container-title,URL",
            "rows": "200",
            "mailto": "recsys-notes@papernotes.org",
        }
        if not pure_rec_venue:
            params["query.bibliographic"] = "recommendation recommender collaborative sequential personalization"
        url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
        try:
            raw = _open_url_with_proxy_fallback(url, timeout=25, accept="application/json")
            if not raw.startswith("{"):
                continue
            data = json.loads(raw)
            items = data.get("message", {}).get("items", [])
            for item in items:
                c_titles = item.get("container-title") or []
                c_joined = " | ".join(c_titles).lower()
                if not c_joined or not any(re.search(pat, c_joined, re.IGNORECASE) for pat in cfg.get("tokens", [venue.lower()])):
                    continue
                if any(bad in c_joined for bad in ("workshop", "companion", "tutorial")):
                    continue
                t_arr = item.get("title") or []
                raw_t = re.sub(r"\.$", "", (t_arr[0] if t_arr else "").strip())
                if not raw_t or len(raw_t) < 15:
                    continue
                if pure_rec_venue or is_recommendation_relevant(raw_t, ""):
                    if raw_t.lower() not in seen_t:
                        seen_t.add(raw_t.lower())
                        titles.append(raw_t)
                        doi = (item.get("DOI") or "").strip()
                        ee_str = f"https://doi.org/{doi}" if doi else (item.get("URL") or "").strip()
                        norm_k = re.sub(r"[^a-z0-9]", "", raw_t.lower())
                        if ee_str:
                            ee_map[norm_k] = ee_str
        except Exception as e:
            if verbose:
                print(f"    [Crossref Warn] {container_name} ({year}): {e}")
        time.sleep(0.4)

    return titles, ee_map


def fetch_dblp_official_titles(venue: str, year: str, max_hits: int = 1000, use_cache: bool = True, verbose: bool = False) -> tuple:
    """三级官方目录拉取引擎：
    1. 首选 DBLP 官方 SPARQL 知识图谱接口 (`https://sparql.dblp.org/sparql`，独立于主站 Anubis 防火墙，单次精确拉取整本会议/期刊)；
    2. 次选 DBLP Search API（使用 `curl/8.7.1` 免拦截 UA + 内置 Anubis SHA-256 PoW 自动求解器）；
    3. 兜底 Crossref 官方 DOI 注册库接口 (`https://api.crossref.org/works`)。
    """
    cfg = VENUE_CONFIGS.get(venue, {})
    dblp_aliases = cfg.get("dblp", [venue])
    if isinstance(dblp_aliases, str):
        dblp_aliases = [dblp_aliases]
    key_prefixes = DBLP_KEY_PREFIXES.get(venue, ())
    pure_rec_venue = venue in {"RecSys", "TORS"}

    cache_dir = SCRIPT_DIR.parent / "data" / "dblp_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{venue}_{year}.json"

    titles = []
    ee_map = {}
    if use_cache and cache_file.exists():
        try:
            cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(cached_data, list) and cached_data:
                if isinstance(cached_data[0], dict):
                    titles = [item["title"] for item in cached_data if item.get("title")]
                    ee_map = {re.sub(r"[^a-z0-9]", "", item["title"].lower()): item.get("ee", "") for item in cached_data if item.get("title")}
                else:
                    titles = cached_data
                print(f"  [DBLP Cache] 命中本地官方目录缓存 ({venue} {year}): {len(titles)} 篇官方录用推荐候选")
                return titles, ee_map
        except Exception:
            pass

    seen_t = set()
    cached_records = []

    # --- 通道 1: DBLP 官方 SPARQL 知识图谱端点 (https://sparql.dblp.org/sparql) ---
    sparql_titles, sparql_ee = _fetch_from_dblp_sparql(venue, year, verbose=verbose)
    for t in sparql_titles:
        if t.lower() not in seen_t:
            seen_t.add(t.lower())
            titles.append(t)
            norm_k = re.sub(r"[^a-z0-9]", "", t.lower())
            ee_str = sparql_ee.get(norm_k, "")
            if ee_str:
                ee_map[norm_k] = ee_str
            cached_records.append({"title": t, "ee": ee_str, "key": "dblp_sparql"})

    # --- 通道 2: 若 SPARQL 未返回结果，尝试 DBLP Search API (带 curl UA 与 Anubis PoW 自动求解) ---
    if not titles:
        for idx, dblp_venue in enumerate(dblp_aliases):
            q = f"{dblp_venue} {year}" if pure_rec_venue else f"{dblp_venue} {year} recommend"
            for attempt in range(len(DBLP_MIRRORS)):
                mirror = DBLP_MIRRORS[(idx + attempt) % len(DBLP_MIRRORS)]
                url = f"{mirror}/search/publ/api?q={urllib.parse.quote(q)}&h={max_hits}&format=json"
                try:
                    raw = _open_url_with_proxy_fallback(url, timeout=25)
                    if not raw.startswith("{"):
                        if verbose:
                            print(f"    [Debug] {mirror} 仍返回非 JSON 内容: {raw[:75]}")
                        continue
                    data = json.loads(raw)
                    hits = data.get("result", {}).get("hits", {}).get("hit", [])
                    if isinstance(hits, dict):
                        hits = [hits]
                    for h in hits:
                        if not isinstance(h, dict):
                            continue
                        info = h.get("info", {})
                        h_year = str(info.get("year") or "").strip()
                        if h_year and h_year != str(year):
                            continue
                        h_key = str(info.get("key") or "").strip().lower()
                        if key_prefixes and h_key and not any(h_key.startswith(pfx) for pfx in key_prefixes):
                            continue
                        h_venue = str(info.get("venue") or "").lower()
                        if any(bad in h_venue for bad in ("workshop", "companion", "tutorial", "doctoral")):
                            continue
                        title = re.sub(r"\.$", "", (info.get("title") or "").strip())
                        if not title or len(title) < 15 or "Proceedings of" in title:
                            continue
                        if pure_rec_venue or is_recommendation_relevant(title, ""):
                            if title.lower() not in seen_t:
                                seen_t.add(title.lower())
                                titles.append(title)
                                ee_val = info.get("ee", "")
                                if isinstance(ee_val, list):
                                    ee_val = ee_val[0] if ee_val else ""
                                ee_str = str(ee_val or "").strip()
                                norm_k = re.sub(r"[^a-z0-9]", "", title.lower())
                                ee_map[norm_k] = ee_str
                                cached_records.append({"title": title, "ee": ee_str, "key": h_key})
                    break
                except Exception as e:
                    if verbose and attempt == len(DBLP_MIRRORS) - 1:
                        print(f"    [DBLP Search Warn] {mirror}: {e}")
                    time.sleep(1.5)

    # --- 通道 3: 若仍为空，回退至 Crossref 官方 REST API ---
    if not titles:
        if verbose:
            print(f"    [Crossref Fallback] 正在从 Crossref 官方 DOI 库查询 {venue} {year} ...")
        cr_titles, cr_ee = _fetch_from_crossref(venue, year, verbose=verbose)
        for t in cr_titles:
            if t.lower() not in seen_t:
                seen_t.add(t.lower())
                titles.append(t)
                norm_k = re.sub(r"[^a-z0-9]", "", t.lower())
                ee_str = cr_ee.get(norm_k, "")
                if ee_str:
                    ee_map[norm_k] = ee_str
                cached_records.append({"title": t, "ee": ee_str, "key": "crossref"})

    if cached_records:
        try:
            cache_file.write_text(json.dumps(cached_records, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    return titles, ee_map


def discover_via_dblp_and_resolve_arxiv(venue: str, year: str, max_hits: int = 1000, skip_norm_titles: set = None) -> list:
    """从 DBLP 官方目录拉取指定会刊与年份的推荐论文标题，并批量去 arXiv 反查预印本全文。"""
    titles, ee_map = fetch_dblp_official_titles(venue, year, max_hits=max_hits)
    return batch_resolve_titles_on_arxiv(titles[:400], venue=venue, year=year, ee_map=ee_map, skip_norm_titles=skip_norm_titles)


def discover_via_openalex(venue: str, year: str, per_page: int = 100, max_pages: int = 2, skip_norm_titles: set = None) -> list:
    """使用免 Key 的 OpenAlex API 检索目标会刊发表的推荐系统论文（支持多别名、多页抓取、429 退避与严格来源核验）。"""
    cfg = VENUE_CONFIGS.get(venue, {})
    oa_aliases = cfg.get("openalex", [venue])
    if isinstance(oa_aliases, str):
        oa_aliases = [oa_aliases]

    direct_arxiv_ids = set()
    titles_to_resolve = []

    for oa_venue in oa_aliases:
        search_q = f'"{oa_venue}" (recommendation OR recommender OR collaborative OR personalization)'
        for page_num in range(1, max_pages + 1):
            params = {
                "search": search_q,
                "filter": f"publication_year:{year}",
                "per-page": str(per_page),
                "page": str(page_num),
                "mailto": "recsys-notes@papernotes.org",
            }
            url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
            results = []
            for attempt in range(3):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "RecSys-Paper-Notes/3.6"})
                    with urllib.request.urlopen(req, timeout=25) as resp:
                        data = json.loads(resp.read().decode("utf-8", errors="ignore"))
                    results = data.get("results", [])
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"  [Warn] OpenAlex 检索跳过 ({oa_venue} {year} p{page_num}): {e}")
                    time.sleep(1.5 * (attempt + 1))

            if not results:
                break

            for work in results:
                title = (work.get("title") or "").strip()
                if not title or len(title) < 15:
                    continue
                locs = work.get("locations") or []
                primary_loc = work.get("primary_location") or {}
                all_sources = []
                for loc in [primary_loc] + locs:
                    src = (loc or {}).get("source") or {}
                    dn = (src.get("display_name") or "").lower()
                    if dn:
                        all_sources.append(dn)
                joined_src = " | ".join(all_sources)
                # P1-1 修复：joined_src 为空时必须直接跳过，严防无来源论文混入
                if not joined_src or not any(re.search(pat, joined_src, re.IGNORECASE) for pat in cfg.get("tokens", [venue.lower()])):
                    continue

                found_aid = ""
                ids_dict = work.get("ids") or {}
                for candidate_url in [ids_dict.get("arxiv", "")] + [
                    (loc or {}).get("landing_page_url", "") for loc in locs
                ]:
                    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", str(candidate_url or ""))
                    if m:
                        found_aid = m.group(1)
                        break

                if found_aid:
                    direct_arxiv_ids.add(found_aid)
                else:
                    titles_to_resolve.append(title)

            if len(results) < per_page:
                break
            time.sleep(0.3)

    resolved_papers = []
    if direct_arxiv_ids:
        fetched = fetch_arxiv_by_ids(sorted(direct_arxiv_ids), cache_dir=None, download_pdf=False)
        for ep in fetched:
            orig_c = ep.get("comment") or ""
            ep["comment"] = f"{orig_c} | Accepted at {venue} {year} (OpenAlex Verified)".strip(" |")
            ep["trusted_venue_match"] = True
            resolved_papers.append(ep)

    if titles_to_resolve:
        resolved_papers.extend(batch_resolve_titles_on_arxiv(titles_to_resolve[:150], venue=venue, year=year, skip_norm_titles=skip_norm_titles))

    return resolved_papers


def _row_unique_key(row: dict) -> str:
    """P0-4 修复：支持 arXiv ID、Paper URL 或标题 slug 作为行唯一键，保留非 arXiv 行。"""
    aid = re.sub(r"v\d+$", "", (row.get("arxiv_id") or "").strip())
    if aid:
        return f"arxiv:{aid}"
    purl = (row.get("paper_url") or "").strip()
    if purl:
        return f"url:{purl}"
    t = (row.get("title") or "").strip()
    return f"title:{slugify(t)}" if t else ""


def build_for_single_venue(venue: str, args, log: RunLog, single_venue_mode: bool = True) -> tuple:
    """对单个会刊执行多通道高召回发现 + LLM 相关性严格复核，返回 (新增篇数, 总篇数)。"""
    # P0-5 修复：仅在 single_venue_mode 下才使用 args.out
    out_path = Path(args.out) if (args.out and single_venue_mode) else SCRIPT_DIR.parent / "data" / "papers" / f"{venue}{args.year}.csv"
    existing_rows_by_key = {}
    seen_arxiv_ids = set()
    existing_norm_titles = set()
    if out_path.exists() and not args.no_incremental:
        with out_path.open("r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                # P0-6 修复：只保留与当前文件目标年份一致的记录
                r_year = str(r.get("year") or args.year).strip()
                if r_year != str(args.year):
                    continue
                aid = re.sub(r"v\d+$", "", (r.get("arxiv_id") or "").strip())
                if aid:
                    r["arxiv_id"] = aid
                    seen_arxiv_ids.add(aid)
                if r.get("title"):
                    existing_norm_titles.add(re.sub(r"[^a-z0-9]", "", r["title"].lower()))
                ukey = _row_unique_key(r)
                if ukey:
                    existing_rows_by_key[ukey] = r
        print(f"\n[{venue} {args.year}] 已加载现有 CSV 中的 {len(existing_rows_by_key)} 篇论文记录。")

    cfg = VENUE_CONFIGS.get(venue, {"queries": [f'all:"{venue} {{year}}"'], "tokens": [venue.lower()], "neg": []})

    papers = []
    # 2024 及更早年份（或显式指定 --dblp-only）默认以 DBLP 官方录用目录为准，不再依赖可能漏标/误标的 arXiv comment 搜索
    use_dblp_authority_only = bool(getattr(args, "dblp_only", False)) or (args.year.isdigit() and int(args.year) <= 2025)

    if use_dblp_authority_only:
        print(f"[{venue} {args.year}] 启用【DBLP 官方目录权威模式】：直接从 DBLP 拉取官方录用清单并反查 arXiv/ACM 全文 ...")
        dblp_found = discover_via_dblp_and_resolve_arxiv(venue, args.year, skip_norm_titles=existing_norm_titles)
        oa_found = []
        print(f"[{venue} {args.year}] 官方目录共匹配到 {len(dblp_found) + len(oa_found)} 篇可获取全文的推荐系统论文。")
        papers.extend(dblp_found)
        papers.extend(oa_found)
    else:
        for q_tmpl in cfg["queries"]:
            q_formatted = q_tmpl.format(year=args.year)
            if not any(prefix in q_formatted for prefix in ("all:", "co:", "jr:", "ti:", "abs:")):
                q_str = f"all:{q_formatted}"
            else:
                q_str = q_formatted
            print(f"[{venue} Query] {q_str} (max_results={args.max_results})")
            log.write(f"QUERY venue={venue} year={args.year} {q_str} max_results={args.max_results}")
            try:
                papers.extend(
                    _fetch_arxiv_entries(
                        {"search_query": q_str, "max_results": args.max_results, "sortBy": "submittedDate", "sortOrder": "descending"},
                        cache_dir=None,
                        download_pdf=False,
                    )
                )
            except Exception as e:
                print(f"  [Warn] arXiv 查询异常 ({q_str}): {e}")
            time.sleep(2.5)

        if args.use_dblp:
            known_norms = set(existing_norm_titles) | {
                re.sub(r"[^a-z0-9]", "", (p.get("title") or "").lower()) for p in papers if p.get("title")
            }
            print(f"[{venue} DBLP] 正在检索 DBLP 官方目录并过滤已收录标题 ({venue} {args.year}) ...")
            dblp_found = discover_via_dblp_and_resolve_arxiv(venue, args.year, skip_norm_titles=known_norms)
            oa_found = []
            print(f"[{venue} OpenAlex+DBLP] 额外发现 {len(oa_found) + len(dblp_found)} 篇官方录用 arXiv 预印本。")
            papers.extend(oa_found)
            papers.extend(dblp_found)

    candidates = []
    for p in papers:
        arxiv_id = re.sub(r"v\d+$", "", p.get("arxiv_id", "").strip())
        if not arxiv_id or arxiv_id in seen_arxiv_ids:
            continue

        pub_date = p.get("published_date", "")
        upd_date = p.get("updated_date", "")
        if not pub_date:
            m_pub = re.search(r"Published Date:\s*(\d{4}-\d{2}-\d{2})", p.get("content", ""))
            if m_pub:
                pub_date = m_pub.group(1)

        if not p.get("trusted_venue_match") and not is_target_venue_and_year(
            p.get("comment", ""),
            venue,
            args.year,
            pub_date,
            upd_date,
        ):
            continue

        abstract = p.get("content", "").split("Abstract:\n", 1)[-1]
        if not args.no_relevance_filter and not is_recommendation_relevant(p["title"], abstract):
            continue

        seen_arxiv_ids.add(arxiv_id)
        p["arxiv_id"] = arxiv_id
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
            print(f"[{venue} LLM] 使用 {model} 对 {len(candidates)} 篇新发现候选论文做严格推荐相关性判断与分类 ...")
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
                print(f"  [LLM 排除] {p['title'][:82]} | {result.get('reason', '')}")
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
            "venue": venue,
            "year": str(args.year),
        })

    final_rows = list(existing_rows_by_key.values()) + new_rows
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "arxiv_id", "paper_url", "code_url", "category", "venue", "year"])
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"[Done] [{venue} {args.year}] 策展名单共 {len(final_rows)} 篇（本次新增 {len(new_rows)} 篇，LLM 排除 {llm_removed} 篇）-> {out_path}")
    for row in new_rows[:10]:
        print(f"  + [{row['year']}][{row['category']}] {row['title'][:85]} ({row['arxiv_id']})")
    return len(new_rows), len(final_rows)


def main():
    parser = argparse.ArgumentParser(description="从 arXiv + OpenAlex + DBLP 自动构建 22 个推荐系统会刊高召回策展名单")
    parser.add_argument("--venue", default="SIGIR", choices=sorted(VENUES.keys()), help="单个目标会刊 key")
    parser.add_argument("--venues", default="", help="多个目标会刊（逗号分隔，如 ACM_MM,TOIS,TKDE,AAAI,IJCAI）")
    parser.add_argument("--all-venues", action="store_true", help="一键构建全部 22 个顶会与顶刊的候选名单")
    parser.add_argument("--year", default="2026")
    parser.add_argument("--max-results", type=int, default=300)
    parser.add_argument("--use-dblp", action="store_true", help="同时查询 OpenAlex 与 DBLP 官方录用列表并批量反查 arXiv")
    parser.add_argument("--dblp-only", action="store_true", help="严格只以 DBLP/OpenAlex 官方目录为准（2025 及更早年份默认自动开启）")
    parser.add_argument("--test-dblp", action="store_true", help="仅测试各会刊 DBLP 官方目录连通性与推荐论文命中数（不写 CSV、不调 LLM）")
    parser.add_argument("--no-incremental", action="store_true", help="不合并已有 CSV，从零重建名单（默认开启增量合并）")
    parser.add_argument("--no-relevance-filter", action="store_true", help="保留所有主会论文（默认仅保留推荐相关论文）")
    parser.add_argument("--out", default="", help="输出 CSV 路径；默认 data/papers/<VENUE><YEAR>.csv")
    parser.add_argument("--no-llm", action="store_true", help="禁用 LLM，仅用关键词启发式分类")
    parser.add_argument("--opencode-free", action="store_true", help="启用 OpenCode 免费 deepseek-v4.1-flash-free 模型额度")
    parser.add_argument("--llm-api-key", default="", help="LLM API Key（或 LLM_API_KEY / DEEPSEEK_API_KEY / OPENCODE_API_KEY）")
    parser.add_argument("--llm-base-url", default="", help="LLM API Base URL")
    parser.add_argument("--llm-model", default="", help="LLM 模型名称")
    parser.add_argument("--llm-workers", type=int, default=4, help="LLM 并发线程数（默认 4）")
    parser.add_argument("--llm-retries", type=int, default=2, help="LLM 单篇失败重试次数（默认 2）")
    parser.add_argument("--llm-min-confidence", type=float, default=0.6, help="LLM 排除非推荐论文所需的最低置信度")
    parser.add_argument("--llm-cache", default=str(DEFAULT_CACHE_PATH), help="LLM 分类缓存路径")
    parser.add_argument("--refresh-llm-cache", action="store_true", help="忽略本地 LLM 分类缓存")
    args = parser.parse_args()
    if args.opencode_free:
        os.environ['USE_OPENCODE_FREE'] = '1'
    run_start = time.monotonic()

    if args.all_venues:
        venue_list = list(VENUES.keys())
    elif args.venues:
        raw_v = [v.strip() for v in args.venues.split(",") if v.strip()]
        invalid_v = [v for v in raw_v if v not in VENUES]
        if invalid_v:
            parser.error(f"未知会刊标识: {', '.join(invalid_v)}。合法标识为: {', '.join(sorted(VENUES.keys()))}")
        venue_list = raw_v
    else:
        venue_list = [args.venue]

    log = RunLog("build-list")
    log.write(
        f"START venues={','.join(venue_list)} year={args.year} max_results={args.max_results} "
        f"use_dblp={args.use_dblp} no_llm={args.no_llm} workers={args.llm_workers}"
    )

    if args.test_dblp:
        print(f"\n[DBLP Test] 开始测试 {len(venue_list)} 个会刊在 {args.year} 年度的 DBLP 官方目录连通性 ...")
        for v_key in venue_list:
            t_list, _ = fetch_dblp_official_titles(v_key, args.year, use_cache=False, verbose=True)
            sample_t = t_list[0][:70] if t_list else "（当年卷宗暂未入库 DBLP 或无匹配）"
            print(f"  - {v_key:<18} ({args.year}): 命中官方目录推荐论文 {len(t_list):>3} 篇 | 示例: {sample_t}")
        print("[DBLP Test] 测试完成。")
        log.close()
        return

    total_new = 0
    total_all = 0
    failed_venues = []
    single_mode = len(venue_list) == 1
    for v_key in venue_list:
        try:
            n_new, n_all = build_for_single_venue(v_key, args, log, single_venue_mode=single_mode)
            total_new += n_new
            total_all += n_all
        except Exception as e:
            failed_venues.append((v_key, str(e)))
            print(f"[Error] [{v_key}] 构建候选名单失败，已隔离并继续后续会刊: {e}")
            log.write(f"ERROR venue={v_key} {e}")

    elapsed = time.monotonic() - run_start
    print_usage_summary(
        resolve_model(args.llm_model),
        processed=total_new,
        success=total_new,
        elapsed_seconds=elapsed,
        log=log,
    )
    print(f"[Summary] 共处理 {len(venue_list)} 个会刊，本次新增 {total_new} 篇，累计在册 {total_all} 篇（耗时 {format_duration(elapsed)}）")
    if failed_venues:
        print(f"[Warn] 以下 {len(failed_venues)} 个会刊执行出错：{', '.join(v for v, _ in failed_venues)}")
    print(f"[Log] 详细日志: {log.path}")
    log.write(f"TOTAL elapsed={format_duration(elapsed)} venues={len(venue_list)} new={total_new} total={total_all} failed={len(failed_venues)}")
    log.close()


if __name__ == "__main__":
    main()