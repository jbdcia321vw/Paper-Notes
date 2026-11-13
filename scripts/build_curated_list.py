#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
build_curated_list.py (v3.10 DBLP 静态目录页+API双通道 + 自动代理探测 127.0.0.1:7897 + 诊断输出版)
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


def batch_resolve_titles_on_arxiv(titles: list, venue: str, year: str, batch_size: int = 6, ee_map: dict = None) -> list:
    """将从 DBLP/OpenAlex 获得的官方录用论文标题按 6 篇一批在 arXiv 上批量反查预印本，避免单篇串行触发 429。"""
    if not titles:
        return []
    resolved = []
    seen_ids = set()

    norm_map = {}
    clean_queries = []
    for raw_t in titles:
        norm_t = re.sub(r"[^a-z0-9]", "", raw_t.lower())
        if len(norm_t) < 12 or norm_t in norm_map:
            continue
        norm_map[norm_t] = raw_t
        clean_t = re.sub(r"[^a-zA-Z0-9\s]", " ", raw_t)
        clean_t = re.sub(r"\s+", " ", clean_t).strip()
        if clean_t:
            clean_queries.append((norm_t, clean_t))

    for start in range(0, len(clean_queries), batch_size):
        chunk = clean_queries[start : start + batch_size]
        or_clauses = " OR ".join(f'ti:"{item[1][:95]}"' for item in chunk)
        try:
            entries = _fetch_arxiv_entries(
                {"search_query": or_clauses, "max_results": len(chunk) * 2},
                cache_dir=None,
                download_pdf=False,
            )
            for ep in entries:
                aid = re.sub(r"v\d+$", "", ep.get("arxiv_id", "").strip())
                if not aid or aid in seen_ids:
                    continue
                norm_ep = re.sub(r"[^a-z0-9]", "", (ep.get("title") or "").lower())
                if not norm_ep:
                    continue
                matched = norm_ep in norm_map
                if not matched:
                    for target_norm in norm_map:
                        if len(target_norm) >= 25 and (target_norm in norm_ep or norm_ep in target_norm):
                            matched = True
                            break
                if matched:
                    seen_ids.add(aid)
                    orig_c = ep.get("comment") or ""
                    ep["comment"] = f"{orig_c} | Accepted at {venue} {year} (DBLP Official Verified)".strip(" |")
                    ep["trusted_venue_match"] = True
                    if ee_map and norm_ep in ee_map and ee_map[norm_ep]:
                        ep["official_ee"] = ee_map[norm_ep]
                    resolved.append(ep)
            time.sleep(0.6)
        except Exception:
            time.sleep(1.0)
            continue
    return resolved


def _open_url_with_proxy_fallback(url: str, timeout: int = 20) -> str:
    """优先使用系统/环境变量配置请求 URL；若连接超时或被重置，自动探测并尝试本地代理 http://127.0.0.1:7897。"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RecSys-Paper-Notes/3.10",
        "Accept": "application/json, text/html, application/xhtml+xml, */*",
    }
    req = urllib.request.Request(url, headers=headers)
    last_err = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="ignore").strip()
    except urllib.error.HTTPError:
        raise
    except Exception as e:
        last_err = e

    # 自动尝试用户常用的本地代理端口 127.0.0.1:7897
    for proxy_url in ("http://127.0.0.1:7897", "http://127.0.0.1:7890"):
        try:
            proxy_handler = urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
            opener = urllib.request.build_opener(proxy_handler)
            with opener.open(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="ignore").strip()
        except urllib.error.HTTPError:
            raise
        except Exception:
            continue

    raise last_err


# DBLP 静态会议目录页路径模板（当 Search API 限流时可直接读取静态目录页，永不 429）
DBLP_STATIC_TOC_PATHS = {
    "SIGIR": ["db/conf/sigir/sigir{year}.html"],
    "RecSys": ["db/conf/recsys/recsys{year}.html"],
    "WSDM": ["db/conf/wsdm/wsdm{year}.html"],
    "CIKM": ["db/conf/cikm/cikm{year}.html"],
    "ECIR": ["db/conf/ecir/ecir{year}-1.html", "db/conf/ecir/ecir{year}-2.html", "db/conf/ecir/ecir{year}-3.html"],
    "SIGIR-AP": ["db/conf/sigir-ap/sigir-ap{year}.html"],
    "KDD": ["db/conf/kdd/kdd{year}-1.html", "db/conf/kdd/kdd{year}-2.html"],
    "WWW": ["db/conf/www/www{year}.html"],
    "ACM_MM": ["db/conf/mm/mm{year}.html"],
    "ICDE": ["db/conf/icde/icde{year}.html"],
    "ICDM": ["db/conf/icdm/icdm{year}.html"],
    "DASFAA": ["db/conf/dasfaa/dasfaa{year}-1.html", "db/conf/dasfaa/dasfaa{year}-2.html", "db/conf/dasfaa/dasfaa{year}-3.html"],
    "AAAI": ["db/conf/aaai/aaai{year}.html"],
    "IJCAI": ["db/conf/ijcai/ijcai{year}.html"],
    "ACL_EMNLP": ["db/conf/acl/acl{year}-1.html", "db/conf/emnlp/emnlp{year}-main.html", "db/conf/emnlp/emnlp{year}.html"],
    "ICLR_NeurIPS_ICML": ["db/conf/iclr/iclr{year}.html", "db/conf/nips/neurips{year}.html", "db/conf/icml/icml{year}.html"],
}


def _fetch_from_dblp_static_toc(venue: str, year: str) -> tuple:
    """直接抓取 DBLP 静态会议目录页 (db/conf/<venue>/<venue><year>.html)，不受 Search API 限流影响。"""
    paths = DBLP_STATIC_TOC_PATHS.get(venue, [])
    if not paths:
        return [], {}
    titles = []
    ee_map = {}
    seen_t = set()
    pure_rec_venue = venue in {"RecSys", "TORS"}

    for rel_p in paths:
        formatted_p = rel_p.format(year=year)
        for mirror in DBLP_MIRRORS:
            url = f"{mirror}/{formatted_p}"
            try:
                html_str = _open_url_with_proxy_fallback(url, timeout=25)
                if 'class="title"' not in html_str:
                    continue
                # 提取每条论文记录的 title
                for m in re.finditer(r'<span class="title"[^>]*>(.*?)</span>', html_str, re.DOTALL):
                    raw_t = re.sub(r"<[^>]+>", "", m.group(1)).strip()
                    raw_t = re.sub(r"\\.$", "", raw_t).strip()
                    if not raw_t or len(raw_t) < 15 or "Proceedings of" in raw_t or "Workshop on" in raw_t:
                        continue
                    if pure_rec_venue or is_recommendation_relevant(raw_t, ""):
                        if raw_t.lower() not in seen_t:
                            seen_t.add(raw_t.lower())
                            titles.append(raw_t)
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    break
            except Exception:
                continue
        time.sleep(1.0)
    return titles, ee_map


def fetch_dblp_official_titles(venue: str, year: str, max_hits: int = 1000, use_cache: bool = True, verbose: bool = False) -> tuple:
    """从 DBLP 官方 API + 静态目录页双通道拉取指定会刊与年份的推荐相关录用论文清单（返回 (titles, ee_map)）。"""
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
    for idx, dblp_venue in enumerate(dblp_aliases):
        # 对纯推荐会刊 (RecSys / TORS) 直接拉取全量目录；对其余会刊使用 DBLP 默认前缀匹配 `recommend` (自动匹配 recommend/recommendation/recommender)
        q = f"{dblp_venue} {year}" if pure_rec_venue else f"{dblp_venue} {year} recommend"
        for attempt in range(len(DBLP_MIRRORS)):
            mirror = DBLP_MIRRORS[(idx + attempt) % len(DBLP_MIRRORS)]
            url = f"{mirror}/search/publ/api?q={urllib.parse.quote(q)}&h={max_hits}&format=json"
            try:
                raw = _open_url_with_proxy_fallback(url, timeout=25)
                if not raw.startswith("{"):
                    if verbose:
                        print(f"    [Debug] {mirror} 返回非 JSON 内容: {raw[:80]}")
                    time.sleep(2.0)
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
                    title = re.sub(r"\\.$", "", (info.get("title") or "").strip())
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
            except urllib.error.HTTPError as e:
                retry_after = 3.0
                if e.code == 429 and e.headers:
                    ra = e.headers.get("Retry-After")
                    if ra and str(ra).isdigit():
                        retry_after = min(float(ra), 15.0)
                if verbose:
                    print(f"    [DBLP HTTP {e.code}] {mirror} 限流或报错，等待 {retry_after:.0f}s 后切换备用镜像 ...")
                time.sleep(retry_after)
            except Exception as e:
                if verbose and attempt == len(DBLP_MIRRORS) - 1:
                    print(f"    [DBLP Network Warn] {mirror} 连接异常: {e}")
                time.sleep(1.5)
        time.sleep(1.8)

    # 若 Search API 未返回结果（如被限流），自动回退到 DBLP 静态目录页抓取
    if not titles and venue in DBLP_STATIC_TOC_PATHS:
        if verbose:
            print(f"    [Fallback] 正在从 DBLP 静态目录页 (db/conf) 直接读取 {venue} {year} ...")
        static_titles, static_ee = _fetch_from_dblp_static_toc(venue, year)
        for t in static_titles:
            if t.lower() not in seen_t:
                seen_t.add(t.lower())
                titles.append(t)
                cached_records.append({"title": t, "ee": static_ee.get(re.sub(r"[^a-z0-9]", "", t.lower()), ""), "key": "static_toc"})

    if cached_records:
        try:
            cache_file.write_text(json.dumps(cached_records, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    return titles, ee_map


def discover_via_dblp_and_resolve_arxiv(venue: str, year: str, max_hits: int = 1000) -> list:
    """从 DBLP 官方目录拉取指定会刊与年份的推荐论文标题，并批量去 arXiv 反查预印本全文。"""
    titles, ee_map = fetch_dblp_official_titles(venue, year, max_hits=max_hits)
    return batch_resolve_titles_on_arxiv(titles[:220], venue=venue, year=year, ee_map=ee_map)


def discover_via_openalex(venue: str, year: str, per_page: int = 100, max_pages: int = 2) -> list:
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
        resolved_papers.extend(batch_resolve_titles_on_arxiv(titles_to_resolve[:150], venue=venue, year=year))

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
                ukey = _row_unique_key(r)
                if ukey:
                    existing_rows_by_key[ukey] = r
        print(f"\n[{venue} {args.year}] 已加载现有 CSV 中的 {len(existing_rows_by_key)} 篇论文记录。")

    cfg = VENUE_CONFIGS.get(venue, {"queries": [f'all:"{venue} {{year}}"'], "tokens": [venue.lower()], "neg": []})

    papers = []
    # 2024 及更早年份（或显式指定 --dblp-only）默认以 DBLP 官方录用目录为准，不再依赖可能漏标/误标的 arXiv comment 搜索
    use_dblp_authority_only = bool(getattr(args, "dblp_only", False)) or (args.year.isdigit() and int(args.year) <= 2024)

    if use_dblp_authority_only:
        print(f"[{venue} {args.year}] 启用【DBLP 官方目录权威模式】：直接从 DBLP 拉取官方录用清单并反查 arXiv/ACM 全文 ...")
        dblp_found = discover_via_dblp_and_resolve_arxiv(venue, args.year)
        oa_found = discover_via_openalex(venue, args.year)
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
            time.sleep(0.5)

        if args.use_dblp:
            print(f"[{venue} OpenAlex+DBLP] 正在联合检索 OpenAlex 与 DBLP 官方目录 ({venue} {args.year}) ...")
            oa_found = discover_via_openalex(venue, args.year)
            dblp_found = discover_via_dblp_and_resolve_arxiv(venue, args.year)
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
    parser.add_argument("--dblp-only", action="store_true", help="严格只以 DBLP/OpenAlex 官方目录为准（2024 及更早年份默认自动开启）")
    parser.add_argument("--test-dblp", action="store_true", help="仅测试各会刊 DBLP 官方目录连通性与推荐论文命中数（不写 CSV、不调 LLM）")
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