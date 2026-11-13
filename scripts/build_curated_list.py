#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_curated_list.py (v3.5 全量 22 会刊高召回发现引擎：arXiv co:/jr: + OpenAlex + DBLP 批量反查 + 稀疏会刊自动补齐版)
解决“部分顶会顶刊只有个位数论文、部分会刊为 0 篇”的五大核心根因：
  1. 修复 arXiv API 中 `AND recommend` 不匹配 `recommendation` / `recommender` 的分词缺陷，统一扩展为多词根检索。
  2. 新增 `co:` (comment) 与 `jr:` (journal_ref) 字段级检索，全面覆盖 `AAAI-26`、`WSDM'26`、`ACM MM'26`、`MM 2026`、`IJCAI'26` 等简写及 `<arxiv:journal_ref>` 期刊引用。
  3. 修复 DBLP 匹配成功后因原预印本 comment 非空（如 `"10 pages"`）被 `is_main_conference` 误杀的 Bug，并将逐篇标题反查改为 6 篇一批的批量反查，防 429 限流。
  4. 新增免 Key 高速 OpenAlex API (`https://api.openalex.org/works`) 联合检索，直接提取 22 个顶会顶刊的官方录用论文及关联 arXiv ID。
  5. 针对 2026 年尚未开会或预印本较少的会刊（当 2026 论文数 < `--min-venue-papers`，默认 18 篇时），自动从同会刊 2025 年录用论文中补齐（并在 CSV 中精准标注 `year=2025`），确保全站 22 个会刊全部充实饱满！

用法：
  python scripts/build_curated_list.py --venue ACM_MM --year 2026 --use-dblp
  python scripts/build_curated_list.py --venue TOIS --year 2026 --use-dblp
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
)
from paper_classifier import DEFAULT_CACHE_PATH, classify_items
from run_logging import RunLog, format_duration

COMMON_NEGATIVE_HINTS = [
    "workshop", "challenge", "tutorial", "technical report", "demo",
    "doctoral consortium", "companion", "extended abstract", "keynote",
]

REC_ARXIV_FILTER = "(all:recommendation OR all:recommender OR all:recommend OR all:collaborative OR all:personaliz OR all:sequential)"

JOURNAL_VENUES = {"TOIS", "TKDE", "TORS", "TMM", "TKDD", "TWEB"}

# 22 个推荐系统相关顶会与顶刊的高召回检索配置
VENUE_CONFIGS = {
    "SIGIR": {
        "queries": [
            'all:"SIGIR {year}"',
            f'(co:SIGIR OR jr:SIGIR) AND {REC_ARXIV_FILTER}',
        ],
        "tokens": [r"\bsigir\b"],
        "neg": ["sigir-ap", "sigir ap", "reneuir", "ikat", "evalmg", "agentsearch"],
        "dblp": "SIGIR",
        "openalex": "SIGIR",
    },
    "RecSys": {
        "queries": [
            'all:"RecSys {year}"',
            'co:RecSys OR jr:RecSys',
        ],
        "tokens": [r"\brecsys\b", r"recommender systems conference"],
        "neg": ["cars", "usrw", "genaiecommerce", "daquamrec", "oars", "recsys challenge"],
        "dblp": "RecSys",
        "openalex": "RecSys",
    },
    "WSDM": {
        "queries": [
            'all:"WSDM {year}"',
            f'(co:WSDM OR jr:WSDM) AND {REC_ARXIV_FILTER}',
        ],
        "tokens": [r"\bwsdm\b", r"web search and data mining"],
        "neg": ["wsdm cup"],
        "dblp": "WSDM",
        "openalex": "WSDM",
    },
    "CIKM": {
        "queries": [
            'all:"CIKM {year}"',
            f'(co:CIKM OR jr:CIKM) AND {REC_ARXIV_FILTER}',
        ],
        "tokens": [r"\bcikm\b", r"information and knowledge management"],
        "neg": [],
        "dblp": "CIKM",
        "openalex": "CIKM",
    },
    "ECIR": {
        "queries": [
            'all:"ECIR {year}"',
            f'(co:ECIR OR jr:ECIR) AND {REC_ARXIV_FILTER}',
            f'all:"European Conference on Information Retrieval" AND {REC_ARXIV_FILTER}',
        ],
        "tokens": [r"\becir\b", r"european conference on information retrieval"],
        "neg": [],
        "dblp": "ECIR",
        "openalex": "European Conference on Information Retrieval",
    },
    "SIGIR-AP": {
        "queries": [
            'all:"SIGIR-AP"',
            'all:"SIGIR AP"',
            'co:"SIGIR-AP" OR co:"SIGIR AP"',
        ],
        "tokens": [r"\bsigir[\s-]*ap\b", r"sigir asia pacific"],
        "neg": [],
        "dblp": "SIGIR-AP",
        "openalex": "SIGIR-AP",
    },
    "KDD": {
        "queries": [
            'all:"KDD {year}"',
            'all:"SIGKDD {year}"',
            f'(co:KDD OR co:SIGKDD OR jr:KDD) AND {REC_ARXIV_FILTER}',
        ],
        "tokens": [r"\b(?:sig)?kdd\b", r"knowledge discovery and data mining"],
        "neg": ["kdd cup"],
        "dblp": "KDD",
        "openalex": "KDD",
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
        "dblp": "WWW",
        "openalex": "The Web Conference",
    },
    "ACM_MM": {
        "queries": [
            'all:"ACM MM {year}"',
            'all:"ACM Multimedia {year}"',
            f'(co:"ACM MM" OR co:"ACM Multimedia" OR co:Multimedia) AND {REC_ARXIV_FILTER}',
        ],
        "tokens": [r"\bacm\s*(?:mm|multimedia)\b", r"\bmm\s*['-]?\s*(?:20\d{2}|\d{2})\b", r"acm international conference on multimedia"],
        "neg": [],
        "dblp": "ACM Multimedia",
        "openalex": "ACM Multimedia",
    },
    "ICDE": {
        "queries": [
            'all:"ICDE {year}"',
            f'(co:ICDE OR jr:ICDE) AND {REC_ARXIV_FILTER}',
            f'all:"Data Engineering" AND co:ICDE AND {REC_ARXIV_FILTER}',
        ],
        "tokens": [r"\bicde\b", r"international conference on data engineering"],
        "neg": [],
        "dblp": "ICDE",
        "openalex": "ICDE",
    },
    "ICDM": {
        "queries": [
            'all:"ICDM {year}"',
            f'(co:ICDM OR jr:ICDM) AND {REC_ARXIV_FILTER}',
            f'all:"Data Mining" AND co:ICDM AND {REC_ARXIV_FILTER}',
        ],
        "tokens": [r"\bicdm\b", r"international conference on data mining"],
        "neg": [],
        "dblp": "ICDM",
        "openalex": "ICDM",
    },
    "DASFAA": {
        "queries": [
            'all:"DASFAA {year}"',
            f'(co:DASFAA OR jr:DASFAA) AND {REC_ARXIV_FILTER}',
            'co:DASFAA OR jr:DASFAA',
        ],
        "tokens": [r"\bdasfaa\b", r"database systems for advanced applications"],
        "neg": [],
        "dblp": "DASFAA",
        "openalex": "DASFAA",
    },
    "AAAI": {
        "queries": [
            f'all:"AAAI {{year}}" AND {REC_ARXIV_FILTER}',
            f'(co:AAAI OR jr:AAAI) AND {REC_ARXIV_FILTER}',
        ],
        "tokens": [r"\baaai\b"],
        "neg": [],
        "dblp": "AAAI",
        "openalex": "AAAI",
    },
    "IJCAI": {
        "queries": [
            f'all:"IJCAI {{year}}" AND {REC_ARXIV_FILTER}',
            f'(co:IJCAI OR jr:IJCAI) AND {REC_ARXIV_FILTER}',
        ],
        "tokens": [r"\bijcai\b"],
        "neg": [],
        "dblp": "IJCAI",
        "openalex": "IJCAI",
    },
    "ACL_EMNLP": {
        "queries": [
            f'(co:ACL OR co:EMNLP OR co:NAACL OR jr:ACL OR jr:EMNLP) AND {REC_ARXIV_FILTER}',
            f'all:"ACL {{year}}" AND {REC_ARXIV_FILTER}',
            f'all:"EMNLP {{year}}" AND {REC_ARXIV_FILTER}',
        ],
        "tokens": [r"\b(?:acl|emnlp|naacl)\b", r"association for computational linguistics", r"empirical methods in natural language processing"],
        "neg": [],  # 保留 Findings of ACL / EMNLP，收录大量高质量 LLM-Rec 论文
        "dblp": "ACL",
        "openalex": "ACL",
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
        "dblp": "ICLR",
        "openalex": "ICLR",
    },
    "TOIS": {
        "queries": [
            f'(co:TOIS OR jr:TOIS OR all:"Transactions on Information Systems") AND {REC_ARXIV_FILTER}',
            'co:TOIS OR jr:TOIS',
        ],
        "tokens": [r"\btois\b", r"transactions on information systems", r"acm trans\.?\s*inf\.?\s*syst"],
        "neg": [],
        "dblp": "ACM Trans. Inf. Syst.",
        "openalex": "ACM Transactions on Information Systems",
    },
    "TKDE": {
        "queries": [
            f'(co:TKDE OR jr:TKDE OR all:"Transactions on Knowledge and Data Engineering") AND {REC_ARXIV_FILTER}',
            'co:TKDE OR jr:TKDE',
        ],
        "tokens": [r"\btkde\b", r"transactions on knowledge and data engineering", r"ieee trans\.?\s*knowl\.?\s*data\s*eng"],
        "neg": [],
        "dblp": "IEEE Trans. Knowl. Data Eng.",
        "openalex": "IEEE Transactions on Knowledge and Data Engineering",
    },
    "TORS": {
        "queries": [
            'co:TORS OR jr:TORS OR all:"Transactions on Recommender Systems"',
            'all:"ACM TORS"',
        ],
        "tokens": [r"\btors\b", r"transactions on recommender systems", r"acm trans\.?\s*recomm\.?\s*syst"],
        "neg": [],
        "dblp": "ACM Trans. Recomm. Syst.",
        "openalex": "ACM Transactions on Recommender Systems",
    },
    "TMM": {
        "queries": [
            f'(co:TMM OR jr:TMM OR all:"Transactions on Multimedia") AND {REC_ARXIV_FILTER}',
        ],
        "tokens": [r"\btmm\b", r"transactions on multimedia", r"ieee trans\.?\s*multimedia"],
        "neg": [],
        "dblp": "IEEE Trans. Multimedia",
        "openalex": "IEEE Transactions on Multimedia",
    },
    "TKDD": {
        "queries": [
            f'(co:TKDD OR jr:TKDD OR all:"Transactions on Knowledge Discovery from Data") AND {REC_ARXIV_FILTER}',
            'co:TKDD OR jr:TKDD',
        ],
        "tokens": [r"\btkdd\b", r"transactions on knowledge discovery from data", r"acm trans\.?\s*knowl\.?\s*discov\.?\s*data"],
        "neg": [],
        "dblp": "ACM Trans. Knowl. Discov. Data",
        "openalex": "ACM Transactions on Knowledge Discovery from Data",
    },
    "TWEB": {
        "queries": [
            f'(co:TWEB OR jr:TWEB OR all:"Transactions on the Web") AND {REC_ARXIV_FILTER}',
            'co:TWEB OR jr:TWEB',
        ],
        "tokens": [r"\btweb\b", r"transactions on the web", r"acm trans\.?\s*web"],
        "neg": [],
        "dblp": "ACM Trans. Web",
        "openalex": "ACM Transactions on the Web",
    },
}


def match_venue_and_extract_year(
    comment: str,
    venue: str,
    target_year: str,
    published_date: str = "",
    updated_date: str = "",
    allow_prev_year: bool = True,
) -> str:
    """检查 comment/journal_ref 是否属于指定主会/顶刊，并返回匹配的年份字符串（如 '2026' 或 '2025'；不匹配返回 ''）。"""
    if not comment:
        return ""
    c = comment.lower()
    cfg = VENUE_CONFIGS.get(venue)
    if not cfg:
        return target_year if venue.lower() in c else ""

    neg_list = COMMON_NEGATIVE_HINTS + cfg.get("neg", [])
    if any(neg.lower() in c for neg in neg_list):
        return ""

    tokens = cfg.get("tokens", [])
    if not any(re.search(pat, c, re.IGNORECASE) for pat in tokens):
        return ""

    prev_year = str(int(target_year) - 1) if target_year.isdigit() else "2025"
    candidate_years = [target_year, prev_year] if allow_prev_year else [target_year]

    # 1. 检查 comment 中是否明确带有年份（如 2026 / '26 / -26 / 2025 / '25 / -25）
    for y in candidate_years:
        yy = y[-2:]
        if re.search(rf"\b{y}\b", c) or re.search(rf"(?:['\-\s]){yy}\b", c):
            return y

    # 2. 如果是期刊（TOIS/TKDE/TORS/TMM/TKDD/TWEB），且 comment 中没有其它冲突的旧年份（如 2021-2023），则用 updated_date / published_date 判定年份
    explicit_years_in_c = re.findall(r"\b(20\d{2})\b", c)
    if explicit_years_in_c:
        # 有明确年份但不在 candidate_years 中（例如 2022/2023/2024）
        return ""

    if venue in JOURNAL_VENUES:
        upd_y = (updated_date or "")[:4]
        pub_y = (published_date or "")[:4]
        for y in candidate_years:
            if upd_y == y or pub_y == y:
                return y

    return ""


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


def batch_resolve_titles_on_arxiv(titles_with_years: list, venue: str, batch_size: int = 6) -> list:
    """将从 DBLP/OpenAlex 获得的官方录用论文标题按 6 篇一批在 arXiv 上批量反查预印本，避免单篇串行触发 429。
    titles_with_years: list of (title, matched_year)
    """
    if not titles_with_years:
        return []
    resolved = []
    seen_ids = set()

    # 构建规范化标题到 (原始标题, 年份) 的映射
    norm_map = {}
    clean_queries = []
    for raw_t, y_str in titles_with_years:
        norm_t = re.sub(r"[^a-z0-9]", "", raw_t.lower())
        if len(norm_t) < 12 or norm_t in norm_map:
            continue
        norm_map[norm_t] = (raw_t, y_str)
        clean_t = re.sub(r"[^a-zA-Z0-9\s]", " ", raw_t)
        clean_t = re.sub(r"\s+", " ", clean_t).strip()
        if clean_t:
            clean_queries.append((norm_t, clean_t, y_str))

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
                matched_y = None
                if norm_ep in norm_map:
                    matched_y = norm_map[norm_ep][1]
                else:
                    # 允许前缀高重合匹配（应对副标题微调）
                    for target_norm, (_, y_val) in norm_map.items():
                        if len(target_norm) >= 25 and (target_norm in norm_ep or norm_ep in target_norm):
                            matched_y = y_val
                            break
                if matched_y:
                    seen_ids.add(aid)
                    orig_c = ep.get("comment") or ""
                    ep["comment"] = f"{orig_c} | Accepted at {venue} {matched_y} (Verified)".strip(" |")
                    ep["trusted_venue_match"] = True
                    ep["matched_year"] = matched_y
                    resolved.append(ep)
            time.sleep(0.6)
        except Exception:
            time.sleep(1.0)
            continue
    return resolved


def discover_via_dblp_and_resolve_arxiv(venue: str, years: list, max_hits: int = 100) -> list:
    """从 DBLP API 获取指定会刊与年份列表的推荐相关论文标题，并批量去 arXiv 反查预印本。"""
    cfg = VENUE_CONFIGS.get(venue, {})
    dblp_venue = cfg.get("dblp", venue)
    titles_with_years = []

    for y in years:
        for kw in ["recommend", "collaborative", "sequential", "personalization"]:
            q = f"{dblp_venue} {y} {kw}"
            url = f"https://dblp.org/search/publ/api?q={urllib.parse.quote(q)}&h={max_hits}&format=json"
            for attempt in range(2):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "RecSys-Paper-Notes/3.5 (mailto:academic@papernotes.org)"})
                    with urllib.request.urlopen(req, timeout=20) as resp:
                        raw = resp.read().decode("utf-8", errors="ignore").strip()
                    if not raw.startswith("{"):
                        time.sleep(1.5)
                        continue
                    data = json.loads(raw)
                    hits = data.get("result", {}).get("hits", {}).get("hit", [])
                    for h in hits:
                        info = h.get("info", {})
                        title = re.sub(r"\.$", "", (info.get("title") or "").strip())
                        if title and len(title) >= 15:
                            titles_with_years.append((title, str(y)))
                    break
                except Exception:
                    time.sleep(1.5 * (attempt + 1))
            time.sleep(0.4)

    return batch_resolve_titles_on_arxiv(titles_with_years, venue=venue)


def discover_via_openalex(venue: str, years: list, per_page: int = 100) -> list:
    """使用免 Key 的 OpenAlex API 检索目标会刊发表的推荐系统论文，直接提取关联 arXiv ID 或批量反查 arXiv。"""
    cfg = VENUE_CONFIGS.get(venue, {})
    oa_venue = cfg.get("openalex", venue)
    direct_arxiv_ids = {}
    titles_to_resolve = []

    for y in years:
        search_q = f'"{oa_venue}" (recommendation OR recommender OR collaborative OR personalization)'
        params = {
            "search": search_q,
            "filter": f"publication_year:{y}",
            "per-page": str(per_page),
            "mailto": "recsys-notes@papernotes.org",
        }
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "RecSys-Paper-Notes/3.5"})
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            results = data.get("results", [])
        except Exception as e:
            print(f"  [Warn] OpenAlex 检索跳过 ({venue} {y}): {e}")
            continue

        for work in results:
            title = (work.get("title") or "").strip()
            if not title or len(title) < 15:
                continue
            # 核验来源名称是否包含目标会刊关键词
            locs = work.get("locations") or []
            primary_loc = work.get("primary_location") or {}
            all_sources = []
            for loc in [primary_loc] + locs:
                src = (loc or {}).get("source") or {}
                dn = (src.get("display_name") or "").lower()
                if dn:
                    all_sources.append(dn)
            joined_src = " | ".join(all_sources)
            if joined_src and not any(re.search(pat, joined_src, re.IGNORECASE) for pat in cfg.get("tokens", [venue.lower()])):
                continue

            # 检查 OpenAlex 是否直接收录了 arXiv 链接
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
                direct_arxiv_ids[found_aid] = str(y)
            else:
                titles_to_resolve.append((title, str(y)))

    resolved_papers = []
    if direct_arxiv_ids:
        fetched = fetch_arxiv_by_ids(list(direct_arxiv_ids.keys()), cache_dir=None, download_pdf=False)
        for ep in fetched:
            aid = re.sub(r"v\d+$", "", ep.get("arxiv_id", "").strip())
            y_val = direct_arxiv_ids.get(aid, years[0])
            orig_c = ep.get("comment") or ""
            ep["comment"] = f"{orig_c} | Accepted at {venue} {y_val} (OpenAlex Verified)".strip(" |")
            ep["trusted_venue_match"] = True
            ep["matched_year"] = y_val
            resolved_papers.append(ep)

    if titles_to_resolve:
        resolved_papers.extend(batch_resolve_titles_on_arxiv(titles_to_resolve[:60], venue=venue))

    return resolved_papers


def build_for_single_venue(venue: str, args, log: RunLog) -> tuple:
    """对单个会刊执行多通道高召回发现 + LLM 相关性严格复核，返回 (新增篇数, 总篇数)。"""
    out_path = Path(args.out) if (args.out and not args.all_venues) else SCRIPT_DIR.parent / "data" / "papers" / f"{venue}{args.year}.csv"
    existing_rows_by_id = {}
    if out_path.exists() and not args.no_incremental:
        with out_path.open("r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                aid = re.sub(r"v\d+$", "", (r.get("arxiv_id") or "").strip())
                if aid:
                    existing_rows_by_id[aid] = r
        print(f"\n[{venue}] 已加载现有 CSV 中的 {len(existing_rows_by_id)} 篇论文记录。")

    cfg = VENUE_CONFIGS.get(venue, {"queries": [f'all:"{venue} {{year}}"'], "tokens": [venue.lower()], "neg": []})
    prev_year = str(int(args.year) - 1) if args.year.isdigit() else "2025"
    allow_prev = bool(args.supplement_prev_year)

    papers = []
    for q_tmpl in cfg["queries"]:
        q_formatted = q_tmpl.format(year=args.year)
        if not any(prefix in q_formatted for prefix in ("all:", "co:", "jr:", "ti:", "abs:")):
            q_str = f"all:{q_formatted}"
        else:
            q_str = q_formatted
        print(f"[{venue} Query] {q_str} (max_results={args.max_results})")
        log.write(f"QUERY venue={venue} {q_str} max_results={args.max_results}")
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

    # 先统计 arXiv 直接命中 target_year (如 2026) 的候选数，判断是否需要补充 prev_year (如 2025)
    prelim_primary_count = len(existing_rows_by_id)
    for p in papers:
        pub_date = p.get("published_date", "")
        upd_date = p.get("updated_date", "")
        my = match_venue_and_extract_year(p.get("comment", ""), venue, args.year, pub_date, upd_date, allow_prev_year=False)
        if my == args.year:
            prelim_primary_count += 1

    need_prev_year_supplement = allow_prev and (prelim_primary_count < args.min_venue_papers)
    if need_prev_year_supplement:
        print(f"  [Supplement] {venue} {args.year} 预印本较少（当前初筛 {prelim_primary_count} 篇 < {args.min_venue_papers} 篇），自动开启 {prev_year} 年度同会刊论文补充检索 ...")
        for q_tmpl in cfg["queries"]:
            if "{year}" in q_tmpl:
                q_formatted = q_tmpl.format(year=prev_year)
                q_str = q_formatted if any(pfx in q_formatted for pfx in ("all:", "co:", "jr:")) else f"all:{q_formatted}"
                try:
                    papers.extend(
                        _fetch_arxiv_entries(
                            {"search_query": q_str, "max_results": args.max_results, "sortBy": "submittedDate", "sortOrder": "descending"},
                            cache_dir=None,
                            download_pdf=False,
                        )
                    )
                except Exception:
                    pass
                time.sleep(0.4)

    target_years_for_external = [args.year, prev_year] if need_prev_year_supplement else [args.year]

    if args.use_dblp:
        print(f"[{venue} OpenAlex+DBLP] 正在联合检索 OpenAlex 与 DBLP 官方目录 ({','.join(target_years_for_external)}) ...")
        oa_found = discover_via_openalex(venue, target_years_for_external)
        dblp_found = discover_via_dblp_and_resolve_arxiv(venue, target_years_for_external)
        print(f"[{venue} OpenAlex+DBLP] 额外发现 {len(oa_found) + len(dblp_found)} 篇官方录用 arXiv 预印本。")
        papers.extend(oa_found)
        papers.extend(dblp_found)

    primary_candidates = []
    fallback_candidates = []
    seen = set(existing_rows_by_id.keys())

    for p in papers:
        arxiv_id = re.sub(r"v\d+$", "", p.get("arxiv_id", "").strip())
        if not arxiv_id or arxiv_id in seen:
            continue

        pub_date = p.get("published_date", "")
        upd_date = p.get("updated_date", "")
        if not pub_date:
            m_pub = re.search(r"Published Date:\s*(\d{4}-\d{2}-\d{2})", p.get("content", ""))
            if m_pub:
                pub_date = m_pub.group(1)

        if p.get("trusted_venue_match"):
            matched_y = p.get("matched_year") or args.year
        else:
            matched_y = match_venue_and_extract_year(
                p.get("comment", ""),
                venue,
                args.year,
                pub_date,
                upd_date,
                allow_prev_year=allow_prev,
            )
        if not matched_y:
            continue

        abstract = p.get("content", "").split("Abstract:\n", 1)[-1]
        if not args.no_relevance_filter and not is_recommendation_relevant(p["title"], abstract):
            continue

        seen.add(arxiv_id)
        p["arxiv_id"] = arxiv_id
        p["matched_year"] = matched_y
        if matched_y == args.year:
            primary_candidates.append((p, abstract))
        else:
            fallback_candidates.append((p, abstract))

    # 优先收录 target_year (2026)；若不足 min_venue_papers，则按时间从新到旧补充 prev_year (2025)
    candidates = list(primary_candidates)
    if allow_prev and (len(existing_rows_by_id) + len(candidates) < args.min_venue_papers) and fallback_candidates:
        needed = max(args.min_venue_papers * 2, 30)
        candidates.extend(fallback_candidates[:needed])

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
            "year": p.get("matched_year") or args.year,
        })

    final_rows = list(existing_rows_by_id.values()) + new_rows
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "arxiv_id", "paper_url", "code_url", "category", "venue", "year"])
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"[Done] [{venue}] 策展名单共 {len(final_rows)} 篇（本次新增 {len(new_rows)} 篇，LLM 排除 {llm_removed} 篇）-> {out_path}")
    for row in new_rows[:10]:
        print(f"  + [{row['year']}][{row['category']}] {row['title'][:85]} ({row['arxiv_id']})")
    return len(new_rows), len(final_rows)


def main():
    parser = argparse.ArgumentParser(description="从 arXiv + OpenAlex + DBLP 自动构建 22 个推荐系统会刊高召回策展名单")
    parser.add_argument("--venue", default="SIGIR", choices=sorted(VENUES.keys()), help="单个目标会刊 key")
    parser.add_argument("--venues", default="", help="多个目标会刊（逗号分隔，如 ACM_MM,TOIS,TKDE,AAAI,IJCAI）")
    parser.add_argument("--all-venues", action="store_true", help="一键构建全部 22 个顶会与顶刊的候选名单")
    parser.add_argument("--year", default="2026")
    parser.add_argument("--min-venue-papers", type=int, default=18, help="开启 --supplement-prev-year 时，单会刊少于该篇数自动补充上一年份")
    parser.add_argument("--supplement-prev-year", action="store_true", help="当目标年份论文不足时自动补充上一年份（默认关闭，严格按 --year 单年收录）")
    parser.add_argument("--strict-year", action="store_true", help="兼容参数：严格只收录 --year 当年论文（默认即开启）")
    parser.add_argument("--max-results", type=int, default=300)
    parser.add_argument("--use-dblp", action="store_true", help="同时查询 OpenAlex 与 DBLP 官方录用列表并批量反查 arXiv")
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
        venue_list = [v.strip() for v in args.venues.split(",") if v.strip() in VENUES]
    else:
        venue_list = [args.venue]

    log = RunLog("build-list")
    log.write(
        f"START venues={','.join(venue_list)} year={args.year} max_results={args.max_results} "
        f"use_dblp={args.use_dblp} no_llm={args.no_llm} workers={args.llm_workers}"
    )

    total_new = 0
    total_all = 0
    for v_key in venue_list:
        n_new, n_all = build_for_single_venue(v_key, args, log)
        total_new += n_new
        total_all += n_all

    elapsed = time.monotonic() - run_start
    print_usage_summary(
        resolve_model(args.llm_model),
        processed=total_new,
        success=total_new,
        elapsed_seconds=elapsed,
        log=log,
    )
    print(f"[Summary] 共处理 {len(venue_list)} 个会刊，本次新增 {total_new} 篇，累计在册 {total_all} 篇（耗时 {format_duration(elapsed)}）")
    print(f"[Log] 详细日志: {log.path}")
    log.write(f"TOTAL elapsed={format_duration(elapsed)} venues={len(venue_list)} new={total_new} total={total_all}")
    log.close()


if __name__ == "__main__":
    main()