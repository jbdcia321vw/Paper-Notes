#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_recsys_notes.py (v3.3 高密度精读 Note + 图表“按需精选内嵌、不堆砌文末、自动清理未用图片”版)
核心改进：
  1. 彻底移除文末堆砌的 `## 论文图表`：
     - 仅当大模型认为某张图（如核心模型架构图、关键动机对比图）适合放在对应正文段落位置时，才在正文中内嵌最多 1–3 张图片。
     - 未被正文引用的提取图片不再追加到文末，且会自动从 `docs/assets/papers/` 中删除未使用的图片文件，防止仓库体积膨胀。
     - 自动过滤掉冗余的表格截图（`[[TABLE:n]]`），因为数据集与实验结果已用原生 Markdown 表格呈现。
  2. 提示词定位为「1,800–2,600 字高密度精读笔记 (Concise Paper Note)」，拒绝流水账。
"""

import csv
import html
import os
import posixpath
import re
import sys
import json
import time
import threading
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET
import argparse
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from init_clean_recsys_repo import VENUES, CATEGORIES, SUPPORTED_YEARS, DEFAULT_YEAR
from extract_pdf_figures import extract_figures
from run_logging import ProgressReporter, RunLog, format_duration, format_money

# 16 大推荐系统正交细分类别的代表性检索词（用于 --seed-all-categories）
CATEGORY_SEED_QUERIES = {
    "collaborative_filtering": "collaborative filtering two-tower retrieval recommendation",
    "graph_rec": "graph neural network recommendation knowledge graph LightGCN",
    "self_supervised_denoising": "self-supervised contrastive learning denoising recommendation implicit feedback",
    "sequential_rec": "sequential recommendation session-based recommendation transformer mamba",
    "generative_rec": "generative recommendation semantic ID RQ-VAE diffusion recommendation",
    "llm_rec": "large language model recommendation LLM4Rec instruction tuning",
    "agent_conversational_rec": "conversational recommender system agent recommendation user simulation",
    "multimodal_rec": "multimodal recommendation visual acoustic graph alignment",
    "social_group_rec": "social recommendation group recommendation trust network",
    "cold_start_rec": "cold-start recommendation zero-shot long-tail item recommendation",
    "cross_domain_multi_scenario": "cross-domain recommendation multi-scenario multi-domain recommendation",
    "poi_spatiotemporal_rec": "next POI recommendation point-of-interest spatio-temporal trajectory",
    "ctr_feature_interaction": "click-through rate prediction CTR feature interaction delayed feedback",
    "multi_task_reranking": "multi-task recommendation listwise reranking pareto recommendation",
    "rl_bandit_rec": "reinforcement learning recommendation contextual bandit long-term engagement",
    "debias_fairness_industrial": "debiasing recommendation popularity bias causal fairness recommender",
}

CATEGORY_LIST_STR = "\n".join(
    [f"- `{slug}`: {info['title']} ({info['en_title']}) — {info['desc']}" for slug, info in CATEGORIES.items()]
)

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_MAX_INPUT_CHARS = 65000
DEFAULT_MAX_OUTPUT_TOKENS = 8192

USAGE_LOCK = threading.Lock()
USAGE_TOTALS = {
    "requests": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "prompt_cache_hit_tokens": 0,
    "prompt_cache_miss_tokens": 0,
    "cost_usd": 0.0,
}

# DeepSeek 官方定价（USD / 1M tokens，高峰价；非高峰为 5 折）
PRICING_USD_PER_M = {
    "deepseek-flash": {
        "cache_hit": 0.006,
        "cache_miss": 0.30,
        "output": 1.20,
    },
    "deepseek-v4-pro": {
        "cache_hit": 0.044,
        "cache_miss": 1.32,
        "output": 3.96,
    },
}


def _as_int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def record_usage(usage, model: str = "") -> None:
    if not isinstance(usage, dict):
        return
    prompt = _as_int(usage.get("prompt_tokens"))
    completion = _as_int(usage.get("completion_tokens"))
    total = _as_int(usage.get("total_tokens")) or (prompt + completion)
    cache_hit = _as_int(usage.get("prompt_cache_hit_tokens"))
    cache_miss = _as_int(usage.get("prompt_cache_miss_tokens"))
    details = usage.get("prompt_tokens_details")
    if not cache_hit and isinstance(details, dict):
        cache_hit = _as_int(details.get("cached_tokens"))
    if not cache_miss:
        cache_miss = max(prompt - cache_hit, 0)
    cost = _usage_cost_usd(
        {
            "prompt_cache_hit_tokens": cache_hit,
            "prompt_cache_miss_tokens": cache_miss,
            "completion_tokens": completion,
        },
        model,
    )
    with USAGE_LOCK:
        USAGE_TOTALS["requests"] += 1
        USAGE_TOTALS["prompt_tokens"] += prompt
        USAGE_TOTALS["completion_tokens"] += completion
        USAGE_TOTALS["total_tokens"] += total
        USAGE_TOTALS["prompt_cache_hit_tokens"] += cache_hit
        USAGE_TOTALS["prompt_cache_miss_tokens"] += cache_miss
        if cost is not None:
            USAGE_TOTALS["cost_usd"] += cost


def is_peak_now() -> bool:
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    return (1 <= now.hour < 4) or (6 <= now.hour < 10)


def _pricing_for_model(model: str):
    name = (model or "").lower()
    if "pro" in name:
        return PRICING_USD_PER_M["deepseek-v4-pro"]
    if "deepseek" in name or not name:
        return PRICING_USD_PER_M["deepseek-flash"]
    return None


def _usage_cost_usd(usage: dict, model: str):
    prices = _pricing_for_model(model)
    if not prices:
        return None
    multiplier = 1.0 if is_peak_now() else 0.5
    hit = _as_int(usage.get("prompt_cache_hit_tokens"))
    miss = _as_int(usage.get("prompt_cache_miss_tokens"))
    completion = _as_int(usage.get("completion_tokens"))
    return multiplier * (
        hit / 1_000_000 * prices["cache_hit"]
        + miss / 1_000_000 * prices["cache_miss"]
        + completion / 1_000_000 * prices["output"]
    )


def estimate_usage_cost_usd(model: str):
    prices = _pricing_for_model(model)
    if not prices:
        return None
    if USAGE_TOTALS["cost_usd"] > 0:
        return USAGE_TOTALS["cost_usd"]
    return _usage_cost_usd(USAGE_TOTALS, model)


def print_usage_summary(
    model: str = "",
    processed: int = 0,
    success: int = 0,
    elapsed_seconds: float = None,
    log: RunLog = None,
) -> None:
    if not USAGE_TOTALS["requests"] and elapsed_seconds is None:
        return

    lines = []
    if USAGE_TOTALS["requests"]:
        prompt = USAGE_TOTALS["prompt_tokens"]
        hit = USAGE_TOTALS["prompt_cache_hit_tokens"]
        hit_rate = (hit / prompt * 100) if prompt else 0.0
        lines.append(
            "[Usage] requests={requests} prompt={prompt} completion={completion} "
            "total={total} cache_hit={hit} ({rate:.1f}%)".format(
                requests=USAGE_TOTALS["requests"],
                prompt=prompt,
                completion=USAGE_TOTALS["completion_tokens"],
                total=USAGE_TOTALS["total_tokens"],
                hit=hit,
                rate=hit_rate,
            )
        )

    cost = estimate_usage_cost_usd(model)
    if cost is not None and USAGE_TOTALS["requests"]:
        lines.append(
            f"[Cost] 本次总花费约 {format_money(cost)}"
            "（按请求时高峰/非高峰价格估算；中国法定节假日未单独扣除）"
        )
        if processed > 0:
            lines.append(
                f"[Cost] 平均每篇（按 {processed} 篇任务）约 {format_money(cost / processed)}"
            )
        if success > 0 and success != processed:
            lines.append(
                f"[Cost] 平均每篇成功生成（按 {success} 篇）约 {format_money(cost / success)}"
            )

    if elapsed_seconds is not None:
        lines.append(f"[Time] 总耗时 {format_duration(elapsed_seconds)}")
        if processed > 0:
            lines.append(
                f"[Time] 平均每篇（按 {processed} 篇任务）{format_duration(elapsed_seconds / processed)}"
            )
        if success > 0 and success != processed:
            lines.append(
                f"[Time] 平均每篇成功生成（按 {success} 篇）{format_duration(elapsed_seconds / success)}"
            )

    print()
    for line in lines:
        print(line)
        if log:
            log.write(line)


def resolve_api_key(explicit: str = "") -> str:
    return (
        explicit
        or os.environ.get("LLM_API_KEY", "")
        or os.environ.get("DEEPSEEK_API_KEY", "")
    ).strip()


def resolve_base_url(explicit: str = "") -> str:
    return (
        explicit
        or os.environ.get("LLM_BASE_URL", "")
        or os.environ.get("DEEPSEEK_BASE_URL", "")
        or DEFAULT_BASE_URL
    ).strip()


def resolve_model(explicit: str = "") -> str:
    return (
        explicit
        or os.environ.get("LLM_MODEL", "")
        or os.environ.get("DEEPSEEK_MODEL", "")
        or DEFAULT_MODEL
    ).strip()


def call_chat_completion(
    messages: list,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float = 0.0,
    timeout: int = 240,
    response_format: dict = None,
    max_tokens: int = None,
    thinking: bool = False,
) -> str:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {"model": model, "messages": messages, "temperature": temperature}
    if response_format:
        payload["response_format"] = response_format
    if max_tokens and max_tokens > 0:
        payload["max_tokens"] = max_tokens
    if thinking is not None:
        payload["thinking"] = {"type": "enabled" if thinking else "disabled"}

    def _post(request_payload):
        data = json.dumps(request_payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        res_json = _post(payload)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:1200]
        if e.code == 400:
            if payload.get("max_tokens", 0) > 8192:
                payload["max_tokens"] = 8192
            else:
                payload.pop("thinking", None)
            try:
                res_json = _post(payload)
            except urllib.error.HTTPError as e2:
                payload.pop("thinking", None)
                payload.pop("max_tokens", None)
                try:
                    res_json = _post(payload)
                except urllib.error.HTTPError as retry_error:
                    retry_body = retry_error.read().decode("utf-8", errors="replace")[:1200]
                    raise RuntimeError(f"LLM HTTP {retry_error.code}: {retry_body}") from retry_error
        else:
            raise RuntimeError(f"LLM HTTP {e.code}: {body}") from e

    record_usage(res_json.get("usage") if isinstance(res_json, dict) else None, model)
    try:
        choice = res_json["choices"][0]
        return (choice["message"].get("content") or "").strip()
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"LLM 响应格式异常: {str(res_json)[:800]}") from e


SYSTEM_PROMPT = f"""你是一位推荐系统（Recommender Systems）与信息检索（Information Retrieval）领域的资深研究员。
你的任务是将输入的学术论文提炼为一篇**高信息密度、5分钟即可读懂核心思想的精读笔记（Concise Paper Note）**，而不是写冗长的全文翻译或逐段复述的流水账！

### 一、16 大推荐系统正交分类归档
请从以下 16 个推荐系统细分方向中，为该论文选择 **1 个主分类** 和 **最多 1-2 个交叉分类**：
{CATEGORY_LIST_STR}

### 二、「高密度精读 Note」写作铁律（严禁流水账，严控篇幅！）
1. **定位是“精炼笔记 (Note)”，全文总字数严格控制在 1,800–2,600 中文字符以内**：
   - **砍掉一切常识性流水账**：不要单独开章节罗列常规数学符号（如解释什么是用户集 $\\mathcal{{U}}$、物品集 $\\mathcal{{I}}$），不要抄写常规 Transformer Attention、GCN 邻接矩阵归一化或标准 BPR/InfoNCE 的基础教科书定义！
   - **只写本文的“增量创新 (Delta)”**：每个模块只用 2-3 句话讲清楚 **「前人卡在哪里 $\\to$ 本文改了哪一步算子/损失 $\\to$ 核心公式长什么样 $\\to$ 为什么这样改有效（物理直觉）」**。
   - 若开启了思考模式（Thinking），思考过程请控制在 800 字以内，把输出配额留给完整的实验表格与洞察。
2. **推荐系统核心要素一个不能少（短小但刀刀见血）**：
   - **评估协议必查**：用精炼短句点明数据集过滤（如 `5-core`）、划分方式（`Leave-One-Out` / `比例划分` / `全局时间戳切分` / `冷启动划分`）、候选打分协议（无偏 `Full-Ranking 全库排序` vs. `Sampled 负采样排序`）以及训练负采样策略。
   - **实验表格必全**：主实验表保留原文 **5–8 个最具代表性的主流 Baseline** 与核心指标数值（加粗最优并标出 `Improv.%`），消融实验单独列一张紧凑表格，并在表后写出 **3 条结合具体数字的归因发现（Key Findings）**。

### 三、输出 Markdown 模板规范（第 1 行必须是 `---`，严格使用以下二级标题！）
---
title: "[Paper Note] 论文英文原名"
description: "[会议或期刊名][主分类中文名] 一句话概括核心方法、评估协议与关键收益（80字以内）"
tags:
  - 会议或期刊名
  - 主分类中文名
  - 推荐链路阶段(如召回/粗排/精排/重排/生成式端到端)
  - 核心技术词1
  - 核心技术词2
date: {datetime.now().strftime('%Y-%m-%d')}
venue: 会议/期刊 key (如 SIGIR / RecSys / KDD / TOIS)
year: 发表年份 (如 2026)
paper_url: "论文原文完整URL"
code_url: "开源代码完整URL(若无则留空字符串)"
categories:
  - 主分类英文slug(必须严格属于上述16个slug之一)
  - 次分类英文slug(可选，严格属于上述16个slug之一)
---

# 论文英文原名 (附中文直译标题)

- **会议/期刊**: 会议或期刊名
- **作者与机构**: 论文主要作者与核心机构
- **推荐链路定位**: 召回 / 粗排 / 精排 / 重排 / 端到端生成式
- **研究方向**: 主分类中文名 / 次分类中文名
- **论文原文**: [论文原文链接](论文原文完整URL)
- **开源代码**: [开源代码仓库链接](开源代码完整URL) （若无开源代码则写“暂未开源”）

## TL;DR (1分钟核心速览)

用 120–160 字直接点明：本文针对推荐系统什么核心痛点？提出了什么核心方法？在什么评估协议（如 Full-Ranking）和基准数据集上取得了多少相对提升？

## Background & Motivation (背景与核心矛盾)

（每条严格控制在 1-2 句话以内，直奔主题：）
- **主流范式**: 当前该子方向的主流做法是什么？
- **前人局限**: 现有代表作在什么具体场景或数据分布下会失效（列出 2 个核心矛盾）？
- **核心洞察 (Key Insight)**: 作者观察到了什么关键现象或理论性质？
- **本文切入点**: 一句话说明本文如何破局。

## Method (核心方法设计)

### 1. 整体框架 (Overall Architecture)

用一段精炼文字（150–220字）串起模型从输入、核心模块交互到训练与推理解码的完整闭环，并附上一个简洁的 Mermaid 流程图：

```mermaid
flowchart TD
    A["输入特征/历史序列"] --> B["核心创新模块"]
    B --> C["预测打分/生成解码"]
```

### 2. 关键设计拆解 (Key Designs)

只提炼本文最重要的 **2–3 个核心创新点**（严禁罗列常规预处理或普通网络层；每点控制在 150 字以内 + 1 个最核心的数学公式）：
1. **创新设计一（名称 + 一句话作用）**：痛点起因 $\\to$ 核心算子与公式 $\\to$ 为什么有效的直觉。
2. **创新设计二（名称 + 一句话作用）**：痛点起因 $\\to$ 核心算子与公式 $\\to$ 为什么有效的直觉。
3. **训练目标、负采样与推理效率**：写出总损失函数 $\\mathcal{{L}}$、训练负采样策略（Uniform / In-batch / Hard Negative / 免负采样）及在线推理开销。

## Experimental Setup & Evaluation Protocol (实验设置与评估协议)

- **数据集与预处理**: 列出数据集统计表及过滤方式（如 `5-core`）：

| 数据集 | #Users | #Items | #Interactions | 稀疏度/密度 | 特征/场景说明 |
| :--- | :---: | :---: | :---: | :---: | :--- |

- **数据划分与防穿越 (Splitting Strategy)**: 标注是 `Leave-One-Out`、`比例划分`、`全局时间戳切分` 还是 `冷启动划分`。
- **排名评测协议 (Ranking Protocol)**: 标注是无偏 `Full-Ranking (全库未交互排序)` 还是 `Sampled Ranking (采样排序，注明正负比)`。
- **对比基线与超参设置**: 简述对比的基线类别、Embedding 维度 $d$ 及显著性检验标准（如 $p < 0.05$）。

## Key Experimental Results (实验关键数据)

### 1. 主实验对比 (Main Results)

提取原文主实验表（保留 **5–8 个代表性 SOTA 基线** 与核心指标列，加粗最优值，标出提升幅度 `Improv.`）：

### 2. 消融实验 (Ablation Study)

用紧凑的 Markdown 表格展示各核心组件移除后（`w/o ...`）的性能变化。

### 3. 关键实验发现 (Key Findings)

结合主实验、消融实验、长尾/稀疏分桶实验或线上 A/B 测试的具体数字，写出 **3 条一针见血的定量归因结论**（每条 1-2 句话）。

## Highlights, Limitations & Rating (亮点、对比与审稿评价)

- **与代表作对比 (vs. Prior Work)**: 用 2-3 行简明对比本文与最相关的 2-3 篇前人基线的本质区别。
- **可迁移亮点**: 1-2 个可直接迁移到其他推荐任务的技巧。
- **审稿级局限批判**: 客观指出 2 个潜在短板（如推理延迟、评估局限或超参敏感度）。
- **四维评分**: 创新性 ⭐⭐⭐⭐ | 实验扎实度 ⭐⭐⭐⭐ | 方法清晰度 ⭐⭐⭐⭐ | 落地价值 ⭐⭐⭐⭐

### 四、图表按需内嵌规范（宁缺毋滥，严禁堆砌！）
- 如果提供了可用图表占位符（如 `[[FIG:1]]`），**仅当某张图（如核心架构图或关键动机对比图）确实非常适合放在某一段文字正下方辅助理解时，才在对应段落后单独另起一行插入最多 1–2 个占位符**。
- **如果没有特别合适的位置，或者图片只是次要曲线/表格截图，请直接忽略不要插入任何占位符（不是非放不可）！严禁在文末集中罗列图片！**
- 正文引用图表时统一使用中文“图 n / 表 n”，所有图注（figcaption）必须使用中文。
"""


def extract_code_and_paper_urls(text: str, default_paper_url: str = "") -> tuple:
    code_matches = re.findall(
        r"https?://(?:github\.com|gitlab\.com|huggingface\.co|gitee\.com)/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+",
        text,
    )
    ignore_repos = {"huggingface.co/datasets", "github.com/pytorch", "github.com/tensorflow", "github.com/google"}
    clean_code_urls = []
    for u in code_matches:
        u_clean = u.rstrip(".,;:)]}\"'")
        if not any(ig in u_clean.lower() for ig in ignore_repos) and u_clean not in clean_code_urls:
            clean_code_urls.append(u_clean)

    code_url = clean_code_urls[0] if clean_code_urls else ""

    paper_url = default_paper_url
    if not paper_url:
        arxiv_m = re.search(r"arXiv:(\d{4}\.\d{4,5})", text, re.IGNORECASE)
        if arxiv_m:
            paper_url = f"https://arxiv.org/abs/{arxiv_m.group(1)}"
        else:
            doi_m = re.search(r"https?://doi\.org/10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", text)
            if doi_m:
                paper_url = doi_m.group(0).rstrip(".,;:)]}\"'")

    return paper_url, code_url, clean_code_urls


def _normalize_unicode_escapes(text: str) -> str:
    decoded = text
    if "\\u" in text:
        decoded = re.sub(
            r"\\u([0-9a-fA-F]{4})",
            lambda m: chr(int(m.group(1), 16)),
            text,
        )
        try:
            decoded = decoded.encode("utf-16", "surrogatepass").decode("utf-16")
        except UnicodeDecodeError:
            decoded = decoded.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    return decoded.encode("utf-8", "ignore").decode("utf-8")


def _fetch_arxiv_html_text(arxiv_id: str, max_chars: int = 80000) -> str:
    if not arxiv_id:
        return ""
    url = f"https://arxiv.org/html/{arxiv_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "RecSys-Paper-Notes/3.3"})
        with urllib.request.urlopen(req, timeout=25) as resp:
            if resp.status != 200:
                return ""
            raw_html = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw_html, "html.parser")
        for math_tag in soup.find_all("math"):
            alt = math_tag.get("alttext")
            if alt:
                math_tag.replace_with(f" ${alt.strip()}$ ")

        for table in soup.find_all("table"):
            rows_md = []
            for tr in table.find_all("tr"):
                cells = [re.sub(r"\s+", " ", c.get_text(" ", strip=True)) for c in tr.find_all(["th", "td"])]
                if any(cells):
                    rows_md.append("| " + " | ".join(cells) + " |")
            if rows_md:
                ncols = rows_md[0].count("|") - 1
                if ncols > 0:
                    sep = "| " + " | ".join(["---"] * ncols) + " |"
                    rows_md.insert(1, sep)
                table.replace_with("\n\n" + "\n".join(rows_md) + "\n\n")

        for tag in soup.find_all(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        for bib in soup.select(".ltx_bibliography, #bib"):
            bib.decompose()

        article = soup.find("article") or soup.find("main") or soup.body or soup
        text = article.get_text("\n", strip=False)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return _normalize_unicode_escapes(text.strip())[:max_chars]
    except Exception:
        return ""


def extract_text_from_pdf(pdf_path: Path, max_chars: int = 80000) -> str:
    try:
        import pdfplumber
        pages_out = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                p_txt = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
                tables = page.extract_tables() or []
                t_strs = []
                for tb in tables:
                    clean_rows = []
                    for row in tb:
                        if not row:
                            continue
                        cells = [re.sub(r"\s+", " ", (c or "").strip()) for c in row]
                        if any(cells):
                            clean_rows.append("| " + " | ".join(cells) + " |")
                    if len(clean_rows) >= 2:
                        t_strs.append("\n".join(clean_rows))
                if t_strs:
                    p_txt += "\n\n[Extracted Page Tables]:\n" + "\n\n".join(t_strs)
                pages_out.append(p_txt)
        full_text = "\n\n".join(pages_out)
        if len(full_text.strip()) > 500:
            return _normalize_unicode_escapes(full_text)[:max_chars]
    except Exception:
        pass

    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        pages_text = []
        for page in reader.pages:
            pages_text.append(page.extract_text() or "")
        full_text = "\n".join(pages_text)
        return _normalize_unicode_escapes(full_text)[:max_chars]
    except Exception as e:
        print(f"  [Warn] 解析 PDF 失败 ({pdf_path.name}): {e}")
        return ""


def compact_paper_content(text: str, max_chars: int = DEFAULT_MAX_INPUT_CHARS) -> str:
    if not text:
        return text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    half_pos = len(text) // 2
    ref_match = re.search(
        r"\n\s*(?:\d+\.?\s+|I+V?X*\.?\s+)?(?:references|bibliography)\s*\n",
        text[half_pos:],
        re.IGNORECASE,
    )
    if ref_match:
        ref_start = half_pos + ref_match.start()
        tail = text[ref_start:]
        app_match = re.search(
            r"\n\s*(?:appendix|appendices|supplementary\s+material|[A-Z]\.\s+[A-Z])",
            tail[200:],
            re.IGNORECASE,
        )
        if app_match:
            app_start = ref_start + 200 + app_match.start()
            text = text[:ref_start] + "\n\n[... References Omitted ...]\n\n" + text[app_start:]
        else:
            text = text[:ref_start]

    if max_chars and max_chars > 0:
        text = text[:max_chars]
    return text


def _zh_figure_label(label: str) -> str:
    label = label or ""
    m = re.search(r"\b(?:Figure|Fig\.?)\s*(\d+)", label, re.IGNORECASE)
    if m:
        return f"图 {m.group(1)}"
    m = re.search(r"\bTable\s*(\d+)", label, re.IGNORECASE)
    if m:
        return f"表 {m.group(1)}"
    return "论文图表"


def translate_figure_captions(figures: list, api_key: str, base_url: str, model: str) -> list:
    if not figures:
        return figures
    if not api_key:
        for fig in figures:
            fig.setdefault("caption_zh", _zh_figure_label(fig.get("label")))
        return figures

    captions = {
        fig["token"]: re.sub(r"\s+", " ", fig.get("caption", "")).strip()
        for fig in figures
        if fig.get("caption")
    }
    if not captions:
        for fig in figures:
            fig.setdefault("caption_zh", _zh_figure_label(fig.get("label")))
        return figures

    prompt = (
        "请把下列论文图注翻译成简洁、准确的中文。保留 Figure/Table 编号（分别译为“图/表”）、"
        "数学符号、数据集名、模型名与专有名词，不要添加原文没有的信息。"
        "所有图注必须输出中文，不得保留英文原句。"
        "只输出一个 JSON 对象，键为 token，值为中文图注，不要输出 Markdown 代码块。\n\n"
        + json.dumps(captions, ensure_ascii=False, indent=2)
    )
    try:
        raw = call_chat_completion(
            [{"role": "user", "content": prompt}],
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0.0,
            timeout=120,
            response_format={"type": "json_object"},
            max_tokens=2048,
            thinking=False,
        )
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("响应不是 JSON 对象")
        mapping = data.get("translations") if isinstance(data.get("translations"), dict) else data
        for fig in figures:
            zh = mapping.get(fig["token"])
            if isinstance(zh, str) and zh.strip():
                fig["caption_zh"] = re.sub(r"\s+", " ", zh).strip()
            else:
                fig["caption_zh"] = _zh_figure_label(fig.get("label"))
    except Exception as e:
        print(f"  [Warn] 图注翻译失败，使用中文编号兜底: {e}")
        for fig in figures:
            fig.setdefault("caption_zh", _zh_figure_label(fig.get("label")))
    return figures


def format_figure_context(figures: list) -> str:
    lines = []
    for fig in figures:
        zh = re.sub(r"\s+", " ", fig.get("caption_zh") or "").strip()
        en = re.sub(r"\s+", " ", fig.get("caption") or "").strip()
        label = _zh_figure_label(fig.get("label"))
        if zh and zh != label:
            caption = zh
        elif en:
            caption = f"{zh or label}: {en}" if zh else en
        else:
            caption = fig.get("label") or "Paper figure"
        if len(caption) > 320:
            caption = caption[:317] + "..."
        lines.append(f"- {fig['token']} {caption}")
    return "\n".join(lines)


def figure_html(fig: dict, rel_path: str) -> str:
    caption = fig.get("caption_zh") or fig.get("caption") or fig.get("label") or "论文图表"
    alt = html.escape(caption, quote=False).replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]").replace("\n", " ")
    cap = html.escape(caption, quote=False)
    src = urllib.parse.quote(rel_path, safe="/.-_~")
    return (
        '<figure class="paper-figure" markdown="span">\n'
        f'  ![{alt}]({src})<figcaption>{cap}</figcaption>\n'
        "</figure>"
    )


def apply_figure_placeholders(md_note: str, figures: list, rel_paths: dict, asset_dir: Path = None) -> tuple:
    """仅替换正文中被 LLM 主动放置在合适段落位置的 `[[FIG:n]]` 占位符。
    未被正文引用的图表绝不堆砌在文末，并自动从磁盘清理未引用的图片文件，保持笔记与仓库干净清爽。
    返回 (替换后的 md_note, 实际被使用的 figures 列表)。
    """
    if not figures:
        return md_note, []

    used_figs = []
    used_tokens = set()
    for fig in figures:
        token = fig["token"]
        if token not in md_note:
            continue
        rel_path = rel_paths.get(token)
        if not rel_path:
            continue
        used_tokens.add(token)
        used_figs.append(fig)
        # 仅替换第一次出现，清除重复出现的同一 token
        md_note = md_note.replace(token, figure_html(fig, rel_path), 1)
        md_note = md_note.replace(token, "")

    md_note = re.sub(
        r"(</figure>)[ \t]*(?:图|表|Figure|Table)\s*\d+[^\n]*",
        r"\1",
        md_note,
        flags=re.IGNORECASE,
    )

    caption_norms = {
        re.sub(r"\s+", "", fig.get("caption_zh") or fig.get("caption") or "")
        for fig in figures
    }

    def _drop_duplicate_caption_line(match):
        line = match.group(1)
        return "</figure>" if re.sub(r"\s+", "", line) in caption_norms else match.group(0)

    md_note = re.sub(r"</figure>\n([^\n]+)", _drop_duplicate_caption_line, md_note)
    # 清除任何残留的未识别占位符
    md_note = re.sub(r"\[\[(?:FIG|TABLE):\d+\]\]", "", md_note)

    # 自动删除未被正文使用的图片文件，防止仓库堆积无用图片
    if asset_dir and asset_dir.exists():
        for fig in figures:
            if fig["token"] not in used_tokens:
                unused_file = asset_dir / fig.get("filename", "")
                if unused_file.is_file():
                    try:
                        unused_file.unlink()
                    except OSError:
                        pass
        # 如果该论文没有任何图片被嵌入，删除空的资源子目录
        try:
            if not any(asset_dir.iterdir()):
                asset_dir.rmdir()
        except OSError:
            pass

    return md_note, used_figs


def split_venue_year(venue_arg: str, year_arg: str = ""):
    venue_arg = (venue_arg or "").strip()
    m = re.fullmatch(r"([A-Za-z_\-]+?)(\d{4})", venue_arg)
    if m:
        return m.group(1), year_arg or m.group(2)
    return venue_arg, year_arg or DEFAULT_YEAR


def infer_venue_from_comment(comment: str, default_venue: str, default_year: str = DEFAULT_YEAR):
    if not comment:
        return default_venue, default_year
    c_upper = comment.upper()
    mapping = [
        ("SIGIR-AP", "SIGIR-AP"),
        ("SIGIR", "SIGIR"),
        ("RECSYS", "RecSys"),
        ("WSDM", "WSDM"),
        ("CIKM", "CIKM"),
        ("ECIR", "ECIR"),
        ("KDD", "KDD"),
        ("WWW", "WWW"),
        ("THEWEBCONF", "WWW"),
        ("ACM MM", "ACM_MM"),
        ("MULTIMEDIA", "ACM_MM"),
        ("ICDE", "ICDE"),
        ("ICDM", "ICDM"),
        ("DASFAA", "DASFAA"),
        ("TOIS", "TOIS"),
        ("TKDE", "TKDE"),
        ("TORS", "TORS"),
        ("TMM", "TMM"),
        ("TKDD", "TKDD"),
        ("TWEB", "TWEB"),
        ("AAAI", "AAAI"),
        ("IJCAI", "IJCAI"),
        ("ACL", "ACL_EMNLP"),
        ("EMNLP", "ACL_EMNLP"),
        ("ICLR", "ICLR_NeurIPS_ICML"),
        ("NEURIPS", "ICLR_NeurIPS_ICML"),
        ("ICML", "ICLR_NeurIPS_ICML"),
    ]
    year = default_year
    ym = re.search(r"\b(20\d{2})\b", comment)
    if ym and ym.group(1) in SUPPORTED_YEARS:
        year = ym.group(1)
    for kw, v in mapping:
        if kw in c_upper:
            return v, year
    return default_venue, year


def _fetch_arxiv_entries(api_params: dict, cache_dir: Path = None, download_pdf: bool = True):
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(api_params)
    req = urllib.request.Request(url, headers={"User-Agent": "RecSys-Paper-Notes/3.3"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        xml_data = resp.read()

    root = ET.fromstring(xml_data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        id_el = entry.find("atom:id", ns)
        if title_el is None or summary_el is None or id_el is None:
            continue
        title = re.sub(r"\s+", " ", (title_el.text or "").strip())
        summary = (summary_el.text or "").strip()
        abs_link = (id_el.text or "").strip().replace("http://", "https://")
        arxiv_id = abs_link.rstrip("/").split("/")[-1]
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        published_el = entry.find("atom:published", ns)
        published = (published_el.text or "")[:10] if published_el is not None else ""
        comment_el = entry.find("{http://arxiv.org/schemas/atom}comment")
        comment = comment_el.text.strip() if comment_el is not None and comment_el.text else ""

        full_body = ""
        pdf_file = None
        if download_pdf and cache_dir:
            pdf_file = cache_dir / f"{re.sub(r'[^a-zA-Z0-9._-]', '_', arxiv_id)}.pdf"
            if not pdf_file.exists():
                try:
                    pdf_req = urllib.request.Request(pdf_url, headers={"User-Agent": "RecSys-Paper-Notes/3.3"})
                    with urllib.request.urlopen(pdf_req, timeout=60) as p_resp:
                        pdf_file.write_bytes(p_resp.read())
                except Exception as e:
                    print(f"  [Warn] 下载 arXiv PDF 失败 ({arxiv_id}): {e}")
                    pdf_file = None

            html_cache = cache_dir / f"{re.sub(r'[^a-zA-Z0-9._-]', '_', arxiv_id)}.html.txt"
            if html_cache.exists():
                full_body = html_cache.read_text(encoding="utf-8", errors="ignore")
            else:
                html_text = _fetch_arxiv_html_text(arxiv_id)
                if len(html_text) > 2500:
                    html_cache.write_text(html_text, encoding="utf-8")
                    full_body = html_text

            if not full_body and pdf_file is not None and pdf_file.exists():
                full_body = extract_text_from_pdf(pdf_file)

        combined_raw = f"{comment}\n{summary}\n{full_body}"
        paper_url, code_url, all_code_urls = extract_code_and_paper_urls(combined_raw, default_paper_url=abs_link)

        content_str = (
            f"Title: {title}\n"
            f"Published Date: {published}\n"
            f"Official Paper URL (paper_url): {paper_url}\n"
            f"Detected Open-Source Code URL (code_url): {code_url or 'None detected'}\n"
            f"All Candidate Code URLs in Text: {', '.join(all_code_urls) if all_code_urls else 'None'}\n"
            f"Author Comment / Venue Note: {comment}\n\n"
            f"Abstract:\n{summary}\n\n"
        )
        if full_body:
            content_str += f"Full Paper Text (with LaTeX formulas & experimental tables):\n{full_body}"

        papers.append({
            "title": title,
            "content": content_str,
            "comment": comment,
            "paper_url": paper_url,
            "code_url": code_url,
            "arxiv_id": arxiv_id,
            "pdf_file": str(pdf_file) if pdf_file else "",
            "full_text": bool(full_body),
            "source": "arxiv",
        })
    return papers


def fetch_arxiv_papers(query: str, max_results: int = 5, cache_dir: Path = None, download_pdf: bool = True):
    params = {
        "search_query": f"all:({query}) AND (cat:cs.IR OR cat:cs.AI OR cat:cs.LG)",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": max_results,
    }
    return _fetch_arxiv_entries(params, cache_dir=cache_dir, download_pdf=download_pdf)


def fetch_arxiv_by_ids(arxiv_ids, cache_dir: Path = None, download_pdf: bool = True, batch_size: int = 40):
    ids = [re.sub(r"v\d+$", "", i.strip()) for i in arxiv_ids if i and i.strip()]
    if not ids:
        return []
    all_papers = []
    for start in range(0, len(ids), batch_size):
        chunk = ids[start : start + batch_size]
        params = {"id_list": ",".join(chunk), "max_results": len(chunk)}
        all_papers.extend(_fetch_arxiv_entries(params, cache_dir=cache_dir, download_pdf=download_pdf))
        if start + batch_size < len(ids):
            time.sleep(1.0)
    return all_papers


def load_curated_csv(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): (v or "").strip() for k, v in row.items() if k})
    return rows


def call_llm_api(
    paper_text: str,
    venue: str,
    api_key: str,
    base_url: str,
    model: str,
    forced_cat: str = "",
    figure_context: str = "",
    max_output_tokens: int = None,
    thinking: bool = False,
) -> str:
    """调用兼容 OpenAI 格式的大模型接口（默认 DeepSeek deepseek-flash）生成高密度 Markdown Note。"""
    cat_hint = f"\n优先归入推荐系统分类标识：`{forced_cat}`" if forced_cat else ""
    figure_hint = ""
    if figure_context:
        figure_hint = (
            "\n\n### 可选论文插图（按需挑选 0–2 张内嵌，宁缺毋滥！）\n"
            "以下是从论文 PDF 提取的插图占位符。**仅当某张图（如模型整体架构图或核心动机示意图）非常适合放在对应正文段落（如 `### 1. 整体框架` 或 `## Background`）中辅助理解时，才在该段落正下方单独一行插入该占位符（最多选用 1–2 张）**。\n"
            "**如果某张图没有特别合适的位置，或者只是次要实验曲线，请直接忽略不要插入，不是非放不可！严禁在文末集中堆放图片！**\n"
            f"{figure_context}\n"
        )
    effective_max_tokens = max_output_tokens
    if thinking and effective_max_tokens and effective_max_tokens < 12000:
        effective_max_tokens = 12000

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"目标所属会议/期刊目录标识：`{venue}`{cat_hint}\n\n"
                "请将以下论文提炼为一篇 **1,800–2,600 字的高密度精读笔记 (Paper Note)**——只写增量创新、核心公式、评估协议与完整实验对比表，严禁写成冗长的逐段翻译流水账！第 1 行必须直接以 `---` 开启 YAML Frontmatter："
                f"{figure_hint}\n\n{paper_text}"
            ),
        },
    ]
    raw = call_chat_completion(
        messages,
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0.15,
        timeout=240,
        max_tokens=effective_max_tokens,
        thinking=thinking,
    )
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
    inline = re.search(r"categories:\s*\[([^\]]+)\]", md_content)
    if inline:
        for c in inline.group(1).split(","):
            c = c.strip().strip("'\"")
            if c in CATEGORIES:
                return c
    return fallback


def _set_frontmatter_field(frontmatter: str, name: str, value: str) -> str:
    if re.search(rf"^{name}:", frontmatter, re.MULTILINE):
        return re.sub(rf"^{name}:.*$", f"{name}: {value}", frontmatter, count=1, flags=re.MULTILINE)
    return frontmatter.rstrip() + f"\n{name}: {value}"


def ensure_and_normalize_frontmatter(
    md_note: str,
    title: str,
    venue: str,
    year: str,
    paper_url: str = "",
    code_url: str = "",
    fallback_cat: str = "collaborative_filtering",
) -> str:
    md_note = md_note.strip()
    md_note = re.sub(r"^```(?:yaml|markdown)?\s*\n(---\s*\n.*?\n---)\s*\n```\s*\n", r"\1\n\n", md_note, flags=re.DOTALL)

    m = re.search(r"(?:^|\n)---\s*\n(.*?)\n---\s*\n", md_note[:2500], re.DOTALL)
    if m and m.start() < 300:
        fm = m.group(1)
        body = md_note[m.end():].lstrip()
        body = re.sub(r"^```\s*\n", "", body)
    elif re.match(r"^title:\s*", md_note):
        h1_match = re.search(r"\n#\s+", md_note)
        if h1_match:
            fm = md_note[:h1_match.start()].strip().rstrip("-").strip()
            body = md_note[h1_match.start():].lstrip()
        else:
            fm = ""
            body = md_note
    else:
        fm = ""
        body = md_note

    cat_slug = fallback_cat if fallback_cat in CATEGORIES else "collaborative_filtering"
    cat_title = CATEGORIES[cat_slug]["title"]
    safe_title = title.replace('"', "'")

    if not fm:
        fm = (
            f'title: "[Paper Note] {safe_title}"\n'
            f'description: "[{venue} {year}][{cat_title}] {safe_title}"\n'
            f"tags:\n"
            f"  - {venue} {year}\n"
            f"  - {cat_title}\n"
            f"date: {datetime.now().strftime('%Y-%m-%d')}\n"
            f"venue: {venue}\n"
            f"year: {year}\n"
            f'paper_url: "{paper_url}"\n'
            f'code_url: "{code_url}"\n'
            f"categories:\n"
            f"  - {cat_slug}"
        )
    else:
        fm = _set_frontmatter_field(fm, "venue", venue)
        fm = _set_frontmatter_field(fm, "year", year)
        if paper_url:
            fm = _set_frontmatter_field(fm, "paper_url", f'"{paper_url}"')
        if code_url:
            fm = _set_frontmatter_field(fm, "code_url", f'"{code_url}"')
        if "categories:" not in fm:
            fm = fm.rstrip() + f"\ncategories:\n  - {cat_slug}"

    if not body.startswith("#"):
        body = f"# {title}\n\n" + body

    return f"---\n{fm.strip()}\n---\n\n{body.strip()}\n"


SECTION_PATTERNS = [
    ("TL;DR", r"^##+\s+.*?(?:TL;DR|核心速览|速读|Executive Summary)"),
    ("Experimental Setup", r"^##+\s+.*?(?:Experimental Setup|实验设置|评估协议|Evaluation Protocol)"),
    ("Key Experimental Results", r"^##+\s+.*?(?:Experimental Results|实验结果|关键实验|实验关键数据|Main Results)"),
]


def validate_note(md_note: str, venue: str, year: str) -> tuple:
    if not md_note.startswith("---"):
        return False, "缺少 YAML Frontmatter"
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", md_note, re.DOTALL)
    if not m:
        return False, "Frontmatter 未正确闭合"
    fm = m.group(1)

    for label, pat in SECTION_PATTERNS:
        if not re.search(pat, md_note, re.IGNORECASE | re.MULTILINE):
            return False, f"缺少必需章节: ## {label}"

    fm_venue = re.search(r"^venue:\s*['\"]?([^'\"\n]+)", fm, re.MULTILINE)
    if not fm_venue or fm_venue.group(1).strip() != venue:
        return False, f"Frontmatter venue 与归档会刊不一致 ({venue})"

    fm_year = re.search(r"^year:\s*['\"]?(\d{4})", fm, re.MULTILINE)
    if not fm_year or fm_year.group(1) != year:
        return False, f"Frontmatter year 与归档年份不一致 ({year})"

    cats = re.findall(r"^\s*-\s*([a-z0-9_]+)\s*$", fm, re.MULTILINE)
    if not any(c in CATEGORIES for c in cats):
        return False, "categories 中缺少合法分类 slug"

    if "论文英文原名" in md_note:
        return False, "输出仍包含模板占位符"

    return True, ""


def process_single_paper(item: dict, default_venue: str, default_year: str, docs_dir: Path, args):
    slug = slugify(item["title"])
    venue = item.get("venue") or default_venue
    year = item.get("year") or default_year
    if item.get("comment") and not item.get("venue"):
        venue, year = infer_venue_from_comment(item["comment"], default_venue, default_year)
    venue, year = split_venue_year(venue, year)

    if venue not in VENUES:
        return f"[Error] 未知会刊 `{venue}`，已跳过: {item['title'][:50]}", None
    if not re.fullmatch(r"\d{4}", year):
        return f"[Error] 非法年份 `{year}`，已跳过: {item['title'][:50]}", None

    if not item.get("full_text") and not args.allow_abstract:
        return f"[Skip] 未获取到全文（仅摘要），按规则跳过: {item['title'][:50]}", None

    forced_cat = item.get("forced_cat", "")
    existing = list((docs_dir / "venues" / venue / year).rglob(f"{slug}.md"))
    if not existing:
        existing = list((docs_dir / "venues").rglob(f"{slug}.md"))
    if existing and not args.overwrite:
        return f"[Skip] 已存在跳过: {existing[0].relative_to(docs_dir.parent)}", None

    figures = []
    figure_context = ""
    asset_key = re.sub(r"[^a-zA-Z0-9._-]", "_", item.get("arxiv_id") or slug)
    asset_dir = docs_dir / "assets" / "papers" / asset_key
    if not args.no_figures and item.get("pdf_file"):
        try:
            raw_figures = extract_figures(
                Path(item["pdf_file"]),
                asset_dir,
                dpi=args.figure_dpi,
                max_figures=args.max_figures,
            )
            # 过滤掉 [[TABLE:n]] 表格截图（因为正文中已有原生 Markdown 表格），只保留真正的 Figure 插图并删除表格截图文件
            for fg in raw_figures:
                if str(fg.get("token", "")).startswith("[[TABLE:"):
                    tb_file = asset_dir / fg.get("filename", "")
                    if tb_file.is_file():
                        try:
                            tb_file.unlink()
                        except OSError:
                            pass
                else:
                    figures.append(fg)

            if figures and not args.no_figure_translation:
                figures = translate_figure_captions(figures, args.api_key, args.base_url, args.model)
            elif figures:
                for fig in figures:
                    fig.setdefault("caption_zh", _zh_figure_label(fig.get("label")))
            figure_context = format_figure_context(figures)
            if figures:
                print(f"  [Figures] {item['title'][:45]} -> 候选插图 {len(figures)} 张（按需内嵌）")
        except Exception as e:
            print(f"  [Warn] 提取图表失败 ({item['title'][:45]}): {e}")

    last_error = ""
    md_note = ""
    for attempt in range(1, args.retries + 1):
        try:
            use_thinking = args.thinking if attempt == 1 else False
            md_note = call_llm_api(
                compact_paper_content(item["content"], args.max_input_chars),
                f"{venue} {year}",
                args.api_key,
                args.base_url,
                args.model,
                forced_cat=forced_cat,
                figure_context=figure_context,
                max_output_tokens=args.max_output_tokens,
                thinking=use_thinking,
            )
            md_note = ensure_and_normalize_frontmatter(
                md_note,
                title=item["title"],
                venue=venue,
                year=year,
                paper_url=item.get("paper_url", ""),
                code_url=item.get("code_url", ""),
                fallback_cat=forced_cat or "collaborative_filtering",
            )
            ok, reason = validate_note(md_note, venue, year)
            if ok:
                break
            last_error = reason
            print(f"  [Retry {attempt}/{args.retries}] {item['title'][:45]} -> {reason}")
            time.sleep(1)
        except Exception as e:
            last_error = str(e)
            print(f"  [Retry {attempt}/{args.retries}] {item['title'][:45]} -> API 调用失败: {e}")
            time.sleep(2)
    else:
        return f"[Error] 校验未通过，已放弃 ({last_error}): {item['title'][:45]}", None

    primary_cat = extract_primary_category(md_note, fallback=forced_cat or "collaborative_filtering")
    target_dir = docs_dir / "venues" / venue / year / primary_cat
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / f"{slug}.md"

    used_figures = []
    if figures:
        note_dir = posixpath.dirname(out_file.relative_to(docs_dir).as_posix())
        rel_paths = {}
        for fig in figures:
            asset_rel = f"assets/papers/{asset_key}/{fig['filename']}"
            rel_paths[fig["token"]] = posixpath.relpath(asset_rel, start=note_dir)
        md_note, used_figures = apply_figure_placeholders(md_note, figures, rel_paths, asset_dir=asset_dir)

    out_file.write_text(md_note + "\n", encoding="utf-8")

    record = {
        "title": item["title"],
        "slug": slug,
        "venue": venue,
        "year": year,
        "category": primary_cat,
        "paper_url": item.get("paper_url", ""),
        "code_url": item.get("code_url", ""),
        "arxiv_id": item.get("arxiv_id", ""),
        "source": item.get("source", "manual"),
        "note_path": out_file.relative_to(docs_dir.parent).as_posix(),
        "figures": [
            {
                "token": fig["token"],
                "filename": fig["filename"],
                "label": fig["label"],
                "page": fig["page"],
                "caption_zh": fig.get("caption_zh", ""),
            }
            for fig in used_figures
        ],
        "model": args.model,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    return f"[OK] {item['title'][:55]}... -> venues/{venue}/{year}/{primary_cat}/{slug}.md (内嵌插图 {len(used_figures)} 张)", record


def tasks_from_curated_csv(csv_path: Path, cache_dir: Path, download_pdf: bool, limit: int = 0):
    rows = load_curated_csv(csv_path)
    if limit and limit > 0:
        rows = rows[:limit]
    ids = []
    for row in rows:
        aid = row.get("arxiv_id", "")
        if not aid:
            m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", row.get("paper_url", ""))
            aid = m.group(1) if m else ""
        if aid:
            aid_clean = re.sub(r"v\d+$", "", aid.strip())
            row["arxiv_id"] = aid_clean
            ids.append(aid_clean)

    fetched = fetch_arxiv_by_ids(ids, cache_dir=cache_dir, download_pdf=download_pdf)
    by_id = {re.sub(r"v\d+$", "", p["arxiv_id"]): p for p in fetched}
    tasks = []
    for row in rows:
        aid = row.get("arxiv_id", "")
        paper = by_id.get(aid)
        if not paper:
            print(f"  [Warn] 策展名单中未能从 arXiv 获取全文，已跳过: {row.get('title') or aid}")
            continue
        for key in ("title", "venue", "year", "category", "paper_url", "code_url"):
            if row.get(key):
                paper[key] = row[key]
        if row.get("category"):
            paper["forced_cat"] = row["category"]
        paper["source"] = f"curated:{csv_path.name}"
        tasks.append(paper)
    return tasks


def append_manifest(repo_root: Path, records: list):
    if not records:
        return
    manifest_path = repo_root / "data" / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            try:
                existing.add(json.loads(line).get("note_path"))
            except json.JSONDecodeError:
                continue
    with manifest_path.open("a", encoding="utf-8") as f:
        for record in records:
            if record["note_path"] in existing:
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="批量生成推荐系统论文高密度精读笔记，归档至 venues/<VENUE>/<YEAR>/<category>/")
    parser.add_argument("--venue", default="SIGIR", help="目标会议/期刊 key，如 SIGIR, RecSys, TOIS；兼容 SIGIR2026 旧写法")
    parser.add_argument("--year", default=DEFAULT_YEAR, help="目标年份，如 2026")
    parser.add_argument("--pdf-dir", default="", help="本地 PDF 论文文件夹路径")
    parser.add_argument("--papers-csv", default="", help="策展名单 CSV 路径（生产推荐），列: title,arxiv_id,paper_url,code_url,category,venue,year")
    parser.add_argument("--all-curated", action="store_true", help="处理 data/papers/*.csv 下的全部策展名单")
    parser.add_argument("--arxiv-query", default="", help="arXiv 关键词检索（仅本地试点，生产禁用）")
    parser.add_argument("--seed-all-categories", action="store_true", help="遍历 16 大方向自动抓取（仅本地试点，生产禁用）")
    parser.add_argument("--max-papers", type=int, default=3, help="每个查询/文件夹的最大处理篇数（默认 3）")
    parser.add_argument("--limit", type=int, default=0, help="每个策展 CSV 最多处理前 N 篇（0 表示全部）")
    parser.add_argument("--workers", type=int, default=2, help="并发调用 LLM 的线程数（默认 2）")
    parser.add_argument("--retries", type=int, default=2, help="单篇生成失败后的最大尝试次数（默认 2）")
    parser.add_argument("--no-pdf-download", action="store_true", help="仅使用摘要，不下载 PDF 全文（默认拒绝摘要模式）")
    parser.add_argument("--allow-abstract", action="store_true", help="允许仅凭摘要生成笔记（不推荐，默认关闭）")
    parser.add_argument("--overwrite", action="store_true", help="强制覆盖已存在的同名笔记")
    parser.add_argument("--no-figures", action="store_true", help="不提取、不插入 PDF 图表")
    parser.add_argument("--no-figure-translation", action="store_true", help="不调用 LLM 翻译图注，改用“图 n/表 n”中文编号兜底")
    parser.add_argument("--max-figures", type=int, default=6, help="每篇论文最多提取候选插图数（默认 6）")
    parser.add_argument("--figure-dpi", type=int, default=200, help="图表渲染 DPI（默认 200）")
    parser.add_argument("--max-input-chars", type=int, default=DEFAULT_MAX_INPUT_CHARS, help="送入 LLM 的论文最大字符数（默认 65000；0 表示不限制）")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help="单次生成最大输出 token（默认 8192）")
    parser.add_argument("--thinking", action="store_true", help="启用模型 thinking/reasoning")
    parser.add_argument("--api-key", default="", help="大模型 API Key（或设置 LLM_API_KEY / DEEPSEEK_API_KEY）")
    parser.add_argument("--base-url", default="", help="API Base URL（默认 https://api.deepseek.com/v1）")
    parser.add_argument("--model", default="", help="模型名称，默认 deepseek-flash（可用 LLM_MODEL / DEEPSEEK_MODEL 覆盖）")
    args = parser.parse_args()
    run_start = time.monotonic()

    args.api_key = resolve_api_key(args.api_key)
    args.base_url = resolve_base_url(args.base_url)
    args.model = resolve_model(args.model)

    if not args.api_key:
        print("[Error] 未检测到 API Key！请传入 --api-key 或设置环境变量 LLM_API_KEY / DEEPSEEK_API_KEY")
        return

    repo_root = SCRIPT_DIR.parent
    docs_dir = repo_root / "docs"
    cache_dir = repo_root / ".cache_pdfs"
    default_venue, default_year = split_venue_year(args.venue, args.year)

    tasks = []
    if args.pdf_dir:
        pdf_folder = Path(args.pdf_dir)
        for p in sorted(pdf_folder.glob("*.pdf"))[: args.max_papers]:
            print(f"[PDF] 正在提取本地 PDF 全文与代码链接: {p.name} ...")
            txt = extract_text_from_pdf(p)
            if not txt:
                continue
            paper_url, code_url, all_codes = extract_code_and_paper_urls(txt)
            header = (
                f"Title: {p.stem}\n"
                f"Official Paper URL (paper_url): {paper_url or 'Local PDF'}\n"
                f"Detected Open-Source Code URL (code_url): {code_url or 'None detected'}\n"
                f"All Candidate Code URLs: {', '.join(all_codes) if all_codes else 'None'}\n\n"
            )
            tasks.append({
                "title": p.stem,
                "content": header + txt,
                "comment": "",
                "paper_url": paper_url,
                "code_url": code_url,
                "pdf_file": str(p),
                "full_text": True,
                "source": f"pdf:{p.name}",
            })
    elif args.papers_csv or args.all_curated:
        csv_paths = []
        if args.papers_csv:
            csv_paths.append(Path(args.papers_csv))
        if args.all_curated:
            csv_paths.extend(sorted((repo_root / "data" / "papers").glob("*.csv")))
        seen = set()
        for csv_path in csv_paths:
            if not csv_path.exists():
                print(f"[Warn] 策展名单不存在: {csv_path}")
                continue
            if csv_path.resolve() in seen:
                continue
            seen.add(csv_path.resolve())
            print(f"[Curated] 读取策展名单: {csv_path}")
            tasks.extend(tasks_from_curated_csv(csv_path, cache_dir, download_pdf=not args.no_pdf_download, limit=args.limit))
    elif args.seed_all_categories:
        print(f"[Seed] 警告：自动抓取仅供本地试点，生产收录请使用 --papers-csv。每个方向获取 {args.max_papers} 篇...")
        for cat_slug, q in CATEGORY_SEED_QUERIES.items():
            print(f"  -> 检索方向 [{cat_slug}]: {q}")
            fetched = fetch_arxiv_papers(q, max_results=args.max_papers, cache_dir=cache_dir, download_pdf=not args.no_pdf_download)
            for item in fetched:
                item["forced_cat"] = cat_slug
                tasks.append(item)
            time.sleep(1)
    elif args.arxiv_query:
        print(f"[arXiv] 警告：关键词检索仅供本地试点，生产收录请使用 --papers-csv。正在检索 '{args.arxiv_query}'...")
        tasks = fetch_arxiv_papers(args.arxiv_query, max_results=args.max_papers, cache_dir=cache_dir, download_pdf=not args.no_pdf_download)
    else:
        print(
            "[Info] 请指定以下任一参数开始填充论文：\n"
            "  1. --papers-csv data/papers/SIGIR2026.csv  (生产推荐)\n"
            "  2. --pdf-dir ./本地PDF目录\n"
            "  3. --arxiv-query \"关键词\"  (仅本地试点)\n"
            "  4. --seed-all-categories  (仅本地试点)"
        )
        return

    if not tasks:
        print("[Info] 没有可处理的论文任务。")
        return

    log = RunLog("generate")
    log.write(
        f"START tasks={len(tasks)} workers={args.workers} model={args.model} "
        f"venue={default_venue} year={default_year} thinking={args.thinking} "
        f"max_input_chars={args.max_input_chars} max_output_tokens={args.max_output_tokens} "
        f"max_figures={args.max_figures} translate_figures={not args.no_figure_translation}"
    )

    print(f"\n[Start] 共收集 {len(tasks)} 篇待处理论文，启用 {args.workers} 个并发线程调用 {args.model} ...\n")
    generation_start = time.monotonic()
    reporter = ProgressReporter(
        total=len(tasks),
        label="生成笔记",
        log=log,
        cost_fn=lambda: estimate_usage_cost_usd(args.model),
    )
    reporter.start()
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(process_single_paper, item, default_venue, default_year, docs_dir, args): item
            for item in tasks
        }
        for fut in as_completed(future_map):
            item = future_map[fut]
            try:
                msg, record = fut.result()
                if msg.startswith("[OK]"):
                    status = "ok"
                elif msg.startswith("[Skip]"):
                    status = "skip"
                elif msg.startswith("[Error]"):
                    status = "error"
                else:
                    status = "other"
                reporter.update(status, title=item.get("title", ""), detail=msg)
                if record:
                    records.append(record)
            except Exception as e:
                detail = f"[Error] 处理失败 ({item['title'][:45]}): {e}"
                reporter.update("error", title=item.get("title", ""), detail=detail)

    reporter.finish()
    generation_elapsed = time.monotonic() - generation_start
    append_manifest(repo_root, records)

    print("\n[Index] 正在重新构建分类/年份/会刊导航索引与卡片计数...")
    from build_nav_and_index import (
        collect_all_notes,
        discover_years,
        build_topic_pages,
        build_year_pages,
        build_venue_pages,
        build_home_and_mkdocs_yml,
    )
    all_notes = collect_all_notes(docs_dir)
    all_years = discover_years(docs_dir, all_notes)
    build_topic_pages(docs_dir, all_notes)
    build_year_pages(docs_dir, all_notes, all_years)
    build_venue_pages(docs_dir, all_notes, all_years)
    build_home_and_mkdocs_yml(repo_root, all_notes, all_years)
    print(f"[Done] 本次成功生成 {len(records)} 篇，全库现已收录 {len(all_notes)} 篇推荐系统论文解读！")
    print_usage_summary(
        args.model,
        processed=len(tasks),
        success=len(records),
        elapsed_seconds=generation_elapsed,
        log=log,
    )
    total_elapsed = time.monotonic() - run_start
    print(f"[Time] 本次运行总耗时（含抓取/索引）{format_duration(total_elapsed)}")
    print(f"[Log] 详细日志: {log.path}")
    log.write(f"TOTAL elapsed={format_duration(total_elapsed)} generated={len(records)}")
    log.close()


if __name__ == "__main__":
    main()