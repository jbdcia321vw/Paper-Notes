#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_recsys_notes.py
支持四种填充模式（均按 venues/<VENUE>/<YEAR>/<category>/ 归档）：
  1. 本地 PDF 文件夹批量全文精读：
     python scripts/generate_recsys_notes.py --pdf-dir ./my_pdfs --venue SIGIR --year 2026
  2. 按关键词从 arXiv 自动检索 + 下载 PDF 全文（仅建议本地试点）：
     python scripts/generate_recsys_notes.py --venue SIGIR --year 2026 --arxiv-query "generative recommendation" --max-papers 5
  3. 按策展 CSV 名单逐篇抓取 arXiv 全文（生产推荐）：
     python scripts/generate_recsys_notes.py --papers-csv data/papers/SIGIR2026.csv
  4. 一键遍历 16 大方向自动填充（仅本地试点，生产禁用）：
     python scripts/generate_recsys_notes.py --seed-all-categories --venue RecSys --year 2026

默认模型为 DeepSeek `deepseek-chat`（可用 LLM_MODEL 覆盖），API Key 支持
`LLM_API_KEY` 或 `DEEPSEEK_API_KEY`，支持断点续传与并发。
"""

import csv
import html
import os
import posixpath
import re
import sys
import json
import time
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from init_clean_recsys_repo import VENUES, CATEGORIES, SUPPORTED_YEARS, DEFAULT_YEAR
from extract_pdf_figures import extract_figures

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
DEFAULT_MODEL = "deepseek-chat"


def resolve_api_key(explicit: str = "") -> str:
    """解析 API Key：显式参数 > LLM_API_KEY > DEEPSEEK_API_KEY。"""
    return (
        explicit
        or os.environ.get("LLM_API_KEY", "")
        or os.environ.get("DEEPSEEK_API_KEY", "")
    ).strip()


def resolve_base_url(explicit: str = "") -> str:
    """解析 Base URL：显式参数 > LLM_BASE_URL > DEEPSEEK_BASE_URL > DeepSeek 默认值。"""
    return (
        explicit
        or os.environ.get("LLM_BASE_URL", "")
        or os.environ.get("DEEPSEEK_BASE_URL", "")
        or DEFAULT_BASE_URL
    ).strip()


def resolve_model(explicit: str = "") -> str:
    """解析模型名：显式参数 > LLM_MODEL > DEEPSEEK_MODEL > deepseek-chat。"""
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
    timeout: int = 180,
    response_format: dict = None,
) -> str:
    """调用兼容 OpenAI Chat Completions 的接口，返回 assistant 文本。"""
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {"model": model, "messages": messages, "temperature": temperature}
    if response_format:
        payload["response_format"] = response_format
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
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:1200]
        raise RuntimeError(f"LLM HTTP {e.code}: {body}") from e
    try:
        return res_json["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"LLM 响应格式异常: {str(res_json)[:800]}") from e


SYSTEM_PROMPT = f"""你是一位在推荐系统（Recommender Systems）与信息检索（Information Retrieval）领域深耕多年的资深研究员与顶会审稿人（SIGIR / RecSys / KDD / WWW / TOIS / TKDE）。
请对输入的推荐系统学术论文全文进行严谨、深度的技术拆解，输出符合 Paper-Notes 规范的中文 Markdown 论文精读笔记（5分钟读懂核心思想与实验细节）。

### 一、16 大推荐系统正交分类归档要求
请先从以下 16 个推荐系统细分方向中，为该论文选择 **1 个最核心的主分类 (primary_category)** 和 **1-2 个交叉分类 (secondary_categories)**：
{CATEGORY_LIST_STR}

### 二、推荐系统领域审稿级拆解准则（核心硬性要求）
1. **必须提取并展示论文原文 URL 与开源代码 URL**：
   - 在 YAML Frontmatter 中必须准确填写 `paper_url: "论文原文链接"` 和 `code_url: "开源代码链接"`（若原文和正则预提取均未发现开源代码链接，则 `code_url: ""` 留空，并在正文元数据栏写“暂未开源”）。
   - 在正文开头的元数据列表中，将论文原文与开源代码渲染为可点击的 Markdown 超链接。
2. **严抠评估协议与数据划分 (Evaluation Protocol & Splitting)**：
   - **过滤与正样本定义**：是否采用 `5-core` / `10-core` 过滤？隐式反馈如何界定正样本（全交互 vs. 评分阈值 $\\ge 4$ vs. 有效观看时长/购买）？
   - **数据集划分策略**：是 **留一法 (Leave-One-Out, LOO)**、**按比例划分 (Ratio Split, 如 8:1:1)**、**全局时间戳切分 (Global Temporal Split，防数据穿越)**，还是 **冷启动/跨域划分 (Zero-shot / Cold-start / Overlapping-user Split)**？是否存在未来信息泄露（Data Leakage）风险？
   - **候选集打分协议**：是无偏的 **全库排序 (Full-Ranking / All-Ranking，对全量未交互物品打分)**，还是有偏的 **负采样排序 (Sampled Ranking，如 1 正配 99/999 负样本)**？
3. **还原张量维度、负采样与损失函数 (Algebraic & Training Rigor)**：
   - 写明关键特征维度（Embedding 维度 $d$、多模态编码器如 CLIP/LLaMA/ResNet 及其原始维度、序列最大截断长度 $L$、RQ-VAE 码本层数与大小）。
   - 写明训练阶段的**负采样策略**（Uniform 随机采样、In-batch 负采样、Popularity 采样、Dynamic Hard Negative 难负例挖掘，或免负采样）及完整损失函数公式（BPR、InfoNCE 温度系数 $\\tau$、BCE、扩散/流匹配损失等）。
4. **计算开销与统计严谨性 (Efficiency & Statistical Rigor)**：
   - 区分**离线预处理/构图/量化耗时**与**在线训练单 Epoch 耗时 / 在线单次推理延迟 (ms/request)** 及峰值显存 (Peak GPU Memory)。
   - 标注实验是否包含多随机种子方差、配对显著性检验（Paired $t$-test / Wilcoxon / Bootstrap $p < 0.05$）以及控制变量的消融基线（如参数量对齐、图度数匹配等）。
5. **100% 忠实于原文数据**：所有实验表格数值、数据集统计量必须严格提取自论文原文，严禁凭空捏造数据；若原文未提及某项指标或未做线上 A/B 测试，请明确注明“原文未报告”。

### 三、输出 Markdown 模板规范
你的输出必须直接以 YAML Frontmatter 开头（严禁在最外层包裹 ```markdown 代码块），严格遵循以下章节结构：
---
title: "[Paper Note] 论文英文原名"
description: "[会议或期刊名][主分类中文名] 一句话概括核心方法、评估协议与关键收益（80字以内）"
tags:
  - 会议或期刊名
  - 主分类中文名
  - 推荐链路阶段(如召回/精排/重排/生成式端到端)
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
- **推荐链路定位**: 召回 (Retrieval) / 粗排 (Pre-ranking) / 精排 (CTR/Multi-Task Ranking) / 重排 (Re-ranking) / 端到端生成式 (End-to-End Generative)
- **研究方向**: 主分类中文名 / 次分类中文名
- **论文原文**: [论文原文链接](论文原文完整URL)
- **开源代码**: [开源代码仓库链接](开源代码完整URL) （若无开源代码则写“暂未开源”）

## TL;DR (核心速览)

用 150 字以内精炼概括：本文针对推荐系统什么具体场景与核心痛点？提出了什么核心算子或训练范式？在什么评估协议（如 Full-Ranking）和数据集上取得了多少相对提升？

## Background & Motivation (背景与动机)

- **领域背景与主流范式**: 当前该推荐子方向的主流建模范式是什么？
- **现有方法局限 (Limitations of Prior Work)**: 现有代表性基线在表征对齐、交互稀疏性、长序列复杂度或工业落地上存在哪 2-3 个具体矛盾？
- **核心挑战 (Key Challenge)**: 为什么简单的特征拼接、增加网络深度或常规对比学习无法解决该问题？
- **核心洞察与切入点 (Key Insight)**: 作者通过什么理论分析或先验观察（Pilot Study）引出了本文的设计？

## Problem Formulation & Method (问题定义与方法架构)

### 1. 符号定义与输入表征 (Notations & Input Representations)

明确列出用户集 $\\mathcal{{U}}$、物品集 $\\mathcal{{I}}$、交互矩阵/序列、多模态/辅助特征维度，以及模型的输入与输出形式。

### 2. 整体流水线 (Overall Architecture)

详细说明数据流向：特征提取与投影 $\\to$ 核心交互/图传播/序列生成模块 $\\to$ 预测打分或解码输出。并附上一段合法的 Mermaid 架构流程图：

```mermaid
flowchart TD
    A["输入层: 用户历史序列 / 物品多模态与ID特征"] --> B["核心表征与对齐模块"]
    B --> C["打分预测 / 生成式解码层"]
```

### 3. 核心模块设计与数学原理 (Key Designs & Formulations)

分 2-3 点深入拆解本文的核心创新模块（必须给出关键矩阵/张量运算公式、维度变化以及为什么这样设计的物理直觉）。

### 4. 训练目标、负采样与推理复杂度 (Optimization, Negative Sampling & Complexity)

- **损失函数**: 完整写出总损失函数 $\\mathcal{{L}}$ 及其各分项权重、温度系数 $\\tau$。
- **训练负采样策略**: 明确说明采用何种负采样（如 In-batch、Uniform、Hard Negative 或免负采样自回归）。
- **时间与空间复杂度**: 对比本文方法与主流基线在离线预处理、训练阶段和在线推理阶段（如是否支持 ANN 向量检索或需 Beam Search）的理论复杂度与实际耗时/显存开销。

## Experimental Setup & Evaluation Protocol (实验设置与评估协议)

- **数据集统计与预处理**: 提取原文使用的基准数据集（如 Amazon、Yelp、Gowalla、MovieLens、KuaiRec、工业数据集等）及过滤规则（如 `5-core`）：

| 数据集 (Dataset) | 用户数 (#Users) | 物品数 (#Items) | 交互数 (#Interactions) | 稀疏度/密度 (Sparsity) | 模态/辅助特征说明 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| (按原文提取) | - | - | - | - | - |

- **数据划分方式 (Splitting Strategy)**: 明确标注是 **Leave-One-Out (留一法)**、**Ratio Split (如 8:1:1)**、**Global Temporal Split (全局时间戳切分)** 还是 **Cold-Start Split**，并说明是否存在数据泄露风险。
- **排名评测协议 (Ranking Protocol)**: 明确标注是 **Full-Ranking (全库未交互物品排序)** 还是 **Sampled Ranking (采样负例排序，注明正负例比例)**。
- **评价指标与显著性检验**: 列出具体指标（如 `Recall@10/20`、`NDCG@10/20`、`AUC`、`GAUC`、长尾覆盖率等）、Embedding 维度 $d$ 对齐情况，以及是否进行了配对统计显著性检验（如 $p < 0.05$）。

## Key Experimental Results (核心实验结果)

### 1. 主实验对比 (Main Results)

提取原文主实验表格中的关键数据集结果（包含各类最强 Baseline、本文方法数值及相对次优基线的提升比例 `Improv.`）：

| 数据集 | 指标 | 传统/协同过滤基线 | 序列/图/多模态最强基线 | 本文方法 (Ours) | 相对提升 (Improv.) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| (按原文提取) | Recall@10 / NDCG@10 / AUC | - | - | **-** | +x.xx% |

### 2. 消融实验与控制变量分析 (Ablation Study)

提取原文消融实验表格或结论，明确移除各核心子模块后的性能下降幅度，并指出哪一个模块贡献最大、消融对照是否严格控制了参数量或图拓扑度数。

### 3. 稀疏度/冷启动分桶与超参敏感度 (Sparsity Buckets & Hyperparameters)

- **用户/物品交互频次分桶表现**: 在长尾物品（Tail Items）或少交互冷启动用户群上的提升是否显著大于热门头部物品？
- **关键超参数敏感度**: 温度系数 $\\tau$、损失权重、网络层数 $L$ 或码本大小变化对性能的影响趋势。
- **线上 A/B 测试结果 (Online A/B Test，如有)**: 若包含工业界线上实验，列出真实业务场景、实验周期、流量占比及 CTR/CVR/GMV/留存率等线上指标收益（若无则注明“原文仅包含离线实验”）。

## Highlights, Limitations & Critiques (亮点、局限与审稿级批判)

- **核心亮点与可迁移性**: 哪些算子、特征对齐方式或损失函数设计具有普适性，可直接迁移到其他推荐子任务？
- **局限性与审稿视角批判 (Critical Limitations)**: 从顶会审稿人视角客观指出本文的潜在短板（如评估协议局限、在线推理延迟、超参敏感度或理论与实现落差等）。

## Rating (四维评分)

- **方法创新性 (Novelty)**: ⭐⭐⭐⭐
- **评估协议与实验扎实度 (Protocol & Experiments)**: ⭐⭐⭐⭐
- **写作与数学规范度 (Clarity & Rigor)**: ⭐⭐⭐⭐
- **工业落地与复现价值 (Industrial & Practical Value)**: ⭐⭐⭐⭐

### 四、图表与图注语言规范
- 正文引用图表时统一使用中文“图 n / 表 n”，不要写 Figure/Table。
- 所有图注（figcaption）必须使用中文；保留数学符号、数据集名、模型名与专有名词。
"""


def extract_code_and_paper_urls(text: str, default_paper_url: str = "") -> tuple:
    """使用正则表达式从论文摘要、作者注释及 PDF 全文中自动挖掘开源代码链接与 DOI/arXiv 原文链接。"""
    # 匹配 GitHub / GitLab / HuggingFace / Gitee 开源代码仓库链接
    code_matches = re.findall(
        r"https?://(?:github\.com|gitlab\.com|huggingface\.co|gitee\.com)/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+",
        text,
    )
    # 过滤掉常见的无关公共库链接
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
    """把 PDF 中字面量 \\uXXXX 解码为真实 Unicode，并移除孤立代理字符。"""
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


def extract_text_from_pdf(pdf_path: Path, max_chars: int = 52000) -> str:
    """使用 pypdf 提取 PDF 论文全文文本。"""
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


def _zh_figure_label(label: str) -> str:
    """把 Figure 1 / Table 2 转成中文“图 1 / 表 2”，作为翻译失败时的兜底。"""
    label = label or ""
    m = re.search(r"\b(?:Figure|Fig\.?)\s*(\d+)", label, re.IGNORECASE)
    if m:
        return f"图 {m.group(1)}"
    m = re.search(r"\bTable\s*(\d+)", label, re.IGNORECASE)
    if m:
        return f"表 {m.group(1)}"
    return "论文图表"


def translate_figure_captions(figures: list, api_key: str, base_url: str, model: str) -> list:
    """将英文图注批量翻译为中文，写入 fig['caption_zh']。"""
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
    """把图表清单压缩成适合放进 LLM 提示词的文本。"""
    lines = []
    for fig in figures:
        caption = re.sub(r"\s+", " ", fig.get("caption_zh") or fig.get("caption", "")).strip()
        if len(caption) > 320:
            caption = caption[:317] + "..."
        if not caption:
            caption = fig.get("label", "Paper figure")
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


def apply_figure_placeholders(md_note: str, figures: list, rel_paths: dict) -> str:
    """替换 LLM 插入的 [[FIG:n]] / [[TABLE:n]]，未使用的图表追加到文末。"""
    if not figures:
        return md_note

    used = set()
    for fig in figures:
        token = fig["token"]
        if token not in md_note:
            continue
        rel_path = rel_paths.get(token)
        if not rel_path:
            continue
        used.add(token)
        md_note = md_note.replace(token, figure_html(fig, rel_path))

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
    md_note = re.sub(r"\[\[(?:FIG|TABLE):\d+\]\]", "", md_note)

    unused = [fig for fig in figures if fig["token"] not in used]
    if unused:
        lines = ["", "## 论文图表", ""]
        for fig in unused:
            rel_path = rel_paths.get(fig["token"])
            if not rel_path:
                continue
            lines.append(figure_html(fig, rel_path))
            lines.append("")
        md_note = md_note.rstrip() + "\n" + "\n".join(lines)

    return md_note


def split_venue_year(venue_arg: str, year_arg: str = ""):
    """兼容 --venue SIGIR2026 旧写法，返回 (venue, year)。"""
    venue_arg = (venue_arg or "").strip()
    m = re.fullmatch(r"([A-Za-z_\-]+?)(\d{4})", venue_arg)
    if m:
        return m.group(1), year_arg or m.group(2)
    return venue_arg, year_arg or DEFAULT_YEAR


def infer_venue_from_comment(comment: str, default_venue: str, default_year: str = DEFAULT_YEAR):
    """从 arXiv comment 中推断会刊与年份，返回 (venue, year)。"""
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
    """调用 arXiv API，下载 PDF 全文，返回统一结构的论文任务列表。"""
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(api_params)
    req = urllib.request.Request(url, headers={"User-Agent": "RecSys-Paper-Notes/2.3"})
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
                    pdf_req = urllib.request.Request(pdf_url, headers={"User-Agent": "RecSys-Paper-Notes/2.3"})
                    with urllib.request.urlopen(pdf_req, timeout=60) as p_resp:
                        pdf_file.write_bytes(p_resp.read())
                except Exception as e:
                    print(f"  [Warn] 下载 arXiv PDF 失败 ({arxiv_id}): {e}")
                    pdf_file = None
            if pdf_file is not None and pdf_file.exists():
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
            content_str += f"Full Paper Text Extracted from PDF:\n{full_body}"

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
    """按关键词检索 arXiv 论文。"""
    params = {
        "search_query": f"all:({query}) AND (cat:cs.IR OR cat:cs.AI OR cat:cs.LG)",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": max_results,
    }
    return _fetch_arxiv_entries(params, cache_dir=cache_dir, download_pdf=download_pdf)


def fetch_arxiv_by_ids(arxiv_ids, cache_dir: Path = None, download_pdf: bool = True):
    """按 arXiv ID 列表精确抓取论文全文（用于策展名单）。"""
    ids = [i.strip() for i in arxiv_ids if i and i.strip()]
    if not ids:
        return []
    params = {"id_list": ",".join(ids), "max_results": len(ids)}
    return _fetch_arxiv_entries(params, cache_dir=cache_dir, download_pdf=download_pdf)


def load_curated_csv(csv_path: Path):
    """读取策展名单 CSV：title, arxiv_id, paper_url, code_url, category, venue, year。"""
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
) -> str:
    """调用兼容 OpenAI 格式的大模型接口（默认 DeepSeek deepseek-chat）生成结构化 Markdown 笔记。"""
    cat_hint = f"\n优先归入推荐系统分类标识：`{forced_cat}`" if forced_cat else ""
    figure_hint = ""
    if figure_context:
        figure_hint = (
            "\n\n### 可用论文图表（请插入占位符）\n"
            "请在正文最相关的位置插入下列占位符（例如方法、实验设置或结果段落），"
            "每张图最多使用一次。占位符必须原样保留，不要改写、不要编造图片路径，"
            "也不要把占位符放进代码块。图注已提供中文版本；正文引用图表时统一写"
            "“图 n / 表 n”，不要写 Figure/Table：\n"
            f"{figure_context}\n"
        )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"目标所属会议/期刊目录标识：`{venue}`{cat_hint}\n\n"
                "请严格按照推荐系统评估协议规范深入解读以下论文，"
                "并确保准确填入 paper_url 与 code_url："
                f"{figure_hint}\n\n{paper_text}"
            ),
        },
    ]
    raw = call_chat_completion(
        messages,
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0.2,
        timeout=180,
    )
    raw = re.sub(r"^```(?:markdown|yaml)?\s*\n", "", raw)
    raw = re.sub(r"\n```\s*$", "", raw)
    return raw.strip()


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title.lower()).strip("_")
    return slug[:75] or f"paper_{int(time.time())}"


REQUIRED_SECTIONS = [
    "## TL;DR",
    "## Experimental Setup",
    "## Key Experimental Results",
]


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


def normalize_frontmatter(md_note: str, venue: str, year: str, paper_url: str = "", code_url: str = "") -> str:
    """强制校正 LLM 输出中的 venue/year/paper_url/code_url，避免归档路径与元数据不一致。"""
    m = re.match(r"^(---\s*\n)(.*?)(\n---\s*\n)", md_note, re.DOTALL)
    if not m:
        return md_note
    fm = m.group(2)
    fm = _set_frontmatter_field(fm, "venue", venue)
    fm = _set_frontmatter_field(fm, "year", year)
    if paper_url:
        fm = _set_frontmatter_field(fm, "paper_url", f'"{paper_url}"')
    if code_url:
        fm = _set_frontmatter_field(fm, "code_url", f'"{code_url}"')
    return m.group(1) + fm + m.group(3) + md_note[m.end():]


def validate_note(md_note: str, venue: str, year: str) -> tuple:
    """生成后自动校验：Frontmatter、必需章节、分类、占位符与年份一致性。"""
    if not md_note.startswith("---"):
        return False, "缺少 YAML Frontmatter"
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", md_note, re.DOTALL)
    if not m:
        return False, "Frontmatter 未正确闭合"
    fm = m.group(1)

    for section in REQUIRED_SECTIONS:
        if section not in md_note:
            return False, f"缺少必需章节: {section}"

    fm_venue = re.search(r"^venue:\s*['\"]?([^'\"\n]+)", fm, re.MULTILINE)
    if not fm_venue or fm_venue.group(1).strip() != venue:
        return False, f"Frontmatter venue 与归档会刊不一致 ({venue})"

    fm_year = re.search(r"^year:\s*['\"]?(\d{4})", fm, re.MULTILINE)
    if not fm_year or fm_year.group(1) != year:
        return False, f"Frontmatter year 与归档年份不一致 ({year})"

    cats = re.findall(r"^\s*-\s*([a-z0-9_]+)\s*$", fm, re.MULTILINE)
    if not any(c in CATEGORIES for c in cats):
        return False, "categories 中缺少合法分类 slug"

    if "(按原文提取)" in md_note or "论文英文原名" in md_note:
        return False, "输出仍包含模板占位符"

    return True, ""


def process_single_paper(item: dict, default_venue: str, default_year: str, docs_dir: Path, args):
    """生成单篇笔记，返回 (状态消息, manifest 记录或 None)。"""
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
    if not args.no_figures and item.get("pdf_file"):
        try:
            asset_dir = docs_dir / "assets" / "papers" / asset_key
            figures = extract_figures(
                Path(item["pdf_file"]),
                asset_dir,
                dpi=args.figure_dpi,
                max_figures=args.max_figures,
            )
            if figures and not args.no_figure_translation:
                figures = translate_figure_captions(figures, args.api_key, args.base_url, args.model)
            figure_context = format_figure_context(figures)
            if figures:
                print(f"  [Figures] {item['title'][:45]} -> {len(figures)} 张图表")
        except Exception as e:
            print(f"  [Warn] 提取图表失败 ({item['title'][:45]}): {e}")

    last_error = ""
    md_note = ""
    for attempt in range(1, args.retries + 1):
        try:
            md_note = call_llm_api(
                item["content"],
                f"{venue} {year}",
                args.api_key,
                args.base_url,
                args.model,
                forced_cat=forced_cat,
                figure_context=figure_context,
            )
            md_note = normalize_frontmatter(md_note, venue, year, item.get("paper_url", ""), item.get("code_url", ""))
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

    if figures:
        note_dir = posixpath.dirname(out_file.relative_to(docs_dir).as_posix())
        rel_paths = {}
        for fig in figures:
            asset_rel = f"assets/papers/{asset_key}/{fig['filename']}"
            rel_paths[fig["token"]] = posixpath.relpath(asset_rel, start=note_dir)
        md_note = apply_figure_placeholders(md_note, figures, rel_paths)

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
            for fig in figures
        ],
        "model": args.model,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    return f"[OK] {item['title'][:55]}... -> venues/{venue}/{year}/{primary_cat}/{slug}.md", record


def tasks_from_curated_csv(csv_path: Path, cache_dir: Path, download_pdf: bool, limit: int = 0):
    """读取策展名单，按 arXiv ID 抓取全文，并保留名单中的会刊/年份/分类信息。"""
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
            row["arxiv_id"] = aid
            ids.append(aid)

    fetched = fetch_arxiv_by_ids(ids, cache_dir=cache_dir, download_pdf=download_pdf)
    by_id = {p["arxiv_id"]: p for p in fetched}
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
    parser = argparse.ArgumentParser(description="批量生成推荐系统论文全文精读笔记，归档至 venues/<VENUE>/<YEAR>/<category>/")
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
    parser.add_argument("--no-figure-translation", action="store_true", help="不将图注翻译为中文")
    parser.add_argument("--max-figures", type=int, default=8, help="每篇论文最多提取图表数（默认 8）")
    parser.add_argument("--figure-dpi", type=int, default=200, help="图表渲染 DPI（默认 200）")
    parser.add_argument("--api-key", default="", help="大模型 API Key（或设置 LLM_API_KEY / DEEPSEEK_API_KEY）")
    parser.add_argument("--base-url", default="", help="API Base URL（默认 https://api.deepseek.com/v1）")
    parser.add_argument("--model", default="", help="模型名称，默认 deepseek-chat（可用 LLM_MODEL / DEEPSEEK_MODEL 覆盖）")
    args = parser.parse_args()

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

    print(f"\n[Start] 共收集 {len(tasks)} 篇待处理论文，启用 {args.workers} 个并发线程调用 {args.model} ...\n")
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(process_single_paper, item, default_venue, default_year, docs_dir, args): item
            for item in tasks
        }
        for idx, fut in enumerate(as_completed(future_map), 1):
            item = future_map[fut]
            try:
                msg, record = fut.result()
                print(f"[{idx}/{len(tasks)}] {msg}")
                if record:
                    records.append(record)
            except Exception as e:
                print(f"[{idx}/{len(tasks)}] [Error] 处理失败 ({item['title'][:45]}): {e}")

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


if __name__ == "__main__":
    main()
