#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_recsys_notes.py (Pro 16大分类 + 自动提取原文/开源代码 URL + DeepSeek-V4.1-Flash 默认适配版)
支持三种填充模式：
  1. 本地 PDF 文件夹批量全文精读：
     python3 scripts/generate_recsys_notes.py --pdf-dir ./my_pdfs --venue SIGIR2026
  2. 按关键词/会议从 arXiv 自动检索 + 自动下载 PDF 全文精读（含公式、数据集划分、评估协议、原文与开源代码链接）：
     python3 scripts/generate_recsys_notes.py --venue SIGIR2026 --arxiv-query "generative recommendation" --max-papers 5
  3. 一键遍历 16 大推荐系统细分方向各抓取 N 篇代表作填充全站 (--seed-all-categories)：
     python3 scripts/generate_recsys_notes.py --seed-all-categories --max-papers 2 --venue RecSys2026

默认模型已配置为 DeepSeek-V4.1-Flash (`deepseek-flash`)，支持断点续传与多线程并发 (--workers 3)。
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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from init_clean_recsys_repo import VENUES, CATEGORIES

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
   - **过滤与正样本定义**：是否采用 `5-core` / `10-core` 过滤？隐式反馈如何界定正样本（全交互 vs. 评分阈值 $\ge 4$ vs. 有效观看时长/购买）？
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
venue: 会议目录名(如SIGIR2026)
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


def extract_text_from_pdf(pdf_path: Path, max_chars: int = 52000) -> str:
    """使用 pypdf 提取 PDF 论文全文文本。"""
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
    """通过 arXiv API 检索推荐系统论文，自动下载 PDF 提取全文并挖掘开源代码链接。"""
    encoded_q = urllib.parse.quote(f"all:({query}) AND (cat:cs.IR OR cat:cs.AI OR cat:cs.LG)")
    url = f"http://export.arxiv.org/api/query?search_query={encoded_q}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    req = urllib.request.Request(url, headers={"User-Agent": "RecSys-Paper-Notes/2.2"})
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
                    pdf_req = urllib.request.Request(pdf_url, headers={"User-Agent": "RecSys-Paper-Notes/2.2"})
                    with urllib.request.urlopen(pdf_req, timeout=45) as p_resp:
                        pdf_file.write_bytes(p_resp.read())
                except Exception as e:
                    print(f"  [Warn] 下载 arXiv PDF 失败 ({arxiv_id})，将回退使用摘要: {e}")
            if pdf_file.exists():
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
        })
    return papers


def call_llm_api(paper_text: str, venue: str, api_key: str, base_url: str, model: str, forced_cat: str = "") -> str:
    """调用兼容 OpenAI 格式的大模型接口（默认 DeepSeek-V4.1-Flash: deepseek-flash）生成结构化 Markdown 笔记。"""
    endpoint = base_url.rstrip("/") + "/chat/completions"
    cat_hint = f"\n优先归入推荐系统分类标识：`{forced_cat}`" if forced_cat else ""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"目标所属会议/期刊目录标识：`{venue}`{cat_hint}\n\n请严格按照推荐系统评估协议规范深入解读以下论文，并确保准确填入 paper_url 与 code_url：\n\n{paper_text}",
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
    parser = argparse.ArgumentParser(description="批量生成推荐系统论文全文精读笔记并自动归档至 22 会刊 x 16 正交分类目录")
    parser.add_argument("--venue", default="SIGIR2026", help="默认目标会议/期刊目录名，如 SIGIR2026, RecSys2026, TOIS2026, TKDE2026")
    parser.add_argument("--pdf-dir", default="", help="本地 PDF 论文文件夹路径")
    parser.add_argument("--arxiv-query", default="", help="arXiv 搜索关键词，如 'generative recommendation' 或 'SIGIR recommendation'")
    parser.add_argument("--seed-all-categories", action="store_true", help="一键遍历 16 大推荐系统细分方向自动从 arXiv 拉取代表作填充全库")
    parser.add_argument("--max-papers", type=int, default=3, help="每个查询或文件夹的最大处理篇数（默认 3）")
    parser.add_argument("--workers", type=int, default=2, help="并发调用 LLM 的线程数（默认 2）")
    parser.add_argument("--no-pdf-download", action="store_true", help="arXiv 模式下仅使用摘要，不下载 PDF 全文")
    parser.add_argument("--overwrite", action="store_true", help="强制覆盖已存在的同名笔记")
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", ""), help="大模型 API Key (或设置环境变量 LLM_API_KEY)")
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1"), help="API Base URL")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "deepseek-flash"), help="模型名称，默认 deepseek-flash (DeepSeek-V4.1-Flash)")
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
            print(f"[PDF] 正在提取本地 PDF 全文与代码链接: {p.name} ...")
            txt = extract_text_from_pdf(p)
            if txt:
                paper_url, code_url, all_codes = extract_code_and_paper_urls(txt)
                header = (
                    f"Title: {p.stem}\n"
                    f"Official Paper URL (paper_url): {paper_url or 'Local PDF'}\n"
                    f"Detected Open-Source Code URL (code_url): {code_url or 'None detected'}\n"
                    f"All Candidate Code URLs: {', '.join(all_codes) if all_codes else 'None'}\n\n"
                )
                tasks.append({"title": p.stem, "content": header + txt, "comment": "", "paper_url": paper_url, "code_url": code_url})
    elif args.seed-all-categories if hasattr(args, 'seed-all-categories') else args.seed_all_categories:
        print(f"[Seed] 正在遍历 16 大推荐系统正交细分方向，每个方向从 arXiv 获取 {args.max_papers} 篇论文...")
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
        print("[Info] 请指定以下任一参数开始填充论文：\n  1. --seed-all-categories (一键为 16 大推荐方向自动填充代表性论文)\n  2. --arxiv-query \"关键词\"\n  3. --pdf-dir ./本地PDF目录")
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
