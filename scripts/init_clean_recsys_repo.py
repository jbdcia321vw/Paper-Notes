#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
init_clean_recsys_repo.py
一键清空原 Paper-Notes 中的泛 AI (CV/NLP/ML) 历史论文目录，
并初始化涵盖 22 个推荐系统相关顶会/顶刊与 14 大推荐系统细分方向的双维度目录架构。
"""

import os
import shutil
from pathlib import Path

# 1. 推荐系统核心会议与顶级期刊配置 (Venues)
VENUES = {
    # 信息检索与推荐核心顶会
    "SIGIR": {"name": "SIGIR", "full_name": "ACM SIGIR Conference on Research and Development in Information Retrieval", "tier": "CCF-A · IR/RecSys 顶会", "type": "conference"},
    "RecSys": {"name": "RecSys", "full_name": "ACM Conference on Recommender Systems", "tier": "旗舰专会 · RecSys 顶会", "type": "conference"},
    "WSDM": {"name": "WSDM", "full_name": "ACM International Conference on Web Search and Data Mining", "tier": "CCF-B · 搜索与推荐顶会", "type": "conference"},
    "CIKM": {"name": "CIKM", "full_name": "ACM International Conference on Information and Knowledge Management", "tier": "CCF-B · IR/KM 顶会", "type": "conference"},
    "ECIR": {"name": "ECIR", "full_name": "European Conference on Information Retrieval", "tier": "IR 重要会议", "type": "conference"},
    "SIGIR-AP": {"name": "SIGIR-AP", "full_name": "ACM SIGIR Conference on Information Retrieval in the Asia Pacific", "tier": "IR 亚太专会", "type": "conference"},
    # 数据挖掘、Web 与多媒体顶会
    "KDD": {"name": "KDD", "full_name": "ACM SIGKDD Conference on Knowledge Discovery and Data Mining", "tier": "CCF-A · 数据挖掘与工业推荐顶会", "type": "conference"},
    "WWW": {"name": "WWW (TheWebConf)", "full_name": "ACM Web Conference", "tier": "CCF-A · Web 与推荐系统顶会", "type": "conference"},
    "ACM_MM": {"name": "ACM MM", "full_name": "ACM International Conference on Multimedia", "tier": "CCF-A · 多模态推荐重镇", "type": "conference"},
    "ICDE": {"name": "ICDE", "full_name": "IEEE International Conference on Data Engineering", "tier": "CCF-A · 数据工程顶会", "type": "conference"},
    "ICDM": {"name": "ICDM", "full_name": "IEEE International Conference on Data Mining", "tier": "CCF-B · 数据挖掘顶会", "type": "conference"},
    "DASFAA": {"name": "DASFAA", "full_name": "International Conference on Database Systems for Advanced Applications", "tier": "CCF-B · 数据库与推荐应用会议", "type": "conference"},
    # 综合 AI / ML / NLP 顶会 (仅收录推荐与检索相关论文)
    "AAAI": {"name": "AAAI (RecSys Track)", "full_name": "AAAI Conference on Artificial Intelligence", "tier": "CCF-A · AI 综合顶会", "type": "conference"},
    "IJCAI": {"name": "IJCAI (RecSys Track)", "full_name": "International Joint Conference on Artificial Intelligence", "tier": "CCF-A · AI 综合顶会", "type": "conference"},
    "ACL_EMNLP": {"name": "ACL / EMNLP (RecSys & CRS)", "full_name": "ACL & EMNLP Conferences", "tier": "CCF-A/B · NLP 与对话推荐顶会", "type": "conference"},
    "ICLR_NeurIPS_ICML": {"name": "ICLR / NeurIPS / ICML (RecSys)", "full_name": "Top ML Conferences (RecSys Papers)", "tier": "CCF-A · 机器学习三大顶会", "type": "conference"},
    # 顶级期刊 (Journals)
    "TOIS": {"name": "ACM TOIS", "full_name": "ACM Transactions on Information Systems", "tier": "CCF-A · 信息系统与推荐顶刊", "type": "journal"},
    "TKDE": {"name": "IEEE TKDE", "full_name": "IEEE Transactions on Knowledge and Data Engineering", "tier": "CCF-A · 知识与数据工程顶刊", "type": "journal"},
    "TORS": {"name": "ACM TORS", "full_name": "ACM Transactions on Recommender Systems", "tier": "ACM 推荐系统旗舰专刊", "type": "journal"},
    "TMM": {"name": "IEEE TMM", "full_name": "IEEE Transactions on Multimedia", "tier": "CCF-B · 多模态推荐顶刊", "type": "journal"},
    "TKDD": {"name": "ACM TKDD", "full_name": "ACM Transactions on Knowledge Discovery from Data", "tier": "CCF-B · 数据挖掘顶刊", "type": "journal"},
    "TWEB": {"name": "ACM TWEB", "full_name": "ACM Transactions on the Web", "tier": "CCF-B · Web 顶刊", "type": "journal"},
}

# 2. 推荐系统 14 大核心与前沿细分类别 (Categories)
CATEGORIES = {
    "collaborative_filtering": {
        "title": "协同过滤与表征学习",
        "en_title": "Collaborative Filtering & Representation Learning",
        "desc": "矩阵分解、双塔召回、对比学习自监督表征、解耦表征与负采样策略",
    },
    "sequential_rec": {
        "title": "序列与会话推荐",
        "en_title": "Sequential & Session-based Recommendation",
        "desc": "长短期用户行为序列建模、Transformer/Mamba/RNN 架构、会话匿名推荐与 Next-Item 预测",
    },
    "generative_rec": {
        "title": "生成式推荐与语义 ID",
        "en_title": "Generative Recommendation & Semantic IDs",
        "desc": "RQ-VAE 离散化语义 Token、TIGER 生成式检索范式、扩散模型推荐 (Diffusion Rec) 与自回归生成",
    },
    "llm_agent_rec": {
        "title": "大模型与智能体推荐",
        "en_title": "LLM4Rec & Agentic Recommendation",
        "desc": "大模型指令微调推荐、In-Context Prompting、知识蒸馏对齐、RecSys Agent 用户行为仿真",
    },
    "multimodal_rec": {
        "title": "多模态推荐与内容理解",
        "en_title": "Multimodal Recommendation",
        "desc": "视觉-文本-音频多模态特征融合、多模态图对比学习、短视频/微视频推荐与商品图文匹配",
    },
    "graph_rec": {
        "title": "图神经网络与知识图谱推荐",
        "en_title": "Graph & Knowledge-aware Recommendation",
        "desc": "LightGCN 谱图卷积改进、超图学习 (Hypergraph Rec)、知识图谱推理推荐 (KGRec) 与异构图建模",
    },
    "social_rec": {
        "title": "社交与群组推荐",
        "en_title": "Social & Group Recommendation",
        "desc": "社交信任网络传播、社交去噪与同质性建模、群组偏好聚合 (Group Rec) 与意见领袖分析",
    },
    "cold_start_cross_domain": {
        "title": "冷启动与跨域/多场景推荐",
        "en_title": "Cold-Start, Cross-Domain & Multi-Scenario Rec",
        "desc": "新用户/新物品冷启动、元学习 (Meta-Learning)、跨域迁移学习 (CDR) 与多场景自适应建模",
    },
    "ctr_multi_task_ranking": {
        "title": "CTR/CVR 预估与多任务精排",
        "en_title": "CTR Prediction, Multi-Task & Cascade Ranking",
        "desc": "高阶特征交叉网络、MMoE/PLE 多目标优化、粗排-精排-重排级联一致性与延迟反馈建模",
    },
    "conversational_explainable": {
        "title": "对话推荐与可解释性",
        "en_title": "Conversational & Explainable Recommendation",
        "desc": "多轮主动追问对话推荐 (CRS)、推荐理由生成、反事实可解释性与用户意图澄清",
    },
    "rl_bandit_rec": {
        "title": "强化学习与动态探索推荐",
        "en_title": "RL, Bandits & Long-term Engagement",
        "desc": "Contextual Bandit 探索与利用 (E&E)、Offline RL 长期留存优化、Slate 列表级策略优化",
    },
    "debias_fairness_trust": {
        "title": "去偏、公平性与可信隐私推荐",
        "en_title": "Debiasing, Fairness, Privacy & Security",
        "desc": "曝光/流行度/位置偏差消除、因果推断推荐 (Causal Rec)、群体公平性、联邦推荐与鲁棒防投毒",
    },
    "poi_spatiotemporal_rec": {
        "title": "时空与 POI 本地生活推荐",
        "en_title": "POI & Spatio-Temporal Recommendation",
        "desc": "Next-POI 兴趣点推荐、时空轨迹建模、LBS 本地生活与移动出行推荐",
    },
    "industrial_system_eval": {
        "title": "工业界系统架构与评测基准",
        "en_title": "Industrial RecSys Architecture & Benchmarks",
        "desc": "工业级召回/排序全链路实战、GPU 端到端加速、线上 A/B 测试经验、开源数据集与评测新范式",
    },
}


def clean_legacy_paper_notes(repo_root: Path):
    """清空原仓库中非推荐系统的 2.3 万篇 CV/NLP/ML 历史目录，保留主题框架。"""
    docs_dir = repo_root / "docs"
    if not docs_dir.exists():
        docs_dir.mkdir(parents=True, exist_ok=True)
        return

    keep_names = {"assets", "stylesheets", "javascripts", "CNAME", "venues", "topics"}
    removed = []
    for item in docs_dir.iterdir():
        if item.name in keep_names:
            continue
        if item.is_dir():
            shutil.rmtree(item)
            removed.append(item.name)
        elif item.is_file() and item.name.endswith(".md"):
            item.unlink()
            removed.append(item.name)

    # 清理原仓库的多余 hreflang 映射文件
    hreflang_file = repo_root / ".hreflang_pairs.json"
    if hreflang_file.exists():
        hreflang_file.write_text("{}\n", encoding="utf-8")

    print(f"[Clean] 已清理原仓库历史目录/文件共 {len(removed)} 项。")


def scaffold_recsys_structure(repo_root: Path, default_year: str = "2026"):
    """创建「按会议/期刊划分 (docs/venues/)」与「按推荐类别划分 (docs/topics/)」的双维度骨架。"""
    docs_dir = repo_root / "docs"
    venues_dir = docs_dir / "venues"
    topics_dir = docs_dir / "topics"
    venues_dir.mkdir(parents=True, exist_ok=True)
    topics_dir.mkdir(parents=True, exist_ok=True)

    # 1. 创建每个会议/期刊目录及其下的细分方向目录
    for venue_code, vinfo in VENUES.items():
        v_folder = venues_dir / f"{venue_code}{default_year}"
        v_folder.mkdir(parents=True, exist_ok=True)
        for cat_slug in CATEGORIES:
            cat_folder = v_folder / cat_slug
            cat_folder.mkdir(parents=True, exist_ok=True)

    # 2. 在示例目录中写入一篇标准的推荐系统论文解读模板示例 (方便立即预览构建效果)
    sample_dir = venues_dir / f"SIGIR{default_year}" / "generative_rec"
    sample_file = sample_dir / "sample_generative_recommendation_semantic_ids.md"
    if not sample_file.exists():
        sample_file.write_text(
            """---
title: "[Paper Note] Generative Recommendation with Hierarchical Semantic IDs (示例模板)"
description: "[SIGIR 2026][生成式推荐与语义 ID] 推荐系统论文 5 分钟结构化深度解读示例模板"
tags:
  - SIGIR 2026
  - 生成式推荐与语义 ID
  - 序列与会话推荐
  - RQ-VAE
date: 2026-11-12
venue: SIGIR2026
categories:
  - generative_rec
  - sequential_rec
---

# Generative Recommendation with Hierarchical Semantic IDs (示例模板)

- **会议/期刊**: SIGIR 2026
- **主分类**: 生成式推荐与语义 ID (`generative_rec`)
- **交叉分类**: 序列与会话推荐 (`sequential_rec`)
- **论文链接**: `https://arxiv.org/abs/xxxx.xxxxx`
- **开源代码**: `https://github.com/example/generative-rec`

## TL;DR (核心速览)

本篇为推荐系统知识库初始化示例文档。当你运行 `python scripts/generate_recsys_notes.py` 时，脚本会自动提取论文 PDF 或 arXiv 全文，将其归类至对应的会议/期刊与 14 大推荐系统细分方向，并生成包含 Mermaid 架构图与实验对比表格的 5 分钟精读笔记。

## Background & Motivation (背景与动机)

- **领域背景**: 传统双塔召回依赖内积近似最近邻检索 (ANN)，特征交互能力受限。
- **现有局限**: 离散语义 ID (如 RQ-VAE) 在长尾冷启动物品上容易发生哈希碰撞 (ID Collision)，且语义空间与协同过滤交互信号存在对齐鸿沟。
- **核心洞察**: 将协同共现图谱与多模态内容特征联合注入残差量化码本，可在自回归生成阶段同时兼顾语义泛化与协同精准度。

## Method (方法架构)

```mermaid
flowchart TD
    A["多模态物品特征 + 协同交互图"] --> B["多模态协同编码器"]
    B --> C["层次化残差量化 (RQ-VAE) 生成语义 ID"]
    C --> D["用户历史行为语义 ID 序列"]
    D --> E["Decoder-only 生成式推荐底座"]
    E --> F["Beam Search 自回归生成 Top-K 候选物品"]
```

## Key Experimental Results (核心实验结果)

| 数据集 | 指标 | 传统序列基线 (SASRec) | 生成式基线 (TIGER) | 本文方法 (Ours) | 相对提升 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Beauty | Recall@10 | 0.0612 | 0.0685 | **0.0742** | +8.32% |
| Beauty | NDCG@10 | 0.0324 | 0.0371 | **0.0408** | +9.97% |
| Sports | Recall@10 | 0.0351 | 0.0398 | **0.0435** | +9.30% |

## Highlights & Limitations (亮点与局限)

- **核心亮点**: 统一了多模态特征量化与序列自回归生成，显著缓解了冷启动物品的语义碰撞。
- **潜在局限**: 自回归 Beam Search 在线推理延迟高于传统双塔向量检索，工业界落地需配合投机解码或 KV Cache 压缩。
""",
            encoding="utf-8",
        )

    print(f"[Scaffold] 已创建 {len(VENUES)} 个会议/期刊目录与 {len(CATEGORIES)} 个推荐系统细分类别。")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    clean_legacy_paper_notes(root)
    scaffold_recsys_structure(root, default_year="2026")
    print("[Done] 请继续运行 `python scripts/build_nav_and_index.py` 生成双维度索引与 mkdocs.yml。")
