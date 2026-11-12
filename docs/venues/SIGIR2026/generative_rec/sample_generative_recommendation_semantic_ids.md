---
title: "[Paper Note] Generative Recommendation with Hierarchical Semantic IDs (示例模板)"
description: "[SIGIR 2026][生成式推荐与离散语义 ID] 融合协同图谱与 RQ-VAE 残差量化的端到端生成式推荐框架 (Full-Ranking 评测)"
tags:
  - SIGIR 2026
  - 生成式推荐与离散语义 ID
  - 序列与会话推荐
  - RQ-VAE
date: 2026-11-12
venue: SIGIR2026
paper_url: "https://arxiv.org/abs/2305.05065"
code_url: "https://github.com/reczoo/FuxiCTR"
categories:
  - generative_rec
  - sequential_rec
---

# Generative Recommendation with Hierarchical Semantic IDs (示例模板)

- **会议/期刊**: SIGIR 2026
- **推荐链路定位**: 端到端生成式召回与排序 (End-to-End Generative Retrieval)
- **研究方向**: 生成式推荐与离散语义 ID (`generative_rec`) / 序列与会话推荐 (`sequential_rec`)
- **论文原文**: [https://arxiv.org/abs/2305.05065](https://arxiv.org/abs/2305.05065)
- **开源代码**: [https://github.com/reczoo/FuxiCTR](https://github.com/reczoo/FuxiCTR)

## TL;DR (核心速览)

本篇为推荐系统知识库标准示例文档。针对生成式推荐中离散语义 ID (RQ-VAE) 在长尾冷启动物品上的哈希碰撞与协同信号丢失问题，本文提出协同感知的层次化残差量化与自回归解码框架，在 Amazon Beauty/Sports 等基准的 **Leave-One-Out + Full-Ranking (全库未交互排序)** 协议下取得 Recall@10 提升 8.3%–9.9%。

## Experimental Setup & Evaluation Protocol (实验设置与评估协议)

- **过滤规则与正样本定义**: 采用标准 `5-core` 过滤，保留所有用户-物品评论交互作为隐式正反馈。
- **数据划分方式 (Splitting Strategy)**: 采用 **Leave-One-Out (留一法)**，最新一次交互作为测试集，倒数第二次作为验证集。
- **候选集打分协议 (Ranking Protocol)**: 采用无偏的 **Full-Ranking (全量候选物品集排序)**，配合 Beam Search (Beam Size = 20) 解码。

| 数据集 | #Users | #Items | #Interactions | 稀疏度 | 评测指标 (Recall@10 / NDCG@10) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Amazon Beauty | 22,363 | 12,101 | 198,502 | 99.93% | **0.0742** / **0.0408** |
| Amazon Sports | 35,598 | 18,357 | 296,337 | 99.95% | **0.0435** / **0.0226** |
