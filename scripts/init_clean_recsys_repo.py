#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
init_clean_recsys_repo.py
一键清空原 Paper-Notes 中的泛 AI (CV/NLP/ML) 历史论文目录，
并初始化涵盖 22 个推荐系统相关顶会/顶刊与优化后的 16 大推荐系统正交细分方向的双维度目录架构。
"""

import shutil
from pathlib import Path

# 1. 推荐系统核心会议与顶级期刊配置 (22 Venues)
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

# 2. 优化后的推荐系统 16 大正交核心细分类别 (4 大板块 × 4 个子方向，边界清晰无重叠)
CATEGORIES = {
    # 板块一：基础协同、图结构与自监督去噪
    "collaborative_filtering": {
        "title": "协同过滤与双塔召回",
        "en_title": "Collaborative Filtering & Two-Tower Retrieval",
        "desc": "矩阵分解、双塔向量内积检索、ANN 索引量化、解耦表征与负采样理论",
    },
    "graph_rec": {
        "title": "图神经网络与知识图谱推荐",
        "en_title": "Graph & Knowledge-Aware Recommendation",
        "desc": "LightGCN 谱图卷积改进、超图推荐 (Hypergraph)、异构图与知识图谱推理 (KGRec)",
    },
    "self_supervised_denoising": {
        "title": "自监督对比学习与交互去噪",
        "en_title": "Self-Supervised, Contrastive & Denoising Rec",
        "desc": "图/序列对比数据增广 (SGL/SimGCL)、隐式反馈假阳性噪声过滤 (Denoising) 与鲁棒表征",
    },
    # 板块二：时序动态、前沿生成式与大模型智能体
    "sequential_rec": {
        "title": "序列与会话推荐",
        "en_title": "Sequential & Session-Based Recommendation",
        "desc": "长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测",
    },
    "generative_rec": {
        "title": "生成式推荐与离散语义 ID",
        "en_title": "Generative Recommendation & Semantic IDs",
        "desc": "RQ-VAE/VQ 残差量化码本、TIGER 自回归生成式检索、扩散/流匹配生成推荐 (Diffusion/Flow Rec)",
    },
    "llm_rec": {
        "title": "大语言模型推荐 (LLM4Rec)",
        "en_title": "LLM-Enhanced & LLM-as-Recommender",
        "desc": "LLM 指令微调推荐、In-Context 列表重排、语义表征增强、大模型向轻量推荐模型知识蒸馏",
    },
    "agent_conversational_rec": {
        "title": "对话推荐与智能体仿真",
        "en_title": "Conversational & Agentic Recommendation",
        "desc": "多轮主动追问对话推荐 (CRS)、基于 LLM Agent 的用户行为模拟沙盒、可解释性推荐理由生成",
    },
    # 板块三：多模态内容、社交时空与迁移泛化
    "multimodal_rec": {
        "title": "多模态推荐与内容理解",
        "en_title": "Multimodal Recommendation",
        "desc": "视觉-文本-音频特征融合与解耦、多模态图对齐、短视频/微视频推荐与图文电商匹配",
    },
    "social_group_rec": {
        "title": "社交与群组推荐",
        "en_title": "Social & Group Recommendation",
        "desc": "社交信任网络传播、社交同质性去噪、群组偏好共识聚合 (Group Rec) 与影响力建模",
    },
    "cold_start_rec": {
        "title": "冷启动与长尾推荐",
        "en_title": "Cold-Start & Long-Tail Recommendation",
        "desc": "Strict Zero-shot 新用户/新物品冷启动、元学习 (Meta-Learning)、少样本适应与长尾物品扶持",
    },
    "cross_domain_multi_scenario": {
        "title": "跨域与多场景推荐",
        "en_title": "Cross-Domain & Multi-Scenario Recommendation",
        "desc": "跨域知识迁移 (CDR)、重叠/非重叠用户对齐、多业务场景联合建模 (STAR/PEPNet 等)",
    },
    "poi_spatiotemporal_rec": {
        "title": "时空轨迹与 POI 本地生活推荐",
        "en_title": "POI & Spatio-Temporal Recommendation",
        "desc": "Next-POI 兴趣点推荐、时空序列轨迹建模、LBS 本地生活服务与移动出行推荐",
    },
    # 板块四：精排重排、长期策略、去偏可信与工业架构
    "ctr_feature_interaction": {
        "title": "CTR/CVR 预估与特征交叉",
        "en_title": "CTR Prediction & Feature Interaction",
        "desc": "显式/隐式高阶特征交叉网络 (DCN/EulerNet)、转化率延迟反馈 (Delayed Feedback) 与概率校准",
    },
    "multi_task_reranking": {
        "title": "多任务学习与列表级重排",
        "en_title": "Multi-Task Learning & Listwise Re-ranking",
        "desc": "MMoE/PLE 多任务多目标精排、帕累托最优、上下文列表重排 (Re-ranking) 与粗精排级联一致性",
    },
    "rl_bandit_rec": {
        "title": "强化学习与动态探索推荐",
        "en_title": "RL, Bandits & Long-Term Engagement",
        "desc": "Contextual Bandit 探索与利用 (E&E)、Offline RL 长期留存与生命周期价值 (LTV) 策略优化",
    },
    "debias_fairness_industrial": {
        "title": "去偏公平、可信隐私与工业架构",
        "en_title": "Debiasing, Fairness, Privacy & Industrial Systems",
        "desc": "曝光/流行度/位置因果去偏 (Causal Rec)、双边公平性、联邦防投毒推荐、工业级全链路架构与评测基准",
    },
}


SUPPORTED_YEARS = ["2026"]
DEFAULT_YEAR = "2026"


def clean_legacy_paper_notes(repo_root: Path):
    """清空原仓库中非推荐系统的 2.3 万篇 CV/NLP/ML 历史目录，保留主题框架与站点级文件。"""
    docs_dir = repo_root / "docs"
    if not docs_dir.exists():
        docs_dir.mkdir(parents=True, exist_ok=True)
        return

    keep_names = {
        "assets",
        "stylesheets",
        "javascripts",
        "venues",
        "topics",
        "years",
        "index.md",
        "robots.txt",
    }
    removed = []
    for item in docs_dir.iterdir():
        if item.name in keep_names:
            continue
        if item.is_dir():
            shutil.rmtree(item)
            removed.append(item.name)
        elif item.is_file():
            item.unlink()
            removed.append(item.name)

    hreflang_file = repo_root / ".hreflang_pairs.json"
    if hreflang_file.exists():
        hreflang_file.write_text("{}\n", encoding="utf-8")

    print(f"[Clean] 已清理原仓库历史目录/残留文件共 {len(removed)} 项。")


def _migrate_flat_venue_dirs(venues_dir: Path, years) -> int:
    """把旧的 venues/<VENUE><YEAR>/<category>/ 迁移到 venues/<VENUE>/<YEAR>/<category>/。"""
    moved = 0
    for venue_code in VENUES:
        for year in years:
            old_dir = venues_dir / f"{venue_code}{year}"
            if not old_dir.exists():
                continue
            new_dir = venues_dir / venue_code / year
            new_dir.mkdir(parents=True, exist_ok=True)
            for child in list(old_dir.iterdir()):
                if child.is_dir() and child.name in CATEGORIES:
                    target_cat = new_dir / child.name
                    target_cat.mkdir(parents=True, exist_ok=True)
                    for f in child.iterdir():
                        if f.is_file() and f.name != "index.md":
                            target = target_cat / f.name
                            if not target.exists():
                                shutil.move(str(f), str(target))
                                moved += 1
                elif child.is_dir():
                    shutil.rmtree(child)
            shutil.rmtree(old_dir)
    return moved


def scaffold_recsys_structure(repo_root: Path, years=None):
    """创建 venues/<VENUE>/<YEAR>/<category>/ 与 topics/、years/ 骨架。"""
    years = years or SUPPORTED_YEARS
    docs_dir = repo_root / "docs"
    venues_dir = docs_dir / "venues"
    topics_dir = docs_dir / "topics"
    years_dir = docs_dir / "years"
    venues_dir.mkdir(parents=True, exist_ok=True)
    topics_dir.mkdir(parents=True, exist_ok=True)
    years_dir.mkdir(parents=True, exist_ok=True)

    moved = _migrate_flat_venue_dirs(venues_dir, years)

    for venue_code in VENUES:
        for year in years:
            v_year_dir = venues_dir / venue_code / year
            v_year_dir.mkdir(parents=True, exist_ok=True)
            # 清理可能残留的旧分类目录
            for child in list(v_year_dir.iterdir()):
                if child.is_dir() and child.name not in CATEGORIES:
                    shutil.rmtree(child)
            for cat_slug in CATEGORIES:
                (v_year_dir / cat_slug).mkdir(parents=True, exist_ok=True)

    # 清理 topics 下旧分类的 .md 文件
    for md_f in list(topics_dir.glob("*.md")):
        if md_f.name != "index.md" and md_f.stem not in CATEGORIES:
            md_f.unlink()

    # 删除任何示例/演示笔记，避免计入真实收录统计
    for sample in venues_dir.rglob("sample_*.md"):
        sample.unlink()

    print(
        f"[Scaffold] 已创建 {len(VENUES)} 个会刊 × {len(years)} 个年份 × "
        f"{len(CATEGORIES)} 个推荐系统正交细分类别目录（迁移旧笔记 {moved} 篇）。"
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    clean_legacy_paper_notes(root)
    scaffold_recsys_structure(root)
