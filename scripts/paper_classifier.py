#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_classifier.py (v3.3 严格推荐系统边界过滤版)
使用 LLM 根据论文标题与摘要判断“是否属于推荐系统核心研究”，并从 16 个方向中
选择 1 个主分类。供 build_curated_list.py 与 classify_curated_csv.py 复用。

核心升级：
  - 严格收紧「推荐系统 (RecSys)」与「纯信息检索 / 网页搜索 (Web Search / IR)」的边界：
    明确将纯网页搜索点击模型 (SERP Click Models 如 PBM/UBM)、给定显式 Query 的文档/段落检索 (MS MARCO / TREC / BEIR)、
    开放域问答与通用 RAG、无个性化推荐交互的通用图文检索/图分类判为 `is_recommendation: false`。
  - 缓存键升级为 `v2:{model}:{arxiv_id}`，自动失效之前宽松标准下的旧缓存。
"""

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from init_clean_recsys_repo import CATEGORIES
from generate_recsys_notes import call_chat_completion, estimate_usage_cost_usd
from run_logging import ProgressReporter, RunLog

DEFAULT_CACHE_PATH = SCRIPT_DIR.parent / "data" / "papers" / ".classification_cache.json"
CACHE_SCHEMA_VERSION = "v2_strict_recsys"

CATEGORY_LINES = "\n".join(
    f"- {slug}: {info['title']} / {info['en_title']} — {info['desc']}"
    for slug, info in CATEGORIES.items()
)

SYSTEM_PROMPT = f"""你是 ACM RecSys 与 SIGIR 推荐系统轨（Recommender Systems Track）的资深领域主席（Area Chair）。
你的任务是严格判断一篇论文是否属于**「推荐系统（Recommender Systems）与个性化计算广告核心研究」**，并从给定的 16 个推荐细分方向中选出 1 个最核心的主分类。

## 16 个推荐系统细分方向
{CATEGORY_LINES}

## 严格相关性判定标准（必须把「推荐系统」与「纯搜索引擎/文档检索」严格区分！）

### 1. 应判为 `is_recommendation: true`（属于本库收录范围）：
- **核心任务是向用户个性化推荐物品/内容/服务/好友/广告**：包括协同过滤、双塔向量召回、序列/会话推荐、生成式推荐（Semantic ID / RQ-VAE / Diffusion Rec）、大模型推荐（LLM4Rec）、对话推荐系统（CRS / 主动偏好询问）、多模态短视频/电商推荐、社交/群组推荐、POI 时空签到推荐、冷启动与跨域/多场景推荐、CTR/CVR 点击转化率预估与计算广告竞价、推荐多任务精排与列表重排（Re-ranking for RecSys）、推荐系统去偏/公平/隐私/防投毒、推荐强化学习/Bandit 以及工业级推荐系统架构与推荐评测。
- **个性化搜索与搜推融合**：仅当搜索系统明确建模了**用户个性化历史行为序列/用户画像**，或服务于**电商商品推荐/赞助商广告推荐（Sponsored Search / E-commerce Product Search & Recommendation）**时，才判为 `true`。

### 2. 应判为 `is_recommendation: false`（必须严格排除！）：
- **纯网页搜索与文档/段落检索 (Ad-hoc Web Search & Passage/Document Retrieval)**：针对给定显式文本 Query 检索网页或文档段落（如基于 MS MARCO、BEIR、TREC、Natural Questions 的 Dense Retrieval、BM25、SPLADE、LLM 零样本/少样本文档重排 Document Re-ranking、查询改写 Query Rewriting/Expansion）。
- **纯搜索引擎结果页点击模型 (Web Search SERP Click Models & ULTR)**：仅研究传统网页搜索引擎结果页（如 Yandex、Baidu Search 日志）上的查询-网址位置偏差点击模型（如经典 PBM、UBM、DBN 点击模型或纯 Web Search Unbiased Learning to Rank），而没有针对个性化物品推荐场景建模。
- **通用问答、智能体网页检索与 RAG (Open-Domain QA, Web Search Agents & General RAG)**：检索增强文本生成、医疗/法律问答、知识库问答（KBQA）、代码检索、数据集搜索、跨模态图文匹配检索（如 COCO/Flickr30k 图搜文/文搜图，无用户历史交互序列）。
- **通用机器学习/图学习/NLP/CV 论文**：虽在标题或摘要中顺带提及“可用于推荐”，但核心方法与实验完全在非推荐数据集（如 Cora/CiteSeer 节点分类、ImageNet、GLUE 等）上进行。

## 分类选择优先级（当 `is_recommendation: true` 时，先看核心业务问题，再看技术手段）
1. 核心任务是 CTR/CVR 预估、特征交叉网络、转化延迟反馈、计算广告竞价 → `ctr_feature_interaction`
2. 核心任务是序列推荐、会话推荐、Next-item/Next-basket、长短期行为序列建模 → `sequential_rec`
3. 核心任务是生成式推荐、离散语义 ID (RQ-VAE/VQ)、自回归生成检索、扩散/流匹配生成推荐 → `generative_rec`
4. 核心任务是多轮对话推荐 (CRS)、主动偏好询问 (Preference Elicitation)、推荐 Agent 与用户行为仿真 → `agent_conversational_rec`
5. 核心任务是多任务/多目标学习 (MMoE/PLE/Pareto)、列表级重排 (Listwise Re-ranking)、粗精排级联一致性 → `multi_task_reranking`
6. 核心任务是跨域推荐 (CDR)、多场景/多业务线联合推荐 → `cross_domain_multi_scenario`
7. 核心任务是严格冷启动 (Zero-shot/Few-shot 新用户或新物品)、长尾物品推荐 → `cold_start_rec`
8. 核心任务是推荐系统曝光/流行度/位置去偏、因果推荐、推荐公平性、隐私保护、防投毒鲁棒性或工业推荐评测基准 → `debias_fairness_industrial`
9. 核心任务是利用大语言模型直接做推荐排序、指令微调推荐或 LLM 知识蒸馏增强推荐 → `llm_rec`
10. 核心任务是图神经网络推荐 (LightGCN/超图)、知识图谱推荐 (KGRec) → `graph_rec`
11. 核心任务是推荐对比学习、自监督数据增广、隐式反馈假阳性去噪 → `self_supervised_denoising`
12. 核心任务是强化学习推荐、Contextual Bandit 探索与利用、长期留存/LTV 优化 → `rl_bandit_rec`
13. 核心任务是社交网络信任推荐、群组推荐 (Group Rec) → `social_group_rec`
14. 核心任务是 POI 兴趣点推荐、时空轨迹移动推荐 → `poi_spatiotemporal_rec`
15. 核心任务是基础协同过滤、矩阵分解、双塔向量召回与 ANN 索引 → `collaborative_filtering`
16. 核心任务是多模态特征（视觉/文本/音频）融合与解耦推荐、微视频推荐 → `multimodal_rec`

## 输出格式
只输出一个合法的 JSON 对象，不要输出 Markdown 代码块或其他文字：
{{"is_recommendation": true, "category": "sequential_rec", "confidence": 0.95, "reason": "一句话说明核心任务与判定依据"}}
- `is_recommendation`: 布尔值（若为纯网页搜索、SERP 点击模型、文档检索、RAG 问答，必须填 `false`！）
- `category`: 必须是上述 16 个 slug 之一
- `confidence`: 0 到 1 的浮点数
- `reason`: 不超过 45 个汉字
"""


def _extract_json(text: str):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def classify_one(
    title: str,
    abstract: str,
    api_key: str,
    base_url: str,
    model: str,
    retries: int = 2,
    timeout: int = 90,
) -> dict:
    """对单篇论文做严格推荐相关性 + 主分类判断。失败时返回 source=error。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"标题：{title}\n\n摘要：\n{(abstract or '')[:6000]}",
        },
    ]
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            kwargs = {"temperature": 0.0, "timeout": timeout, "max_tokens": 512, "thinking": False}
            if attempt == 1:
                kwargs["response_format"] = {"type": "json_object"}
            raw = call_chat_completion(
                messages,
                api_key=api_key,
                base_url=base_url,
                model=model,
                **kwargs,
            )
            data = _extract_json(raw)
            if not isinstance(data, dict):
                raise ValueError("LLM 未返回 JSON 对象")
            category = str(data.get("category", "")).strip()
            if category not in CATEGORIES:
                raise ValueError(f"非法分类 slug: {category}")
            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
            return {
                "is_recommendation": bool(data.get("is_recommendation")),
                "category": category,
                "confidence": confidence,
                "reason": str(data.get("reason", "")).strip()[:120],
                "source": "llm",
            }
        except Exception as e:
            last_error = str(e)
            if attempt < retries:
                time.sleep(1.5 * attempt)
    return {
        "is_recommendation": None,
        "category": "",
        "confidence": 0.0,
        "reason": f"LLM 失败: {last_error}"[:200],
        "source": "error",
    }


def load_cache(cache_path: Path) -> dict:
    if not cache_path or not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_cache(cache_path: Path, cache: dict):
    if not cache_path:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def classify_items(
    items: list,
    api_key: str,
    base_url: str,
    model: str,
    workers: int = 4,
    retries: int = 2,
    cache_path: Path = DEFAULT_CACHE_PATH,
    refresh: bool = False,
    log: RunLog = None,
    label: str = "LLM 严格推荐相关性分类",
) -> dict:
    """并发分类，返回 {arxiv_id: result}，并复用带版本前缀的本地缓存。"""
    cache_path = Path(cache_path) if cache_path else None
    cache = load_cache(cache_path)
    results = {}
    pending = []

    for item in items:
        arxiv_id = re.sub(r"v\d+$", "", str(item.get("arxiv_id", "")).strip())
        if not arxiv_id:
            continue
        cache_key = f"{CACHE_SCHEMA_VERSION}:{model}:{arxiv_id}"
        if not refresh and cache_key in cache:
            results[arxiv_id] = cache[cache_key]
        else:
            pending.append(item)

    if pending:
        reporter = ProgressReporter(
            total=len(pending),
            label=label,
            log=log,
            cost_fn=lambda: estimate_usage_cost_usd(model),
        )
        reporter.start()
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            future_map = {
                executor.submit(
                    classify_one,
                    item.get("title", ""),
                    item.get("abstract", ""),
                    api_key,
                    base_url,
                    model,
                    retries,
                ): item
                for item in pending
            }
            for future in as_completed(future_map):
                item = future_map[future]
                arxiv_id = re.sub(r"v\d+$", "", str(item.get("arxiv_id", "")).strip())
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "is_recommendation": None,
                        "category": "",
                        "confidence": 0.0,
                        "reason": f"线程异常: {e}"[:200],
                        "source": "error",
                    }
                results[arxiv_id] = result
                cache[f"{CACHE_SCHEMA_VERSION}:{model}:{arxiv_id}"] = result
                status = "error" if result.get("source") == "error" else "ok"
                rel_tag = "RecSys" if result.get("is_recommendation") else "Exclude"
                detail = (
                    f"{arxiv_id} [{rel_tag}] -> {result['category'] or '-'} "
                    f"({result['confidence']:.2f}) {result['reason'][:50]}"
                )
                reporter.update(status, title=item.get("title", ""), detail=detail)
        reporter.finish()
        save_cache(cache_path, cache)

    return results