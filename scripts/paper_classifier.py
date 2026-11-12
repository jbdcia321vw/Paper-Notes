#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_classifier.py
使用 LLM 根据论文标题与摘要判断“是否属于推荐系统核心研究”，并从 16 个方向中
选择 1 个主分类。供 build_curated_list.py 与 classify_curated_csv.py 复用。
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

CATEGORY_LINES = "\n".join(
    f"- {slug}: {info['title']} / {info['en_title']} — {info['desc']}"
    for slug, info in CATEGORIES.items()
)

SYSTEM_PROMPT = f"""你是推荐系统（Recommender Systems）与信息检索（IR）领域的资深程序委员会成员。
你的任务是判断一篇论文是否属于“推荐系统核心研究”，并从给定的 16 个分类中选出 1 个最核心的主分类。

## 16 个分类
{CATEGORY_LINES}

## 判定标准
- 推荐系统核心研究包括：推荐/排序/个性化/用户建模/CTR/CVR/广告/搜索推荐/对话推荐/生成式推荐/推荐评估/推荐公平性等。
- 纯信息检索（如 ad-hoc search、文档检索、问答、RAG 检索、查询改写、社区搜索）如果不以推荐/个性化/排序为核心，应判为不相关。
- 纯 NLP/CV/ML 方法论文如果没有明确面向推荐任务，应判为不相关。
- 工业界论文只要核心任务是推荐/广告/排序/用户行为建模，应判为相关。
- 只依据标题和摘要判断，不要臆测摘要中没有的内容。

## 分类选择优先级（先判断核心任务，再看技术）
1. 核心任务是 CTR/CVR/转化率/广告竞价/sponsored search/实时竞价 → `ctr_feature_interaction`
2. 核心任务是序列/会话/next-item/next-basket/长序列建模 → `sequential_rec`
3. 核心任务是生成式检索/语义 ID/自回归生成/扩散或流式生成推荐 → `generative_rec`
4. 核心任务是对话/多轮交互/agent/用户模拟 → `agent_conversational_rec`
5. 核心任务是重排/listwise/多任务/多目标/pre-ranking → `multi_task_reranking`
6. 核心任务是跨域/多场景/多市场 → `cross_domain_multi_scenario`
7. 核心任务是冷启动/长尾/用户召回/零观测 → `cold_start_rec`
8. 核心任务是公平/偏差/隐私/因果/鲁棒性 → `debias_fairness_industrial`
9. 核心任务是工业全链路架构/在线服务/实验平台/评估基准 → `debias_fairness_industrial`（仅当没有更具体的任务类别时）
10. 核心任务是 LLM 作为推荐器或 LLM4Rec → `llm_rec`
11. 核心任务是图/知识图谱 → `graph_rec`
12. 核心任务是对比学习/自监督/去噪 → `self_supervised_denoising`
13. 核心任务是强化学习/bandit/OPE/长期收益 → `rl_bandit_rec`
14. 核心任务是社交/群组 → `social_group_rec`
15. 核心任务是 POI/时空 → `poi_spatiotemporal_rec`
16. 核心任务是协同过滤/双塔/向量召回 → `collaborative_filtering`
17. 核心任务是多媒体内容理解/多模态融合 → `multimodal_rec`

注意：
- 不要因为论文使用了 LLM 就归入 `llm_rec`；如果核心任务是对话、序列、生成式、CTR 等，优先选对应任务类别。
- 不要因为有工业数据或线上实验就归入 `debias_fairness_industrial`；只有偏差/公平/隐私/因果/鲁棒性，或全链路架构/评估基准本身是核心贡献时才选它。
- 广告/竞价/赞助搜索即使使用了生成式模型，也优先归入 `ctr_feature_interaction`。
- 含 “Listwise / Re-ranking / Position Bias” 的 LLM 排序论文优先归入 `multi_task_reranking`。

## 输出
只输出一个 JSON 对象，不要输出 Markdown 代码块或其他文字：
{{"is_recommendation": true, "category": "llm_rec", "confidence": 0.92, "reason": "一句话理由"}}
- is_recommendation: 布尔值
- category: 必须是上述 16 个 slug 之一；不相关时也填最接近的分类
- confidence: 0 到 1 的浮点数
- reason: 不超过 40 个汉字
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
    """对单篇论文做相关性 + 主分类判断。失败时返回 source=error。"""
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
    label: str = "LLM 分类",
) -> dict:
    """并发分类，返回 {arxiv_id: result}，并复用本地缓存。"""
    cache_path = Path(cache_path) if cache_path else None
    cache = load_cache(cache_path)
    results = {}
    pending = []

    for item in items:
        arxiv_id = str(item.get("arxiv_id", "")).strip()
        if not arxiv_id:
            continue
        cache_key = f"{model}:{arxiv_id}"
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
                arxiv_id = str(item.get("arxiv_id", "")).strip()
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
                cache[f"{model}:{arxiv_id}"] = result
                status = "error" if result.get("source") == "error" else "ok"
                detail = (
                    f"{arxiv_id} -> {result['category'] or '-'} "
                    f"({result['confidence']:.2f}) {result['reason'][:50]}"
                )
                reporter.update(status, title=item.get("title", ""), detail=detail)
        reporter.finish()
        save_cache(cache_path, cache)

    return results
