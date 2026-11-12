#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classify_curated_csv.py
对现有策展 CSV 逐篇抓取 arXiv 标题/摘要，调用 LLM 判断推荐相关性并修正主分类。
默认会删除高置信度判定为“非推荐系统核心研究”的条目，可用 --no-remove 关闭。

用法：
  python scripts/classify_curated_csv.py --all
  python scripts/classify_curated_csv.py --csv data/papers/SIGIR2026.csv --limit 10
  python scripts/classify_curated_csv.py --all --dry-run
"""

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from init_clean_recsys_repo import CATEGORIES
from generate_recsys_notes import (
    fetch_arxiv_by_ids,
    resolve_api_key,
    resolve_base_url,
    resolve_model,
)
from paper_classifier import DEFAULT_CACHE_PATH, classify_items
from build_curated_list import classify as heuristic_classify

FIELDNAMES = ["title", "arxiv_id", "paper_url", "code_url", "category", "venue", "year"]


def read_rows(csv_path: Path) -> list:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(csv_path: Path, rows: list):
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="用 LLM 复核策展 CSV 的推荐相关性与主分类")
    parser.add_argument("--csv", action="append", default=[], help="CSV 路径，可重复传入")
    parser.add_argument("--all", action="store_true", help="处理 data/papers/*.csv 下的全部名单")
    parser.add_argument("--limit", type=int, default=0, help="每个 CSV 最多处理前 N 篇（0 表示全部）")
    parser.add_argument("--workers", type=int, default=4, help="并发 LLM 线程数（默认 4）")
    parser.add_argument("--retries", type=int, default=2, help="单篇失败重试次数（默认 2）")
    parser.add_argument("--api-key", default="", help="API Key（或 LLM_API_KEY / DEEPSEEK_API_KEY）")
    parser.add_argument("--base-url", default="", help="API Base URL")
    parser.add_argument("--model", default="", help="模型名称")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="删除非推荐条目所需的最低置信度")
    parser.add_argument("--no-remove", action="store_true", help="只修正分类，不删除非推荐条目")
    parser.add_argument("--dry-run", action="store_true", help="只打印变更，不写回 CSV")
    parser.add_argument("--refresh", action="store_true", help="忽略本地 LLM 分类缓存")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE_PATH), help="LLM 分类缓存路径")
    args = parser.parse_args()

    repo_root = SCRIPT_DIR.parent
    if args.csv:
        csv_paths = [Path(p) for p in args.csv]
    elif args.all:
        csv_paths = sorted((repo_root / "data" / "papers").glob("*.csv"))
    else:
        parser.error("请指定 --csv 或 --all")

    csv_paths = [p for p in csv_paths if p.exists()]
    if not csv_paths:
        print("[Error] 没有找到可处理的 CSV")
        return

    api_key = resolve_api_key(args.api_key)
    if not api_key:
        print("[Error] 未检测到 API Key，请设置 LLM_API_KEY 或 DEEPSEEK_API_KEY")
        return
    base_url = resolve_base_url(args.base_url)
    model = resolve_model(args.model)

    file_rows = {}
    all_ids = []
    for csv_path in csv_paths:
        rows = read_rows(csv_path)
        if args.limit and args.limit > 0:
            rows = rows[: args.limit]
        file_rows[csv_path] = rows
        all_ids.extend(
            row.get("arxiv_id", "").strip()
            for row in rows
            if row.get("arxiv_id", "").strip()
        )

    unique_ids = list(dict.fromkeys(all_ids))
    print(f"[Fetch] 正在获取 {len(unique_ids)} 篇论文的标题与摘要 ...")
    papers = fetch_arxiv_by_ids(unique_ids, cache_dir=None, download_pdf=False)
    by_id = {p.get("arxiv_id", ""): p for p in papers}

    items = []
    for arxiv_id in unique_ids:
        paper = by_id.get(arxiv_id, {})
        abstract = ""
        if paper:
            abstract = paper.get("content", "").split("Abstract:\n", 1)[-1]
        items.append(
            {
                "arxiv_id": arxiv_id,
                "title": paper.get("title", ""),
                "abstract": abstract,
            }
        )

    print(f"[LLM] 使用 {model} 分类 {len(items)} 篇论文 ...")
    results = classify_items(
        items,
        api_key=api_key,
        base_url=base_url,
        model=model,
        workers=args.workers,
        retries=args.retries,
        cache_path=Path(args.cache),
        refresh=args.refresh,
    )

    for csv_path, rows in file_rows.items():
        kept = []
        removed = []
        changed = []
        errors = []
        for row in rows:
            arxiv_id = row.get("arxiv_id", "").strip()
            result = results.get(arxiv_id)
            if not result or result.get("source") == "error":
                errors.append(row.get("title", arxiv_id))
                kept.append(row)
                continue

            is_relevant = result.get("is_recommendation")
            confidence = float(result.get("confidence", 0.0))
            if (
                is_relevant is False
                and confidence >= args.min_confidence
                and not args.no_remove
            ):
                removed.append(
                    (row.get("title", ""), confidence, result.get("reason", ""))
                )
                continue

            new_category = result.get("category", "")
            old_category = row.get("category", "")
            if new_category in CATEGORIES and new_category != old_category:
                changed.append((row.get("title", ""), old_category, new_category))
                row["category"] = new_category
            kept.append(row)

        print(f"\n{'=' * 90}\n{csv_path.name}: {len(rows)} -> {len(kept)}")
        if removed:
            print(f"  [删除] LLM 判定非推荐相关 {len(removed)} 篇：")
            for title, confidence, reason in removed:
                print(f"    - ({confidence:.2f}) {title[:90]} | {reason}")
        if changed:
            print(f"  [改类] {len(changed)} 篇：")
            for title, old, new in changed:
                print(f"    - {old} -> {new}: {title[:90]}")
        if errors:
            print(f"  [警告] {len(errors)} 篇 LLM 失败，保留原分类：")
            for title in errors:
                print(f"    - {title[:90]}")

        if not args.dry_run:
            write_rows(csv_path, kept)
            print(f"  [写入] {csv_path}")
        else:
            print("  [Dry-run] 未写入")

    print("\n[Done] LLM 分类复核完成。")


if __name__ == "__main__":
    main()
