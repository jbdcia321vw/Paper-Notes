#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classify_curated_csv.py (v3.3 严格剔除非推荐论文 + 自动同步删除/移动已生成 .md 笔记与图片版)
对现有策展 CSV 逐篇抓取 arXiv 标题/摘要，调用 LLM 严格判断推荐相关性并修正主分类。

核心升级：
  1. 修复 arXiv ID 版本号 (`v1/v2`) 字典匹配失败导致摘要为空的 Bug。
  2. 当某篇论文被判定为“非推荐系统核心研究”并从 CSV 删除时，自动同步删除 `docs/venues/` 下已生成的 `.md` 笔记、`docs/assets/papers/` 下的对应图片目录及 `data/manifest.jsonl` 记录！
  3. 当某篇论文的主分类发生变更 (`old_category -> new_category`) 时，自动将其已生成的 `.md` 笔记移动到新分类目录并更新 Frontmatter，无需重复花钱重生成！
  4. 自动扫描清理 `docs/venues/<venue>/<year>/` 下不在 CSV 名单中的孤儿 `.md` 笔记。

用法：
  python scripts/classify_curated_csv.py --all
  python scripts/classify_curated_csv.py --csv data/papers/SIGIR2026.csv
  python scripts/classify_curated_csv.py --all --dry-run
"""

import argparse
import csv
import json
import re
import shutil
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from init_clean_recsys_repo import CATEGORIES
from generate_recsys_notes import (
    fetch_arxiv_by_ids,
    print_usage_summary,
    resolve_api_key,
    resolve_base_url,
    resolve_model,
    slugify,
)
from paper_classifier import DEFAULT_CACHE_PATH, classify_items
from run_logging import RunLog, format_duration

FIELDNAMES = ["title", "arxiv_id", "paper_url", "code_url", "category", "venue", "year"]


def read_rows(csv_path: Path) -> list:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(csv_path: Path, rows: list):
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def remove_paper_artifacts(repo_root: Path, title: str, arxiv_id: str, venue: str, year: str, dry_run: bool = False) -> list:
    """删除被排除论文在 docs/venues/ 下的 .md 笔记与 docs/assets/papers/ 下的图片目录。"""
    docs_dir = repo_root / "docs"
    slug = slugify(title)
    deleted_paths = []

    search_root = docs_dir / "venues" / venue / year if (venue and year and (docs_dir / "venues" / venue / year).exists()) else docs_dir / "venues"
    if search_root.exists():
        for md_file in list(search_root.rglob(f"{slug}.md")):
            deleted_paths.append(str(md_file.relative_to(repo_root)))
            if not dry_run:
                try:
                    md_file.unlink()
                except OSError:
                    pass

    for key in {re.sub(r"[^a-zA-Z0-9._-]", "_", arxiv_id), slug}:
        if not key:
            continue
        asset_dir = docs_dir / "assets" / "papers" / key
        if asset_dir.exists() and asset_dir.is_dir():
            deleted_paths.append(str(asset_dir.relative_to(repo_root)))
            if not dry_run:
                shutil.rmtree(asset_dir, ignore_errors=True)

    return deleted_paths


def move_paper_note_category(repo_root: Path, title: str, venue: str, year: str, old_cat: str, new_cat: str, dry_run: bool = False) -> str:
    """当论文分类变更时，自动将已存在的 .md 笔记迁移至新分类目录并更新 Frontmatter。"""
    if not venue or not year or new_cat not in CATEGORIES:
        return ""
    docs_dir = repo_root / "docs"
    slug = slugify(title)
    venue_year_dir = docs_dir / "venues" / venue / year
    if not venue_year_dir.exists():
        return ""

    matches = [p for p in venue_year_dir.rglob(f"{slug}.md") if p.name != "index.md"]
    if not matches:
        return ""

    src_md = matches[0]
    dst_dir = venue_year_dir / new_cat
    dst_md = dst_dir / f"{slug}.md"
    if src_md.resolve() == dst_md.resolve():
        return ""

    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)
        text = src_md.read_text(encoding="utf-8", errors="ignore")
        # 更新 YAML Frontmatter 中的主分类
        text = re.sub(
            r"(categories:\s*\n\s*-\s*)([a-z0-9_]+)",
            rf"\g<1>{new_cat}",
            text,
            count=1,
        )
        dst_md.write_text(text, encoding="utf-8")
        try:
            src_md.unlink()
        except OSError:
            pass
    return f"{src_md.relative_to(repo_root)} -> {dst_md.relative_to(repo_root)}"


def prune_orphan_notes(repo_root: Path, venue: str, year: str, valid_slugs: set, dry_run: bool = False) -> list:
    """清理 docs/venues/<venue>/<year>/ 下不在当前 CSV 有效名单中的孤儿笔记。"""
    docs_dir = repo_root / "docs"
    venue_year_dir = docs_dir / "venues" / venue / year
    pruned = []
    if not venue_year_dir.exists():
        return pruned

    for md_file in list(venue_year_dir.rglob("*.md")):
        if md_file.name == "index.md":
            continue
        if md_file.stem not in valid_slugs:
            pruned.append(str(md_file.relative_to(repo_root)))
            if not dry_run:
                try:
                    md_file.unlink()
                except OSError:
                    pass
    return pruned


def main():
    parser = argparse.ArgumentParser(description="用 LLM 严格复核策展 CSV 的推荐相关性与主分类，并同步清理非推荐笔记")
    parser.add_argument("--csv", action="append", default=[], help="CSV 路径，可重复传入")
    parser.add_argument("--all", action="store_true", help="处理 data/papers/*.csv 下的全部名单")
    parser.add_argument("--limit", type=int, default=0, help="每个 CSV 最多处理前 N 篇（0 表示全部）")
    parser.add_argument("--workers", type=int, default=4, help="并发 LLM 线程数（默认 4）")
    parser.add_argument("--retries", type=int, default=2, help="单篇失败重试次数（默认 2）")
    parser.add_argument("--api-key", default="", help="API Key（或 LLM_API_KEY / DEEPSEEK_API_KEY）")
    parser.add_argument("--base-url", default="", help="API Base URL")
    parser.add_argument("--model", default="", help="模型名称")
    parser.add_argument("--min-confidence", type=float, default=0.55, help="删除非推荐条目所需的最低置信度（默认 0.55）")
    parser.add_argument("--no-remove", action="store_true", help="只修正分类，不删除非推荐条目")
    parser.add_argument("--no-prune-orphans", action="store_true", help="不自动清理孤儿 .md 笔记")
    parser.add_argument("--dry-run", action="store_true", help="只打印变更，不写回 CSV 或删除文件")
    parser.add_argument("--refresh", action="store_true", help="忽略本地 LLM 分类缓存")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE_PATH), help="LLM 分类缓存路径")
    args = parser.parse_args()
    run_start = time.monotonic()

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
    log = RunLog("classify")
    log.write(
        f"START files={len(csv_paths)} workers={args.workers} retries={args.retries} "
        f"model={model} no_remove={args.no_remove} dry_run={args.dry_run}"
    )

    file_rows = {}
    all_ids = []
    row_title_fallback = {}
    for csv_path in csv_paths:
        rows = read_rows(csv_path)
        if args.limit and args.limit > 0:
            rows = rows[: args.limit]
        file_rows[csv_path] = rows
        for row in rows:
            aid = re.sub(r"v\d+$", "", (row.get("arxiv_id") or "").strip())
            if aid:
                row["arxiv_id"] = aid
                all_ids.append(aid)
                row_title_fallback[aid] = row.get("title", "")

    unique_ids = list(dict.fromkeys(all_ids))
    print(f"[Fetch] 正在从 arXiv 批量获取 {len(unique_ids)} 篇论文的标题与摘要 ...")
    papers = fetch_arxiv_by_ids(unique_ids, cache_dir=None, download_pdf=False)
    by_id = {re.sub(r"v\d+$", "", p.get("arxiv_id", "")): p for p in papers}

    items = []
    for arxiv_id in unique_ids:
        paper = by_id.get(arxiv_id, {})
        abstract = ""
        if paper:
            abstract = paper.get("content", "").split("Abstract:\n", 1)[-1]
        items.append(
            {
                "arxiv_id": arxiv_id,
                "title": paper.get("title") or row_title_fallback.get(arxiv_id, ""),
                "abstract": abstract,
            }
        )

    print(f"[LLM] 使用 {model} 严格复核 {len(items)} 篇论文的推荐系统相关性与主分类 ...")
    results = classify_items(
        items,
        api_key=api_key,
        base_url=base_url,
        model=model,
        workers=args.workers,
        retries=args.retries,
        cache_path=Path(args.cache),
        refresh=args.refresh,
        log=log,
    )

    for csv_path, rows in file_rows.items():
        kept = []
        removed = []
        changed = []
        errors = []
        deleted_files = []
        moved_files = []

        for row in rows:
            arxiv_id = re.sub(r"v\d+$", "", (row.get("arxiv_id") or "").strip())
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
                d_paths = remove_paper_artifacts(
                    repo_root,
                    title=row.get("title", ""),
                    arxiv_id=arxiv_id,
                    venue=row.get("venue", ""),
                    year=row.get("year", ""),
                    dry_run=args.dry_run,
                )
                deleted_files.extend(d_paths)
                continue

            new_category = result.get("category", "")
            old_category = row.get("category", "")
            if new_category in CATEGORIES and new_category != old_category:
                changed.append((row.get("title", ""), old_category, new_category))
                mv = move_paper_note_category(
                    repo_root,
                    title=row.get("title", ""),
                    venue=row.get("venue", ""),
                    year=row.get("year", ""),
                    old_cat=old_category,
                    new_cat=new_category,
                    dry_run=args.dry_run,
                )
                if mv:
                    moved_files.append(mv)
                row["category"] = new_category
            kept.append(row)

        pruned_orphans = []
        if not args.no_prune_orphans and not args.limit and kept:
            v_name = kept[0].get("venue", "")
            y_name = kept[0].get("year", "")
            valid_slugs = {slugify(r.get("title", "")) for r in kept if r.get("title")}
            pruned_orphans = prune_orphan_notes(repo_root, v_name, y_name, valid_slugs, dry_run=args.dry_run)

        print(f"\n{'=' * 90}\n{csv_path.name}: 原始 {len(rows)} 篇 -> 保留纯推荐论文 {len(kept)} 篇")
        if removed:
            print(f"  [剔除非推荐论文] 共 {len(removed)} 篇：")
            for title, confidence, reason in removed:
                print(f"    - ({confidence:.2f}) {title[:85]} | {reason}")
        if deleted_files:
            print(f"  [同步清理已生成笔记/图片] 共 {len(deleted_files)} 项")
        if pruned_orphans:
            print(f"  [清理孤儿笔记] 共 {len(pruned_orphans)} 篇：")
            for p_str in pruned_orphans[:10]:
                print(f"    - {p_str}")
        if changed:
            print(f"  [修正推荐子分类] 共 {len(changed)} 篇（同步迁移已有笔记 {len(moved_files)} 篇）：")
            for title, old, new in changed[:15]:
                print(f"    - {old} -> {new}: {title[:80]}")
        if errors:
            print(f"  [警告] {len(errors)} 篇 LLM 失败，保留原分类")

        if not args.dry_run:
            write_rows(csv_path, kept)
            print(f"  [已更新 CSV] {csv_path}")
        else:
            print("  [Dry-run] 未写入文件变更")

    if not args.dry_run:
        print("\n[Index] 正在同步刷新全站导航索引与卡片计数 ...")
        from build_nav_and_index import (
            collect_all_notes,
            discover_years,
            build_topic_pages,
            build_year_pages,
            build_venue_pages,
            build_home_and_mkdocs_yml,
        )
        docs_dir = repo_root / "docs"
        all_notes = collect_all_notes(docs_dir)
        all_years = discover_years(docs_dir, all_notes)
        build_topic_pages(docs_dir, all_notes)
        build_year_pages(docs_dir, all_notes, all_years)
        build_venue_pages(docs_dir, all_notes, all_years)
        build_home_and_mkdocs_yml(repo_root, all_notes, all_years)

    print("\n[Done] LLM 严格推荐相关性清洗与分类复核完成。")
    elapsed = time.monotonic() - run_start
    success = sum(1 for result in results.values() if result.get("source") != "error")
    print_usage_summary(
        model,
        processed=len(items),
        success=success,
        elapsed_seconds=elapsed,
        log=log,
    )
    print(f"[Time] 分类复核总耗时 {format_duration(elapsed)}")
    print(f"[Log] 详细日志: {log.path}")
    log.write(f"TOTAL elapsed={format_duration(elapsed)} items={len(items)} success={success}")
    log.close()


if __name__ == "__main__":
    main()