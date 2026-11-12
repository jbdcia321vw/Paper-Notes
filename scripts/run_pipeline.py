#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py (v3.1 防截断 16384 输出上限 + 多会刊批量自动巡检版)
一键串联论文笔记流水线：候选名单 -> LLM 分类 -> 生成笔记 -> 导航索引 -> 站点构建/链接检查。

示例：
  # 1. 重刷 3 篇论文检验新版深度总结质量（默认关闭冗余 thinking 防止挤占正文，或开启自动 16k 上限）
  python scripts/run_pipeline.py --venue SIGIR --year 2026 --profile quality --overwrite --limit 3

  # 2. 自动巡检多个顶会顶刊（增量发现新论文 + 生成笔记 + 构建站点）
  python scripts/run_pipeline.py --venues SIGIR,RecSys,KDD,WWW,WSDM,CIKM,ACM_MM,TOIS,TKDE --year 2026 --build-list --use-dblp
"""

import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_logging import RunLog, format_duration

PROFILES = {
    "fast": {
        "workers": 4,
        "max_input_chars": 48000,
        "max_output_tokens": 8192,
        "max_figures": 6,
        "translate_figures": False,
    },
    "balanced": {
        "workers": 3,
        "max_input_chars": 65000,
        "max_output_tokens": 16384,
        "max_figures": 8,
        "translate_figures": True,
    },
    "quality": {
        "workers": 2,
        "max_input_chars": 80000,
        "max_output_tokens": 16384,
        "max_figures": 10,
        "translate_figures": True,
    },
}


def is_peak_at(moment: datetime) -> bool:
    if moment.weekday() >= 5:
        return False
    return (1 <= moment.hour < 4) or (6 <= moment.hour < 10)


def is_peak_now() -> bool:
    return is_peak_at(datetime.now(timezone.utc))


def seconds_until_off_peak(now: datetime = None) -> int:
    now = now or datetime.now(timezone.utc)
    for minutes in range(0, 24 * 60):
        if not is_peak_at(now + timedelta(minutes=minutes)):
            return minutes * 60
    return 0


def wait_for_off_peak() -> None:
    wait = seconds_until_off_peak()
    if wait <= 0:
        return
    print(
        f"[Cost] 当前为 DeepSeek 高峰时段，等待约 {wait / 3600:.1f} 小时进入非高峰价 ...",
        flush=True,
    )
    time.sleep(wait)


def run_command(cmd: list, dry_run: bool = False, log: RunLog = None, label: str = "") -> float:
    display = " ".join(shlex.quote(str(part)) for part in cmd)
    title = f"{label} " if label else ""
    print(f"\n[Run] {title}{display}", flush=True)
    if log:
        log.write(f"RUN {title}{display}")
    if dry_run:
        return 0.0
    start = time.monotonic()
    try:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    except Exception as e:
        if log:
            log.write(f"ERROR {title}{e}")
        raise
    elapsed = time.monotonic() - start
    print(f"[Done] {label or '命令'} 用时 {format_duration(elapsed)}", flush=True)
    if log:
        log.write(f"DONE {title}elapsed={format_duration(elapsed)}")
    return elapsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="一键运行论文笔记自动巡检与生成流水线")
    parser.add_argument("--venue", default="SIGIR", help="单个会刊 key（默认 SIGIR）")
    parser.add_argument("--venues", default="", help="批量巡检多个会刊（逗号分隔，如 SIGIR,RecSys,KDD,WWW,TOIS）")
    parser.add_argument("--year", default="2026", help="年份（默认 2026）")
    parser.add_argument("--csv", default="", help="策展 CSV；默认 data/papers/<VENUE><YEAR>.csv")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="balanced", help="成本/质量预设（默认 balanced，65k 输入 / 16k 输出）")
    parser.add_argument("--limit", type=int, default=0, help="最多处理前 N 篇（0 表示全部）")
    parser.add_argument("--workers", type=int, default=0, help="并发数；0 表示使用 profile 默认值")
    parser.add_argument("--retries", type=int, default=2, help="失败重试次数（默认 2）")
    parser.add_argument("--model", default="", help="覆盖模型名（默认 deepseek-flash）")
    parser.add_argument("--thinking", action="store_true", help="启用 thinking 思维链模式")
    parser.add_argument("--build-list", action="store_true", help="先从 arXiv / DBLP 自动增量更新候选 CSV")
    parser.add_argument("--use-dblp", action="store_true", help="--build-list 时同步查询 DBLP 官方目录并反查 arXiv")
    parser.add_argument("--build-list-max-results", type=int, default=300, help="--build-list 的 arXiv 检索上限")
    parser.add_argument("--build-list-no-llm", action="store_true", help="--build-list 时只用关键词分类")
    parser.add_argument("--classify", action="store_true", help="用 LLM 复核 CSV 相关性与分类")
    parser.add_argument("--allow-classify-remove", action="store_true", help="允许 --classify 删除非推荐条目（默认只改分类）")
    parser.add_argument("--skip-generate", action="store_true", help="跳过笔记生成，只重建导航/站点")
    parser.add_argument("--skip-site", action="store_true", help="跳过 mkdocs build（CI 部署前使用）")
    parser.add_argument("--no-check-links", action="store_true", help="跳过站点本地链接检查")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的笔记（重刷旧笔记质量时使用）")
    parser.add_argument("--no-figures", action="store_true", help="不提取图表")
    parser.add_argument("--translate-figures", action="store_true", help="强制用 LLM 翻译图注")
    parser.add_argument("--max-input-chars", type=int, default=None, help="覆盖送入 LLM 的最大字符数（0 表示不限制）")
    parser.add_argument("--max-output-tokens", type=int, default=None, help="覆盖最大输出 token（0 表示用 API 默认）")
    parser.add_argument("--max-figures", type=int, default=None, help="覆盖每篇最大图表数")
    parser.add_argument("--off-peak-wait", action="store_true", help="高峰时段自动等到非高峰再运行")
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的命令")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    pipeline_start = time.monotonic()
    log = None if args.dry_run else RunLog("pipeline")

    venue_list = [v.strip() for v in args.venues.split(",") if v.strip()] if args.venues else [args.venue]

    if log:
        log.write(
            f"START profile={args.profile} venues={','.join(venue_list)} year={args.year} "
            f"csv={args.csv or 'default'} limit={args.limit} workers={args.workers} "
            f"model={args.model or 'default'} thinking={args.thinking} "
            f"build_list={args.build_list} use_dblp={args.use_dblp} classify={args.classify} "
            f"skip_generate={args.skip_generate} skip_site={args.skip_site}"
        )

    workers = args.workers or profile["workers"]
    max_input_chars = profile["max_input_chars"] if args.max_input_chars is None else args.max_input_chars
    max_output_tokens = profile["max_output_tokens"] if args.max_output_tokens is None else args.max_output_tokens
    max_figures = profile["max_figures"] if args.max_figures is None else args.max_figures
    translate_figures = args.translate_figures or profile["translate_figures"]
    needs_llm = (args.build_list and not args.build_list_no_llm) or args.classify or not args.skip_generate

    if needs_llm and is_peak_now():
        print("[Cost] 当前是 DeepSeek 高峰时段，API 价格为非高峰的 2 倍。")
        if args.off_peak_wait:
            if args.dry_run:
                print("[Cost] dry-run 不等待；实际运行会等待到非高峰。")
            else:
                wait_for_off_peak()
    elif args.off_peak_wait:
        print("[Cost] 当前已是非高峰时段，无需等待。")

    if needs_llm and not (os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")):
        print("[Warn] 未检测到 LLM_API_KEY / DEEPSEEK_API_KEY；需要 LLM 的步骤可能失败。")

    python = sys.executable

    for v_key in venue_list:
        csv_path = Path(args.csv) if (args.csv and len(venue_list) == 1) else REPO_ROOT / "data" / "papers" / f"{v_key}{args.year}.csv"
        if not csv_path.is_absolute():
            csv_path = (REPO_ROOT / csv_path).resolve()

        if args.build_list:
            cmd = [
                python,
                str(SCRIPT_DIR / "build_curated_list.py"),
                "--venue",
                v_key,
                "--year",
                args.year,
                "--max-results",
                str(args.build_list_max_results),
                "--llm-workers",
                str(workers),
                "--llm-retries",
                str(args.retries),
                "--out",
                str(csv_path),
            ]
            if args.use_dblp:
                cmd.append("--use-dblp")
            if args.build_list_no_llm:
                cmd.append("--no-llm")
            if args.model:
                cmd += ["--llm-model", args.model]
            run_command(cmd, args.dry_run, log=log, label=f"[{v_key}] 构建候选名单")

        if args.classify:
            if not csv_path.exists() and not args.dry_run:
                print(f"[Warn] 策展 CSV 不存在，跳过分类: {csv_path}")
                continue
            cmd = [
                python,
                str(SCRIPT_DIR / "classify_curated_csv.py"),
                "--csv",
                str(csv_path),
                "--workers",
                str(workers),
                "--retries",
                str(args.retries),
            ]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            if not args.allow_classify_remove:
                cmd.append("--no-remove")
            if args.model:
                cmd += ["--model", args.model]
            run_command(cmd, args.dry_run, log=log, label=f"[{v_key}] LLM 分类复核")

        if not args.skip_generate:
            if not csv_path.exists() and not args.dry_run:
                print(f"[Warn] 策展 CSV 不存在，跳过生成: {csv_path}")
                continue
            cmd = [
                python,
                str(SCRIPT_DIR / "generate_recsys_notes.py"),
                "--papers-csv",
                str(csv_path),
                "--venue",
                v_key,
                "--year",
                args.year,
                "--workers",
                str(workers),
                "--retries",
                str(args.retries),
                "--max-input-chars",
                str(max_input_chars),
                "--max-output-tokens",
                str(max_output_tokens),
                "--max-figures",
                str(max_figures),
            ]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            if args.overwrite:
                cmd.append("--overwrite")
            if args.no_figures:
                cmd.append("--no-figures")
            if not translate_figures:
                cmd.append("--no-figure-translation")
            if args.model:
                cmd += ["--model", args.model]
            if args.thinking:
                cmd.append("--thinking")
            run_command(cmd, args.dry_run, log=log, label=f"[{v_key}] 生成笔记")

    run_command(
        [python, str(SCRIPT_DIR / "build_nav_and_index.py")],
        args.dry_run,
        log=log,
        label="重建导航索引",
    )

    if not args.skip_site:
        run_command(
            [python, "-m", "mkdocs", "build", "--strict"],
            args.dry_run,
            log=log,
            label="构建站点",
        )
        if not args.no_check_links:
            run_command(
                [python, str(SCRIPT_DIR / "check_site_links.py"), "--site", "site"],
                args.dry_run,
                log=log,
                label="检查站点链接",
            )

    total_elapsed = time.monotonic() - pipeline_start
    print(f"\n[Time] 流水线总耗时 {format_duration(total_elapsed)}")
    if log:
        print(f"[Log] 流水线日志: {log.path}")
        log.write(f"TOTAL elapsed={format_duration(total_elapsed)}")
        log.close()
    print("\n[Done] 流水线执行完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())