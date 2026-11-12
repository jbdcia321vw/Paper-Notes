#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_pdf_figures.py
从论文 PDF 中提取 Figure/Table 区域为 PNG，并尽量抓取对应图注。

推荐用法：
  python scripts/extract_pdf_figures.py --pdf .cache_pdfs/2605.29322v1.pdf \
      --out docs/assets/papers/2605.29322v1

生成器会调用 extract_figures()，把图表清单交给 LLM，并替换正文中的
[[FIG:1]] / [[TABLE:1]] 占位符。
"""

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import pymupdf
except ImportError:  # pragma: no cover - 运行时给出明确提示
    pymupdf = None

CAPTION_RE = re.compile(
    r"^\s*(Figure|Fig\.?|Table)\s+([0-9]+[A-Za-z]?)\s*[:.]",
    re.IGNORECASE,
)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _union(rects):
    return pymupdf.Rect(
        min(rect.x0 for rect in rects),
        min(rect.y0 for rect in rects),
        max(rect.x1 for rect in rects),
        max(rect.y1 for rect in rects),
    )


def _x_overlap(a, b) -> float:
    return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))


def _same_column(a, b) -> bool:
    return _x_overlap(a, b) > max(12.0, 0.15 * min(a.width, b.width))


def _vertical_gap(a, b) -> float:
    if a.y1 < b.y0:
        return b.y0 - a.y1
    if b.y1 < a.y0:
        return a.y0 - b.y1
    return 0.0


def _caption_kind(label: str) -> str:
    return "table" if label.lower().startswith("table") else "figure"


def _find_captions(page) -> list:
    captions = []
    for block in page.get_text("blocks"):
        if len(block) < 5:
            continue
        text = _clean_text(block[4])
        match = CAPTION_RE.match(text)
        if not match:
            continue
        rect = pymupdf.Rect(block[:4])
        if rect.width < 40 or rect.height < 8:
            continue
        kind = _caption_kind(match.group(1))
        number = match.group(2)
        label = f"{'Table' if kind == 'table' else 'Figure'} {number}"
        captions.append(
            {
                "kind": kind,
                "number": number,
                "label": label,
                "caption": text,
                "rect": rect,
            }
        )
    return captions


def _assign_images(images: list, captions: list) -> dict:
    assigned = {id(cap): [] for cap in captions}

    for image in images:
        best = None
        best_score = float("inf")
        for cap in captions:
            if not _same_column(image["rect"], cap["rect"]):
                continue
            above = image["rect"].y1 <= cap["rect"].y0 + 8
            below = image["rect"].y0 >= cap["rect"].y1 - 8
            if not (above or below):
                continue
            gap = _vertical_gap(image["rect"], cap["rect"])
            if gap > 280:
                continue
            preferred = above if cap["kind"] == "figure" else below
            score = gap + (0.0 if preferred else 120.0)
            if score < best_score:
                best = cap
                best_score = score
        if best is not None:
            assigned[id(best)].append(image["rect"])

    return assigned


def _cluster_rects(rects: list, max_gap: float = 30.0) -> list:
    ordered = sorted(rects, key=lambda r: (r.y0, r.y1))
    clusters = []
    current = []
    current_bottom = 0.0
    for rect in ordered:
        if not current:
            current = [rect]
            current_bottom = rect.y1
            continue
        if rect.y0 - current_bottom <= max_gap:
            current.append(rect)
            current_bottom = max(current_bottom, rect.y1)
        else:
            clusters.append(current)
            current = [rect]
            current_bottom = rect.y1
    if current:
        clusters.append(current)
    return clusters


def _drawing_region(page, cap: dict, drawings: list):
    prefer_above = cap["kind"] == "figure"
    candidates = []
    for rect in drawings:
        if not _same_column(rect, cap["rect"]):
            continue
        gap = _vertical_gap(rect, cap["rect"])
        if gap > 420:
            continue
        if rect.y1 <= cap["rect"].y0 + 8 or rect.y0 >= cap["rect"].y1 - 8:
            candidates.append(rect)

    if not candidates:
        return None

    if cap["kind"] == "table":
        wide = [r for r in candidates if r.width >= 40]
        if wide:
            candidates = wide

    above = [r for r in candidates if r.y1 <= cap["rect"].y0 + 8]
    below = [r for r in candidates if r.y0 >= cap["rect"].y1 - 8]
    max_gap = 60.0 if cap["kind"] == "table" else 30.0

    def region_for(group, side_above: bool):
        clusters = _cluster_rects(group, max_gap=max_gap)
        if not clusters:
            return None
        if side_above:
            cluster = max(clusters, key=lambda c: max(r.y1 for r in c))
        else:
            cluster = min(clusters, key=lambda c: min(r.y0 for r in c))
        region = _union(cluster)
        if region.width < 60 or region.height < 20:
            return None
        return region

    preferred = region_for(above if prefer_above else below, prefer_above)
    if preferred is not None:
        return preferred
    return region_for(below if prefer_above else above, not prefer_above)


def _render(page, rect, out_path: Path, dpi: int):
    pix = page.get_pixmap(clip=rect, dpi=dpi, alpha=False)
    pix.save(str(out_path))


def extract_figures(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 200,
    max_figures: int = 12,
) -> list:
    """提取 PDF 中的 Figure/Table 区域，返回带 token、caption、filename 的列表。"""
    if pymupdf is None:
        raise RuntimeError("缺少 pymupdf，请先运行: pip install pymupdf")

    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(pdf_path))
    figures = []
    seen_labels = set()
    used_names = set()

    try:
        for page_no, page in enumerate(doc, 1):
            captions = _find_captions(page)
            if not captions:
                continue

            images = []
            for info in page.get_image_info(xrefs=True):
                bbox = info.get("bbox")
                if not bbox:
                    continue
                rect = pymupdf.Rect(bbox)
                if rect.width < 40 or rect.height < 8:
                    continue
                images.append({"rect": rect, "xref": info.get("xref")})

            drawings = []
            for item in page.get_drawings():
                rect = item.get("rect")
                if rect is None:
                    continue
                rect = pymupdf.Rect(rect)
                if rect.width < 3 and rect.height < 3:
                    continue
                drawings.append(rect)

            assigned = _assign_images(images, captions)

            for cap in captions:
                key = (cap["kind"], cap["number"])
                if key in seen_labels:
                    continue

                image_rects = assigned.get(id(cap), [])
                image_region = _union(image_rects) if image_rects else None
                drawing_region = _drawing_region(page, cap, drawings)

                region = None
                source = ""
                if image_region is not None and image_region.width >= 60 and image_region.height >= 30:
                    region = image_region
                    source = "image"
                elif drawing_region is not None:
                    region = drawing_region
                    source = "vector"
                elif image_region is not None:
                    region = image_region
                    source = "image"

                if region is None:
                    continue

                region = pymupdf.Rect(region)
                region.x0 -= 2
                region.y0 -= 2
                region.x1 += 2
                region.y1 += 2
                region &= page.rect

                if region.width < 60 or region.height < 30:
                    continue
                page_area = page.rect.width * page.rect.height
                if region.width * region.height > 0.95 * page_area:
                    continue

                seen_labels.add(key)
                prefix = "table" if cap["kind"] == "table" else "figure"
                stem = f"{prefix}_{cap['number']}"
                filename = f"{stem}.png"
                token_suffix = cap["number"]
                if filename in used_names:
                    suffix = 2
                    while f"{stem}_{suffix}.png" in used_names:
                        suffix += 1
                    filename = f"{stem}_{suffix}.png"
                    token_suffix = f"{cap['number']}_{suffix}"
                used_names.add(filename)

                out_path = out_dir / filename
                _render(page, region, out_path, dpi)

                token_prefix = "TABLE" if cap["kind"] == "table" else "FIG"
                figures.append(
                    {
                        "id": Path(filename).stem,
                        "token": f"[[{token_prefix}:{token_suffix}]]",
                        "kind": cap["kind"],
                        "label": cap["label"],
                        "caption": cap["caption"][:600],
                        "page": page_no,
                        "filename": filename,
                        "source": source,
                        "width": int(region.width),
                        "height": int(region.height),
                    }
                )

                if len(figures) >= max_figures:
                    return figures
    finally:
        doc.close()

    return figures


def main():
    parser = argparse.ArgumentParser(description="提取论文 PDF 中的 Figure/Table 为 PNG")
    parser.add_argument("--pdf", required=True, help="PDF 文件路径")
    parser.add_argument("--out", required=True, help="图片输出目录")
    parser.add_argument("--dpi", type=int, default=200, help="渲染分辨率，默认 200")
    parser.add_argument("--max-figures", type=int, default=12, help="最多提取多少张，默认 12")
    parser.add_argument("--manifest", default="", help="可选：将图表元数据写入 JSON 文件")
    args = parser.parse_args()

    figures = extract_figures(
        Path(args.pdf),
        Path(args.out),
        dpi=args.dpi,
        max_figures=args.max_figures,
    )
    payload = json.dumps(figures, ensure_ascii=False, indent=2)
    if args.manifest:
        Path(args.manifest).write_text(payload, encoding="utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    print(payload)


if __name__ == "__main__":
    main()
