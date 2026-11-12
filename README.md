# 📚 RecSys Paper Notes

推荐系统与信息检索顶会/顶刊论文精读库。按 **会刊 × 年份 × 研究方向** 三维组织，每篇笔记包含核心思想、方法拆解、实验设置与评估协议、关键结果和审稿级批判。

- 🏛️ 覆盖 SIGIR、RecSys、KDD、WWW、ACM MM、TOIS、TKDE 等 22 个核心会刊
- 🔬 16 个推荐系统正交细分方向
- 🗓️ 按年份归档，支持后续扩展多年份
- 🔗 每篇笔记附论文原文与开源代码直达链接
- 🧪 仅收录可获取全文的论文，生成后自动校验 Frontmatter 与必需章节

## 🌐 在线阅读

部署后访问：`https://<your-github-user>.github.io/Paper-Notes/`

## 🗂️ 目录结构

```text
docs/
  venues/
    SIGIR/
      2026/
        generative_rec/
          <paper_slug>.md
        llm_rec/
        ...
    RecSys/
      2026/
        ...
  assets/papers/   # 从 PDF 提取的论文图表（按 arXiv ID 归档）
  topics/          # 按 16 大方向跨会刊聚合（自动生成）
  years/           # 按年份聚合（自动生成）
data/
  papers/
    SIGIR2026.csv  # 策展论文名单
    RecSys2026.csv
  manifest.jsonl   # 已生成笔记清单（自动维护）
scripts/
  init_clean_recsys_repo.py
  build_curated_list.py
  classify_curated_csv.py
  paper_classifier.py
  generate_recsys_notes.py
  extract_pdf_figures.py
  build_nav_and_index.py
  run_pipeline.py       # 一键串联生成/索引/构建
  run_logging.py        # 日志、进度条、耗时与费用格式化
  check_site_links.py   # 检查 site/ 本地链接
```

## 🚀 本地运行

建议使用 Python 3.11+（已在 Python 3.14 验证）。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 初始化目录骨架（会迁移旧结构并删除示例笔记）
python scripts/init_clean_recsys_repo.py

# 配置 DeepSeek API Key（也兼容 LLM_API_KEY）
$env:DEEPSEEK_API_KEY = "sk-..."

# 推荐：一键批处理（默认 balanced，自动构建导航与站点并检查链接）
python scripts/run_pipeline.py --profile balanced

# 小规模试点：只处理前 5 篇，只看将执行的命令
python scripts/run_pipeline.py --profile fast --limit 5 --dry-run

# 高峰时段自动等到非高峰再跑（DeepSeek 非高峰价为 5 折）
python scripts/run_pipeline.py --profile balanced --off-peak-wait

# 可选：从 arXiv 主会录用信号生成候选名单（默认用 LLM 判断相关性与分类）
python scripts/build_curated_list.py --venue SIGIR --year 2026 --max-results 300
python scripts/build_curated_list.py --venue RecSys --year 2026 --max-results 200

# 对已有候选名单做 LLM 复核：默认只修正分类，不删除条目
python scripts/classify_curated_csv.py --all --workers 4 --no-remove

# 生成笔记（生产推荐：策展 CSV）
# 小规模试点：只处理名单前 5 篇
python scripts/generate_recsys_notes.py --papers-csv data/papers/SIGIR2026.csv --limit 5 --workers 2

# 仅重建导航与索引
python scripts/build_nav_and_index.py

# 本地预览
mkdocs serve
```

## 📝 策展名单 CSV

文件：`data/papers/<VENUE><YEAR>.csv`

> `SIGIR2026.csv` / `RecSys2026.csv` 由 `scripts/build_curated_list.py` 从 arXiv 主会录用信号生成，并可用 `scripts/classify_curated_csv.py` 调用 LLM 根据标题与摘要复核相关性和主分类。最终名单仍建议人工抽查，确认会刊、年份、分类与 workshop / challenge 排除规则。

```csv
title,arxiv_id,paper_url,code_url,category,venue,year
示例论文,2401.01234,https://arxiv.org/abs/2401.01234,https://github.com/org/repo,generative_rec,SIGIR,2026
```

- `arxiv_id` 必填（或从 `paper_url` 自动解析）
- `category` 必须是 `scripts/init_clean_recsys_repo.py` 中定义的 16 个 slug 之一
- `venue` 使用会刊 key（如 `SIGIR`、`RecSys`），`year` 为四位年份
- 脚本会下载 arXiv PDF 全文；**仅摘要模式默认拒绝**，除非显式传入 `--allow-abstract`

## ✅ 生成质量约束

- 默认模型：DeepSeek `deepseek-flash`，可通过 `LLM_MODEL` / `DEEPSEEK_MODEL` 覆盖
- API Key：优先 `LLM_API_KEY`，其次 `DEEPSEEK_API_KEY`；也可用 `--api-key` 显式传入
- 自动校验：YAML Frontmatter、`venue`/`year` 一致性、合法分类、必需章节、模板占位符
- 失败自动重试（`--retries`）
- 已存在笔记默认跳过，`--overwrite` 强制覆盖
- 候选名单默认用 LLM 复核相关性与分类；`build_curated_list.py --no-llm` 可回退到关键词启发式
- LLM 分类结果缓存在 `data/papers/.classification_cache.json`，`--refresh` 可强制刷新
- `--seed-all-categories` 与 `--arxiv-query` 仅供本地试点，生产收录请使用策展 CSV

## 💰 Token 成本优化

`run_pipeline.py` 提供三档预设：

| Profile | 输入上限 | 输出上限 | 图表 | 图注翻译 | 适用场景 |
| :--- | ---: | ---: | ---: | :--- | :--- |
| `fast` | 28k 字符 | 5k token | 6 张 | 否（中文编号兜底） | 大批量初筛 |
| `balanced` | 36k 字符 | 7k token | 8 张 | 否（中文编号兜底） | 默认生产 |
| `quality` | 52k 字符 | 10k token | 10 张 | 是 | 重点论文 |

已内置的成本控制：

- **关闭 thinking**：DeepSeek 默认开启 thinking，会额外产生 reasoning token；脚本默认显式关闭，只有 `--thinking` 才开启。
- **压缩输入**：自动去掉参考文献/致谢并限制 `--max-input-chars`，避免把整篇 PDF 全部送入模型。
- **限制输出**：`--max-output-tokens` 防止模型生成过长笔记。
- **图注翻译可选**：`run_pipeline.py` 的 `fast`/`balanced` 默认使用“图 n/表 n”中文编号兜底，省掉每篇一次额外翻译调用；`quality` 或 `--translate-figures` 才调用 LLM 翻译。
- **非高峰运行**：DeepSeek 非高峰价为高峰的 5 折；`--off-peak-wait` 会自动等待。
- **自动缓存**：DeepSeek Context Caching 默认开启，稳定的 system prompt 前缀可命中缓存；运行结束会打印 token、缓存命中率与近似成本。
- **断点续传**：已存在笔记默认跳过，避免重复生成。

> 高峰时段（UTC）：周一至周五 01:00–04:00、06:00–10:00；中国法定节假日按官方规则视为非高峰。

## 📊 运行日志、进度与费用报告

生成笔记、分类复核和 `run_pipeline.py` 运行时都会输出实时进度：

- 控制台进度条显示完成数、成功/跳过/失败数、已用时间、ETA 和当前累计花费。
- 每篇论文的状态、重试原因和错误会写入 `logs/` 下的日志文件，便于中断后排查。
- 运行结束会报告总 token、缓存命中率、总花费（USD 与近似 CNY）、平均每篇花费，以及总耗时和平均每篇耗时。
- `logs/` 已加入 `.gitignore`，不会提交到仓库。

## 🖼️ 论文图表

生成笔记时默认会从 PDF 提取 `Figure` / `Table` 区域，保存到 `docs/assets/papers/<arxiv_id>/`，并把图表清单交给 LLM，让它在方法或实验结果附近插入 `[[FIG:n]]` / `[[TABLE:n]]` 占位符。生成完成后脚本会把占位符替换成相对路径的 `<figure>`，未使用的图表会追加到文末“论文图表”小节。

```powershell
# 默认提取，每篇最多 8 张，200 DPI
python scripts/generate_recsys_notes.py --papers-csv data/papers/SIGIR2026.csv --limit 5 --workers 2

# 调整图表数量 / 分辨率
python scripts/generate_recsys_notes.py --papers-csv data/papers/SIGIR2026.csv --max-figures 12 --figure-dpi 220

# 关闭图表提取
python scripts/generate_recsys_notes.py --papers-csv data/papers/SIGIR2026.csv --no-figures

# 不调用 LLM 翻译图注，使用“图 n/表 n”中文编号兜底（省 token）
python scripts/generate_recsys_notes.py --papers-csv data/papers/SIGIR2026.csv --no-figure-translation

# 单独提取某个 PDF 的图表
python scripts/extract_pdf_figures.py --pdf .cache_pdfs/2605.29322v1.pdf --out docs/assets/papers/2605.29322v1
```

- 依赖 `pymupdf`，已包含在 `requirements.txt`
- 图片采用相对路径，笔记移动到其他分类目录时仍能正常显示
- DeepSeek 文本模型看不到图片内容，只能根据图注与上下文决定插图位置

## 🔄 自动构建

`.github/workflows/init_and_deploy.yml` 负责构建并部署 MkDocs 站点。手动触发时可通过 `papers_csv` + `profile` 调用 `run_pipeline.py` 生成笔记；普通 push 只重建索引并部署，不会自动抓取论文或消耗 token。
