<p align="center">
  <a href="https://jbdcia321vw.github.io/Paper-Notes/">
    <img src="docs/assets/papernotes-logo-horizontal.svg" alt="RecSys Paper Notes" width="100%">
  </a>
</p>

# 📚 RecSys Paper Notes

**5 分钟读懂一篇推荐系统与信息检索顶会/顶刊论文。**

- 仓库：[jbdcia321vw/Paper-Notes](https://github.com/jbdcia321vw/Paper-Notes)
- 在线阅读：**[https://jbdcia321vw.github.io/Paper-Notes/](https://jbdcia321vw.github.io/Paper-Notes/)**
- 当前收录：**1240 篇**推荐系统与信息检索论文精读笔记
- 覆盖范围：**22 个顶会与顶刊**、**16 个推荐系统正交方向**
- 更新方式：自动发现候选论文 + LLM 辅助生成 + 人工抽查纠错

> 仓库 slug 保持为 `Paper-Notes`，因此 GitHub Pages 访问地址稳定为
> **https://jbdcia321vw.github.io/Paper-Notes/**。

## 🌐 在线阅读

**[https://jbdcia321vw.github.io/Paper-Notes/](https://jbdcia321vw.github.io/Paper-Notes/)**

站点支持按 **会刊 × 年份 × 研究方向** 三维浏览，并提供全站搜索、标签聚合和论文原文/代码直达链接。

> 提示：如果数学公式渲染异常，刷新页面通常可以恢复。

## 📖 项目简介

RecSys Paper Notes 是一个面向推荐系统（Recommender Systems）与信息检索（Information Retrieval）领域的论文精读库。目标不是做论文列表，而是把每篇论文压缩成一篇可在 5 分钟内读完的结构化笔记，帮助读者快速判断：

- 这篇论文解决什么问题，核心增量是什么；
- 方法的关键公式与模块如何工作；
- 实验设置、评估协议和主要结果是否可靠；
- 论文有哪些可迁移亮点与审稿级局限。

每篇笔记通常包含：

- **TL;DR**：一句话总结问题、方法与关键收益；
- **Background & Motivation**：主流范式、前人局限与核心洞察；
- **Method**：整体框架、关键设计、损失函数与训练/推理细节；
- **Experimental Setup**：数据集、划分方式、Ranking 协议与基线；
- **Key Experimental Results**：主实验、消融实验与定量发现；
- **Highlights, Limitations & Rating**：与代表作对比、可迁移点、局限与四维评分。

### 与上游项目的关系

本项目的目录组织、MkDocs 站点结构、图表提取与自动构建思路参考了
[zhaoyang97/Paper-Notes](https://github.com/zhaoyang97/Paper-Notes)。本仓库聚焦推荐系统与信息检索，内容、会刊配置、分类体系与生成脚本均针对 RecSys/IR 场景做了定制，不是上游仓库的简单镜像。

## 📊 收录概览

> 统计截至 2026-11，具体数量以[站点首页](https://jbdcia321vw.github.io/Paper-Notes/)为准。

### 🧭 研究方向

| 方向 | 笔记数 | 关注主题 |
| :--- | ---: | :--- |
| [大语言模型推荐 (LLM4Rec)](https://jbdcia321vw.github.io/RecSys-Paper-Notes/topics/llm_rec/) | **379** | 指令微调、In-Context 重排、语义增强、知识蒸馏 |
| [去偏公平、可信隐私与工业架构](https://jbdcia321vw.github.io/RecSys-Paper-Notes/topics/debias_fairness_industrial/) | **321** | 因果去偏、双边公平、隐私保护、工业全链路 |
| [序列与会话推荐](https://jbdcia321vw.github.io/RecSys-Paper-Notes/topics/sequential_rec/) | **286** | 长短期兴趣、Transformer/Mamba、Next-Item 预测 |
| [生成式推荐与离散语义 ID](https://jbdcia321vw.github.io/RecSys-Paper-Notes/topics/generative_rec/) | **199** | RQ-VAE/VQ、自回归生成式检索、扩散/流匹配推荐 |
| [CTR/CVR 预估与特征交叉](https://jbdcia321vw.github.io/RecSys-Paper-Notes/topics/ctr_feature_interaction/) | **155** | 高阶特征交叉、延迟反馈、概率校准 |
| [协同过滤与双塔召回](https://jbdcia321vw.github.io/RecSys-Paper-Notes/topics/collaborative_filtering/) | **144** | 矩阵分解、双塔检索、ANN 量化、负采样 |
| [冷启动与长尾推荐](https://jbdcia321vw.github.io/RecSys-Paper-Notes/topics/cold_start_rec/) | **136** | Zero-shot 冷启动、元学习、长尾扶持 |
| [多模态推荐与内容理解](https://jbdcia321vw.github.io/RecSys-Paper-Notes/topics/multimodal_rec/) | **129** | 视觉-文本对齐、多模态图、短视频/电商推荐 |
| [图神经网络与知识图谱推荐](https://jbdcia321vw.github.io/RecSys-Paper-Notes/topics/graph_rec/) | **122** | LightGCN、超图、异构图、知识图谱推理 |
| [对话推荐与智能体仿真](https://jbdcia321vw.github.io/RecSys-Paper-Notes/topics/agent_conversational_rec/) | **116** | 多轮对话推荐、LLM Agent、可解释推荐 |
| [多任务学习与列表级重排](https://jbdcia321vw.github.io/RecSys-Paper-Notes/topics/multi_task_reranking/) | **116** | MMoE/PLE、Pareto 最优、列表重排 |
| [自监督对比学习与交互去噪](https://jbdcia321vw.github.io/RecSys-Paper-Notes/topics/self_supervised_denoising/) | **113** | 图/序列对比学习、假阳性噪声过滤 |
| [强化学习与动态探索推荐](https://jbdcia321vw.github.io/RecSys-Paper-Notes/topics/rl_bandit_rec/) | **86** | Contextual Bandit、Offline RL、长期价值 |
| [跨域与多场景推荐](https://jbdcia321vw.github.io/RecSys-Paper-Notes/topics/cross_domain_multi_scenario/) | **72** | 跨域迁移、用户对齐、多业务联合建模 |
| [社交与群组推荐](https://jbdcia321vw.github.io/RecSys-Paper-Notes/topics/social_group_rec/) | **23** | 社交信任、群组偏好聚合、影响力建模 |
| [时空轨迹与 POI 推荐](https://jbdcia321vw.github.io/RecSys-Paper-Notes/topics/poi_spatiotemporal_rec/) | **19** | Next-POI、轨迹建模、本地生活推荐 |

### 🏛️ 顶会与顶刊

| 会刊 | 类型 | 笔记数 | 说明 |
| :--- | :--- | ---: | :--- |
| [SIGIR](https://jbdcia321vw.github.io/RecSys-Paper-Notes/venues/SIGIR/) | 会议 | **202** | IR/RecSys 顶会 |
| [RecSys](https://jbdcia321vw.github.io/RecSys-Paper-Notes/venues/RecSys/) | 会议 | **168** | 推荐系统旗舰专会 |
| [WWW (TheWebConf)](https://jbdcia321vw.github.io/RecSys-Paper-Notes/venues/WWW/) | 会议 | **154** | Web 与推荐系统顶会 |
| [CIKM](https://jbdcia321vw.github.io/RecSys-Paper-Notes/venues/CIKM/) | 会议 | **137** | IR/知识管理顶会 |
| [KDD](https://jbdcia321vw.github.io/RecSys-Paper-Notes/venues/KDD/) | 会议 | **136** | 数据挖掘与工业推荐顶会 |
| [ICLR / NeurIPS / ICML](https://jbdcia321vw.github.io/RecSys-Paper-Notes/venues/ICLR_NeurIPS_ICML/) | 会议 | **79** | 机器学习三大顶会 (RecSys Papers) |
| [ACL / EMNLP](https://jbdcia321vw.github.io/RecSys-Paper-Notes/venues/ACL_EMNLP/) | 会议 | **71** | NLP 与对话推荐顶会 |
| [AAAI (RecSys Track)](https://jbdcia321vw.github.io/RecSys-Paper-Notes/venues/AAAI/) | 会议 | **62** | AI 综合顶会 |
| [WSDM](https://jbdcia321vw.github.io/RecSys-Paper-Notes/venues/WSDM/) | 会议 | **40** | 搜索与数据挖掘顶会 |
| [ACM TOIS](https://jbdcia321vw.github.io/RecSys-Paper-Notes/venues/TOIS/) | 期刊 | **34** | 信息系统与推荐顶刊 |
| [IEEE TKDE](https://jbdcia321vw.github.io/RecSys-Paper-Notes/venues/TKDE/) | 期刊 | **32** | 知识与数据工程顶刊 |
| [ACM TORS](https://jbdcia321vw.github.io/RecSys-Paper-Notes/venues/TORS/) | 期刊 | **25** | ACM 推荐系统旗舰专刊 |
| [ECIR](https://jbdcia321vw.github.io/RecSys-Paper-Notes/venues/ECIR/) | 会议 | **22** | 欧洲信息检索会议 |
| [ICDM](https://jbdcia321vw.github.io/RecSys-Paper-Notes/venues/ICDM/) | 会议 | **21** | 数据挖掘顶会 |
| [ACM MM](https://jbdcia321vw.github.io/RecSys-Paper-Notes/venues/ACM_MM/) | 会议 | **13** | 多媒体推荐重镇 |
| [DASFAA](https://jbdcia321vw.github.io/RecSys-Paper-Notes/venues/DASFAA/) | 会议 | **12** | 数据库与推荐应用会议 |
| [ICDE](https://jbdcia321vw.github.io/RecSys-Paper-Notes/venues/ICDE/) | 会议 | **11** | 数据工程顶会 |
| [IJCAI (RecSys Track)](https://jbdcia321vw.github.io/RecSys-Paper-Notes/venues/IJCAI/) | 会议 | **8** | AI 综合顶会 |
| [SIGIR-AP](https://jbdcia321vw.github.io/RecSys-Paper-Notes/venues/SIGIR-AP/) | 会议 | **6** | SIGIR 亚太会议 |
| [ACM TKDD](https://jbdcia321vw.github.io/RecSys-Paper-Notes/venues/TKDD/) | 期刊 | **4** | ACM 数据挖掘顶刊 |
| [IEEE TMM](https://jbdcia321vw.github.io/RecSys-Paper-Notes/venues/TMM/) | 期刊 | **3** | IEEE 多媒体顶刊 |
| [ACM TWEB](https://jbdcia321vw.github.io/RecSys-Paper-Notes/venues/TWEB/) | 期刊 | **0** | ACM Web 顶刊 |

## 🗂️ 目录结构

```text
docs/
  about.md                 # 关于本站
  index.md                 # 站点首页（自动生成）
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
  topics/                  # 16 大方向跨会刊聚合（自动生成）
  years/                   # 按年份聚合（自动生成）
  assets/papers/           # 从 PDF 提取的论文图表（按 arXiv ID 归档）
data/
  papers/
    SIGIR2026.csv          # 策展论文名单
    RecSys2026.csv
    ...
  manifest.jsonl           # 已生成笔记清单（自动维护）
scripts/
  build_curated_list.py    # 自动发现候选论文
  classify_curated_csv.py  # LLM 相关性复核与分类
  paper_classifier.py      # 分类器与缓存
  generate_recsys_notes.py # 生成论文精读笔记
  extract_pdf_figures.py   # 从 PDF 提取图表
  build_nav_and_index.py   # 重建首页、分类/年份/会刊索引与 mkdocs.yml
  run_pipeline.py          # 一键串联生成/索引/构建
  run_logging.py           # 日志、进度条、耗时与费用格式化
  check_site_links.py      # 检查 site/ 本地链接
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

# 自动发现候选论文（可加 --use-dblp 同时查询 OpenAlex / DBLP）
python scripts/build_curated_list.py --venue SIGIR --year 2026 --max-results 300 --use-dblp

# 对已有候选名单做 LLM 复核：默认只修正分类，不删除条目
python scripts/classify_curated_csv.py --all --workers 4 --no-remove

# 生成笔记（生产推荐：策展 CSV）
python scripts/generate_recsys_notes.py --papers-csv data/papers/SIGIR2026.csv --limit 5 --workers 2

# 仅重建导航与索引
python scripts/build_nav_and_index.py

# 本地预览
mkdocs serve
```

## 🔄 自动化流水线

`scripts/run_pipeline.py` 将以下步骤串联起来：

1. `build_curated_list.py`：从 arXiv / OpenAlex / DBLP 自动增量更新候选 CSV；
2. `classify_curated_csv.py`：用 LLM 复核推荐相关性与主分类；
3. `generate_recsys_notes.py`：下载全文、提取图表、生成笔记；
4. `build_nav_and_index.py`：重建首页、分类/年份/会刊索引与 `mkdocs.yml`；
5. `mkdocs build --strict` 与本地链接检查。

`.github/workflows/init_and_deploy.yml` 负责构建并部署 MkDocs 站点。普通 push 只重建索引并部署，不会自动抓取论文或消耗 token；手动触发时可通过 `papers_csv` + `profile` 调用生成流程。

## 📝 策展名单 CSV

文件：`data/papers/<VENUE><YEAR>.csv`

```csv
title,arxiv_id,paper_url,code_url,category,venue,year
示例论文,2401.01234,https://arxiv.org/abs/2401.01234,https://github.com/org/repo,generative_rec,SIGIR,2026
```

- `arxiv_id` 必填（或从 `paper_url` 自动解析）；
- `category` 必须是 `scripts/init_clean_recsys_repo.py` 中定义的 16 个 slug 之一；
- `venue` 使用会刊 key（如 `SIGIR`、`RecSys`），`year` 为四位年份；
- 脚本会下载 arXiv PDF 全文；**仅摘要模式默认拒绝**，除非显式传入 `--allow-abstract`；
- 最终名单仍建议人工抽查，确认会刊、年份、分类与 workshop / challenge 排除规则。

## ✅ 生成质量约束

- 默认模型：DeepSeek `deepseek-flash`，可通过 `LLM_MODEL` / `DEEPSEEK_MODEL` 覆盖；
- API Key：优先 `LLM_API_KEY`，其次 `DEEPSEEK_API_KEY`；也可用 `--api-key` 显式传入；
- 自动校验：YAML Frontmatter、`venue`/`year` 一致性、合法分类、必需章节、模板占位符；
- 失败自动重试（`--retries`）；
- 已存在笔记默认跳过，`--overwrite` 强制覆盖；
- 候选名单默认用 LLM 复核相关性与分类；`build_curated_list.py --no-llm` 可回退到关键词启发式；
- LLM 分类结果缓存在 `data/papers/.classification_cache.json`，`--refresh` 可强制刷新；
- `--seed-all-categories` 与 `--arxiv-query` 仅供本地试点，生产收录请使用策展 CSV。

> 注意：笔记由 LLM 辅助生成，可能存在事实错误、实验数字偏差或引用遗漏。请以论文原文与官方代码为准。

## 💰 Token 成本优化

`run_pipeline.py` 提供三档预设：

| Profile | 并发 | 输入上限 | 输出上限 | 图表 | 图注翻译 | 适用场景 |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| `fast` | 4 | 48k 字符 | 8k token | 6 张 | 否（中文编号兜底） | 大批量初筛 |
| `balanced` | 3 | 65k 字符 | 16k token | 8 张 | 是 | 默认生产 |
| `quality` | 2 | 80k 字符 | 16k token | 10 张 | 是 | 重点论文 |

已内置的成本控制：

- **关闭 thinking**：DeepSeek 默认开启 thinking，会额外产生 reasoning token；脚本默认显式关闭，只有 `--thinking` 才开启；
- **压缩输入**：自动去掉参考文献/致谢并限制 `--max-input-chars`，避免把整篇 PDF 全部送入模型；
- **限制输出**：`--max-output-tokens` 防止模型生成过长笔记；
- **图注翻译可选**：`fast` 默认使用“图 n/表 n”中文编号兜底，省掉每篇一次额外翻译调用；
- **非高峰运行**：DeepSeek 非高峰价为高峰的 5 折；`--off-peak-wait` 会自动等待；
- **自动缓存**：DeepSeek Context Caching 默认开启，稳定的 system prompt 前缀可命中缓存；
- **断点续传**：已存在笔记默认跳过，避免重复生成。

> 高峰时段（UTC）：周一至周五 01:00–04:00、06:00–10:00；中国法定节假日按官方规则视为非高峰。

## 🖼️ 论文图表

生成笔记时默认会从 PDF 提取 `Figure` / `Table` 区域，保存到 `docs/assets/papers/<arxiv_id>/`，并把图表清单交给 LLM。LLM 只会在真正适合辅助理解的段落后插入 `[[FIG:n]]` 占位符；生成完成后脚本把占位符替换成相对路径的 `<figure>`，未使用的图片会自动清理，不会堆砌到文末。

```powershell
# 默认提取，每篇最多 8 张，200 DPI
python scripts/generate_recsys_notes.py --papers-csv data/papers/SIGIR2026.csv --limit 5 --workers 2

# 调整图表数量 / 分辨率
python scripts/generate_recsys_notes.py --papers-csv data/papers/SIGIR2026.csv --max-figures 12 --figure-dpi 220

# 关闭图表提取
python scripts/generate_recsys_notes.py --papers-csv data/papers/SIGIR2026.csv --no-figures

# 不调用 LLM 翻译图注，使用“图 n/表 n”中文编号兜底（省 token）
python scripts/generate_recsys_notes.py --papers-csv data/papers/SIGIR2026.csv --no-figure-translation
```

- 依赖 `pymupdf`，已包含在 `requirements.txt`；
- 图片采用相对路径，笔记移动到其他分类目录时仍能正常显示；
- DeepSeek 文本模型看不到图片内容，只能根据图注与上下文决定插图位置。

## 💬 反馈与纠错

欢迎反馈站点使用、内容准确性、论文收录与下架等问题：

- 提交 [GitHub Issue](https://github.com/jbdcia321vw/Paper-Notes/issues)
- 如果是论文作者并希望调整或下架相关内容，请在 Issue 中附上论文链接与具体页面。

更多项目说明见站点 [关于本站](https://jbdcia321vw.github.io/Paper-Notes/about/)。

## 📄 License

本项目沿用上游项目的 **CC BY-NC-SA 4.0** 许可：

- 署名：使用时请注明来源；
- 非商业性使用：不得用于商业目的；
- 相同方式共享：修改后须以相同协议分享。

上游参考：[zhaoyang97/Paper-Notes](https://github.com/zhaoyang97/Paper-Notes)。
