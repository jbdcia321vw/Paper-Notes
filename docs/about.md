---
description: 关于 RecSys Paper Notes：推荐系统与信息检索顶会/顶刊论文精读库，在线阅读地址与项目说明。
hide:
  - toc
---

# ℹ️ 关于本站

**RecSys Paper Notes** 是一个面向推荐系统（Recommender Systems）与信息检索（Information Retrieval）领域的论文精读库。

- **在线阅读**：[https://jbdcia321vw.github.io/Paper-Notes/](https://jbdcia321vw.github.io/Paper-Notes/)
- **GitHub 仓库**：[jbdcia321vw/Paper-Notes](https://github.com/jbdcia321vw/Paper-Notes)
- **仓库显示名**：RecSys Paper Notes
- **仓库 slug**：`Paper-Notes`（保持不变，以保证 GitHub Pages 地址稳定）

## 本站做什么

每篇论文会被整理成一篇可在 5 分钟内读完的结构化笔记，重点回答四个问题：

1. 论文解决什么问题，核心增量是什么；
2. 方法的关键公式与模块如何工作；
3. 实验设置、评估协议和主要结果是否可靠；
4. 论文有哪些可迁移亮点与审稿级局限。

笔记通常包含：

- **TL;DR**：问题、方法与关键收益；
- **Background & Motivation**：主流范式、前人局限与核心洞察；
- **Method**：整体框架、关键设计、损失函数与训练/推理细节；
- **Experimental Setup**：数据集、划分方式、Ranking 协议与基线；
- **Key Experimental Results**：主实验、消融实验与定量发现；
- **Highlights, Limitations & Rating**：代表作对比、可迁移点、局限与四维评分。

## 收录范围

- **22 个顶会与顶刊**：SIGIR、RecSys、KDD、WWW、CIKM、WSDM、ACM MM、ECIR、SIGIR-AP、ICDE、ICDM、DASFAA、AAAI、IJCAI、ACL/EMNLP、ICLR/NeurIPS/ICML、TOIS、TKDE、TORS、TMM、TKDD、TWEB。
- **16 个推荐系统正交方向**：协同过滤、图推荐、序列推荐、生成式推荐、LLM4Rec、对话推荐、多模态推荐、冷启动、跨域推荐、CTR/CVR、多任务重排、强化学习、去偏公平、自监督去噪、社交群组、POI 时空推荐。
- **按会刊 × 年份 × 方向组织**：支持从会刊、年份和主题三种入口浏览。

当前已收录 300+ 篇 2026 年论文笔记，具体数量以[首页](https://jbdcia321vw.github.io/Paper-Notes/)统计为准。

## 内容如何生成

本站使用自动化流水线辅助生产内容：

1. 从 arXiv、OpenAlex、DBLP 等公开来源发现候选论文；
2. 用 LLM 复核推荐相关性与主分类；
3. 下载论文全文、提取图表、生成结构化笔记；
4. 自动校验 Frontmatter、会刊/年份与必需章节；
5. 重建导航索引并部署到 GitHub Pages。

自动化只负责提高效率，**不替代人工判断**。笔记可能存在事实错误、实验数字偏差、引用遗漏或分类不准确，请以论文原文、官方代码和同行评议结论为准。

## 与上游项目的关系

本项目的目录组织、MkDocs 站点结构、图表提取与自动构建思路参考了
[zhaoyang97/Paper-Notes](https://github.com/zhaoyang97/Paper-Notes)。

本仓库聚焦推荐系统与信息检索，会刊配置、16 大分类体系、候选论文发现与生成脚本均针对 RecSys/IR 场景做了定制，不是上游仓库的简单镜像。

## 反馈与纠错

欢迎反馈站点使用、内容准确性、论文收录与下架等问题：

- 提交 [GitHub Issue](https://github.com/jbdcia321vw/Paper-Notes/issues)
- 如果是论文作者并希望调整或下架相关内容，请在 Issue 中附上论文链接与具体页面。

## License

本项目沿用上游项目的 **CC BY-NC-SA 4.0** 许可：

- 署名：使用时请注明来源；
- 非商业性使用：不得用于商业目的；
- 相同方式共享：修改后须以相同协议分享。

上游参考：[zhaoyang97/Paper-Notes](https://github.com/zhaoyang97/Paper-Notes)。
