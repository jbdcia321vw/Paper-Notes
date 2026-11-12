# RecSys Paper Notes · 推荐系统顶会顶刊论文 5 分钟精读库

欢迎访问 **RecSys Paper Notes**！本知识库基于 [zhaoyang97/Paper-Notes](https://github.com/zhaoyang97/Paper-Notes) 架构重构，专注于**推荐系统与信息检索领域**，支持**按顶会/顶刊**与**按推荐系统 14 大细分方向**双维度检索。

- **当前收录推荐系统论文解读**: **1** 篇
- **覆盖核心顶会与顶刊**: **22** 个（SIGIR / RecSys / KDD / WWW / WSDM / CIKM / ACM MM / TOIS / TKDE / TORS 等）
- **覆盖推荐系统细分方向**: **14** 大类（生成式推荐 / 大模型推荐 / 多模态推荐 / 序列推荐 / 协同过滤 / 图推荐 / 社交推荐 / 冷启动与跨域等）

## 一、按推荐系统细分类别快速检索

| 推荐系统细分方向 | 核心研究问题 | 笔记数 |
| :--- | :--- | :---: |
| [**协同过滤与表征学习**](topics/collaborative_filtering.md) | 矩阵分解、双塔召回、对比学习自监督表征、解耦表征与负采样策略 | **0** |
| [**序列与会话推荐**](topics/sequential_rec.md) | 长短期用户行为序列建模、Transformer/Mamba/RNN 架构、会话匿名推荐与 Next-Item 预测 | **1** |
| [**生成式推荐与语义 ID**](topics/generative_rec.md) | RQ-VAE 离散化语义 Token、TIGER 生成式检索范式、扩散模型推荐 (Diffusion Rec) 与自回归生成 | **1** |
| [**大模型与智能体推荐**](topics/llm_agent_rec.md) | 大模型指令微调推荐、In-Context Prompting、知识蒸馏对齐、RecSys Agent 用户行为仿真 | **0** |
| [**多模态推荐与内容理解**](topics/multimodal_rec.md) | 视觉-文本-音频多模态特征融合、多模态图对比学习、短视频/微视频推荐与商品图文匹配 | **0** |
| [**图神经网络与知识图谱推荐**](topics/graph_rec.md) | LightGCN 谱图卷积改进、超图学习 (Hypergraph Rec)、知识图谱推理推荐 (KGRec) 与异构图建模 | **0** |
| [**社交与群组推荐**](topics/social_rec.md) | 社交信任网络传播、社交去噪与同质性建模、群组偏好聚合 (Group Rec) 与意见领袖分析 | **0** |
| [**冷启动与跨域/多场景推荐**](topics/cold_start_cross_domain.md) | 新用户/新物品冷启动、元学习 (Meta-Learning)、跨域迁移学习 (CDR) 与多场景自适应建模 | **0** |
| [**CTR/CVR 预估与多任务精排**](topics/ctr_multi_task_ranking.md) | 高阶特征交叉网络、MMoE/PLE 多目标优化、粗排-精排-重排级联一致性与延迟反馈建模 | **0** |
| [**对话推荐与可解释性**](topics/conversational_explainable.md) | 多轮主动追问对话推荐 (CRS)、推荐理由生成、反事实可解释性与用户意图澄清 | **0** |
| [**强化学习与动态探索推荐**](topics/rl_bandit_rec.md) | Contextual Bandit 探索与利用 (E&E)、Offline RL 长期留存优化、Slate 列表级策略优化 | **0** |
| [**去偏、公平性与可信隐私推荐**](topics/debias_fairness_trust.md) | 曝光/流行度/位置偏差消除、因果推断推荐 (Causal Rec)、群体公平性、联邦推荐与鲁棒防投毒 | **0** |
| [**时空与 POI 本地生活推荐**](topics/poi_spatiotemporal_rec.md) | Next-POI 兴趣点推荐、时空轨迹建模、LBS 本地生活与移动出行推荐 | **0** |
| [**工业界系统架构与评测基准**](topics/industrial_system_eval.md) | 工业级召回/排序全链路实战、GPU 端到端加速、线上 A/B 测试经验、开源数据集与评测新范式 | **0** |

## 二、按核心会议与期刊快速检索

| 类别 | 代表性会议 / 期刊入口 |
| :--- | :--- |
| **IR 与推荐旗舰会议** | [SIGIR](venues/SIGIR2026/index.md) · [RecSys](venues/RecSys2026/index.md) · [WSDM](venues/WSDM2026/index.md) · [CIKM](venues/CIKM2026/index.md) · [ECIR](venues/ECIR2026/index.md) · [SIGIR-AP](venues/SIGIR-AP2026/index.md) |
| **数据挖掘与多媒体顶会** | [KDD](venues/KDD2026/index.md) · [WWW](venues/WWW2026/index.md) · [ACM MM](venues/ACM_MM2026/index.md) · [ICDE](venues/ICDE2026/index.md) · [ICDM](venues/ICDM2026/index.md) · [DASFAA](venues/DASFAA2026/index.md) |
| **综合 AI/NLP/ML 顶会** | [AAAI](venues/AAAI2026/index.md) · [IJCAI](venues/IJCAI2026/index.md) · [ACL/EMNLP](venues/ACL_EMNLP2026/index.md) · [ICLR/NeurIPS/ICML](venues/ICLR_NeurIPS_ICML2026/index.md) |
| **信息系统与数据工程顶刊** | [ACM TOIS](venues/TOIS2026/index.md) · [IEEE TKDE](venues/TKDE2026/index.md) · [ACM TORS](venues/TORS2026/index.md) · [IEEE TMM](venues/TMM2026/index.md) · [ACM TKDD](venues/TKDD2026/index.md) · [ACM TWEB](venues/TWEB2026/index.md) |
