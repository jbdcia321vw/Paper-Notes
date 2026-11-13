# 🔄 自监督对比学习与交互去噪

**ICLR / NeurIPS / ICML (RecSys) 2026** · **3** 篇论文解读

> **方向说明**：图/序列对比数据增广 (SGL/SimGCL)、隐式反馈假阳性噪声过滤 (Denoising) 与鲁棒表征

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/self_supervised_denoising/) · [ECIR 2026 (1)](../../../ECIR/2026/self_supervised_denoising/) · [KDD 2026 (1)](../../../KDD/2026/self_supervised_denoising/) · [SIGIR 2026 (2)](../../../SIGIR/2026/self_supervised_denoising/) · [WSDM 2026 (1)](../../../WSDM/2026/self_supervised_denoising/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/self_supervised_denoising/)

🔥 **高频主题：** ICML ×3 · 召回 ×2

<dl>
<dt><strong><a href="can_recommender_systems_teach_themselves_a_recursive_self_improving_framewo/">Can Recommender Systems Teach Themselves? A Recursive Self-Improving Framework with Fidelity Control</a></strong></dt>
<dd>
<p>提出 RSIR 递归自我改进框架，用模型自生成序列+保真度质量控制稠密化稀疏交互，理论证明其等价于流形切向隐式正则，多骨干多数据集 Full-Ranking 下 NDCG/Recall 稳定提升 3%–14%。</p>
</dd>
<dt><strong><a href="gcib_graph_contrastive_information_bottleneck_for_multi_behavior_recommenda/">GCIB: Graph Contrastive Information Bottleneck for Multi-Behavior Recommendation</a></strong></dt>
<dd>
<p>用图信息瓶颈在结构层剪枝辅助行为噪声边、用跨行为 InfoNCE 在特征层对齐目标语义，Leave-One-Out 全库排序下 NDCG@10 最高提升 37.2%。</p>
</dd>
<dt><strong><a href="rethinking_contrastive_learning_for_graph_collaborative_filtering_limitatio/">Rethinking Contrastive Learning for Graph Collaborative Filtering: Limitations and a Simple Remedy</a></strong></dt>
<dd>
<p>展开 GCF 预测为邻居对加权聚合，揭示 SSM 仅按物品侧相似度更新且忽略邻居对类型，提出类型感知 NT-SSM，NDCG@20 最高提升 21.45%。</p>
</dd>
</dl>
