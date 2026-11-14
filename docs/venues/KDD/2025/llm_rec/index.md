# 🧠 大语言模型推荐 (LLM4Rec)

**KDD 2025** · **10** 篇论文解读

> **方向说明**：LLM 指令微调推荐、In-Context 列表重排、语义表征增强、大模型向轻量推荐模型知识蒸馏

📌 **同领域跨会议浏览：** [KDD 2026 (7)](../../2026/llm_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/llm_rec/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (19)](../../../ACL_EMNLP/2025/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (10)](../../../ACL_EMNLP/2026/llm_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/llm_rec/) · [CIKM 2026 (9)](../../../CIKM/2026/llm_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/llm_rec/) · [ECIR 2025 (3)](../../../ECIR/2025/llm_rec/) · [ECIR 2026 (2)](../../../ECIR/2026/llm_rec/) · [ICDE 2025 (3)](../../../ICDE/2025/llm_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/llm_rec/) · [ICDM 2025 (2)](../../../ICDM/2025/llm_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (10)](../../../ICLR_NeurIPS_ICML/2025/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/llm_rec/) · [RecSys 2025 (4)](../../../RecSys/2025/llm_rec/) · [RecSys 2026 (11)](../../../RecSys/2026/llm_rec/) · [SIGIR 2025 (14)](../../../SIGIR/2025/llm_rec/) · [SIGIR 2026 (22)](../../../SIGIR/2026/llm_rec/) · [IEEE TKDE 2025 (4)](../../../TKDE/2025/llm_rec/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/llm_rec/) · [ACM TOIS 2025 (9)](../../../TOIS/2025/llm_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/llm_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/llm_rec/) · [WSDM 2025 (1)](../../../WSDM/2025/llm_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/llm_rec/) · [WWW (TheWebConf) 2025 (19)](../../../WWW/2025/llm_rec/) · [WWW (TheWebConf) 2026 (8)](../../../WWW/2026/llm_rec/)

🔥 **高频主题：** 大语言模型推荐 ×10 · 召回 ×3 · 生成式端到端 ×2

<dl>
<dt><strong><a href="agent0_leveraging_llm_agents_to_discover_multi_value_features_from_text_for/">Agent0: Leveraging LLM Agents to Discover Multi-value Features from Text for Enhanced Recommendations</a></strong></dt>
<dd>
<p>用 Architect-Sentinel-Oracle 三角色 LLM 智能体闭环自动挖掘文本多值类别特征，以 DCNv2+RIG 为 Oracle 在真实广告生产数据上取得 0.005+ RIG 提升并提速 2-3 倍。</p>
</dd>
<dt><strong><a href="collaboration_of_large_language_models_and_small_recommendation_models_for_/">Collaboration of Large Language Models and Small Recommendation Models for Device-Cloud Recommendation</a></strong></dt>
<dd>
<p>提出端云协同 LSC4Rec：云端 LLM 召回+设备端 SRM 实时重排，配合协同训练、分数融合与不一致度触发请求，Beauty 平均提升 16.18%。</p>
</dd>
<dt><strong><a href="earn_efficient_inference_acceleration_for_llm_based_generative_recommendati/">EARN: Efficient Inference Acceleration for LLM-based Generative Recommendation by Register Tokens</a></strong></dt>
<dd>
<p>基于 LLMRec 注意力层间稀疏反转与双注意力汇现象，用首尾寄存器 token 压缩历史交互并在第 k 层后剪除 prompt，实现 3.79× 加速与 80.8% KV Cache 缩减且精度反超微调。</p>
</dd>
<dt><strong><a href="exploring_feature_based_knowledge_distillation_for_recommender_system_a_fre/">Exploring Feature-based Knowledge Distillation for Recommender System: A Frequency Perspective</a></strong></dt>
<dd>
<p>从频域视角揭示特征蒸馏等权缺陷，提出轻量图滤波重加权 FreqD，全库排序下 Recall@20 最高提升 18.2%。</p>
</dd>
<dt><strong><a href="goracs_group_level_optimal_transport_guided_coreset_selection_for_llm_based/">GORACS: Group-level Optimal Transport-guided Coreset Selection for LLM-based Recommender Systems</a></strong></dt>
<dd>
<p>用最优传输+梯度范数构造测试损失上界代理目标，配合贪心初始化-交换精炼两阶段算法做组级核心集选择，20%训练开销超越全量微调。</p>
</dd>
<dt><strong><a href="lettingo_explore_user_profile_generation_for_recommendation_system/">LettinGo: Explore User Profile Generation for Recommendation System</a></strong></dt>
<dd>
<p>用多LLM探索开放式用户画像、以下游推荐成败构造偏好对、DPO对齐画像生成器，Amazon Books Acc 66.3%→70.4%。</p>
</dd>
<dt><strong><a href="llm2rec_large_language_models_are_powerful_embedding_models_for_sequential_/">LLM2Rec: Large Language Models Are Powerful Embedding Models for Sequential Recommendation</a></strong></dt>
<dd>
<p>两阶段将 LLM 改造为融合协同信号与语义的物品嵌入模型，Full-Ranking 下域内平均提升约 15%、跨域仍稳定增益。</p>
</dd>
<dt><strong><a href="lost_in_sequence_do_large_language_models_understand_sequential_recommendat/">Lost in Sequence: Do Large Language Models Understand Sequential Recommendation?</a></strong></dt>
<dd>
<p>揭示现有 LLM4Rec 模型无法捕捉交互序列顺序信息，提出冻结 LLM 与 CF-SRec、仅训练轻量 MLP 的蒸馏框架 LLM-SRec，在 Amazon 四数据集上 NDCG@10 全面超越 SOTA。</p>
</dd>
<dt><strong><a href="the_mental_world_of_large_language_models_in_recommendation_a_benchmark_on_/">The Mental World of Large Language Models in Recommendation: A Benchmark on Association, Personalization, and Knowledgeability</a></strong></dt>
<dd>
<p>提出 LRWorld 基准（38K 样本/23M tokens），从关联、个性化、知识性三尺度十因素 31 任务系统评测数十个 LLM，揭示其在神经嵌入检索上的致命短板。</p>
</dd>
<dt><strong><a href="uqabench_evaluating_user_embedding_for_prompting_llms_in_personalized_quest/">UQABench: Evaluating User Embedding for Prompting LLMs in Personalized Question Answering</a></strong></dt>
<dd>
<p>提出首个标准化三阶段（预训练-微调-评测）用户嵌入软提示基准，用序列理解/行为预测/兴趣感知三类任务揭示嵌入式生成推荐与文本式相当且省 8-19× token。</p>
</dd>
</dl>
