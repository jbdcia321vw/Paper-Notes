# 🧠 大语言模型推荐 (LLM4Rec) (LLM-Enhanced & LLM-as-Recommender)

> **研究范畴**：LLM 指令微调推荐、In-Context 列表重排、语义表征增强、大模型向轻量推荐模型知识蒸馏

当前方向共收录 **23** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/a_general_framework_for_multimodal_llm_based_multimedia_understanding_in_la/">A General Framework for Multimodal LLM-Based Multimedia Understanding in Large-Scale Recommendation Systems</a></strong></dt>
<dd>
<p>用 LLaMA2-1.5B+BLIP-2 将图像转为 caption 并 tokenize 为类别特征注入 DLRM 精排，离线 AUC +0.35%、线上 +0.02%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/ace_anisotropy_controllable_embedding_for_llm_enhanced_sequential_recommend/">ACE: Anisotropy-Controllable Embedding for LLM-enhanced Sequential Recommendation</a></strong></dt>
<dd>
<p>用线性自编码器的谱收缩算子连续调控 LLM 物品嵌入的各向异性，在 Full-Ranking 留一法协议下 Recall@20 最高提升 12.4%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/bear_towards_beam_search_aware_optimization_for_recommendation_with_large_l/">BEAR: Towards Beam-Search-Aware Optimization for Recommendation with Large Language Models</a></strong></dt>
<dd>
<p>揭示 SFT 训练与束搜索解码的训练-推理不一致，提出束搜索感知正则 BEAR，四数据集平均提升 12.50%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/beyond_static_best_of_n_bayesian_list_wise_alignment_for_llm_based_recommen/">Beyond Static Best-of-N: Bayesian List-wise Alignment for LLM-based Recommendation</a></strong></dt>
<dd>
<p>提出 BLADE 贝叶斯动态估计框架，融合静态先验与实时 rollout 证据构建自演化目标，突破静态 BoN 对齐的性能上界，在 NDCG/Recall/公平性/多样性上全面超越基线。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/agent_conversational_rec/conv_finre_a_conversational_and_longitudinal_benchmark_for_utility_grounded/">Conv-FinRe: A Conversational and Longitudinal Benchmark for Utility-Grounded Financial Recommendation</a></strong></dt>
<dd>
<p>提出首个对话式纵向股票推荐基准，用逆优化推断用户潜在风险偏好，构建效用/动量/风险/用户选择四视图参考，诊断 LLM 是理性分析、行为模仿还是追涨杀跌。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/graph_rec/dcgl_dual_channel_graph_learning_with_large_language_models_for_knowledge_a/">DCGL: Dual-Channel Graph Learning with Large Language Models for Knowledge-Aware Recommendation</a></strong></dt>
<dd>
<p>双通道解耦LLM语义与ID行为信号，多级对比学习对齐，频率感知门控自适应融合，稀疏场景显著提升</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/every_preference_has_its_strength_injecting_ordinal_semantics_into_llm_base/">Every Preference Has Its Strength: Injecting Ordinal Semantics into LLM-Based Recommenders</a></strong></dt>
<dd>
<p>将显式评分建模为序数语义锚点，在LLM隐空间做强度感知对齐，保留细粒度偏好强度，成对偏好准确率最高提升44.9%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/filling_the_gaps_selective_knowledge_augmentation_for_llm_recommenders/">Filling the Gaps: Selective Knowledge Augmentation for LLM Recommenders</a></strong></dt>
<dd>
<p>提出 KnowSA_CKP，用比较式知识探测量化 LLM 对物品的协同知识缺口，仅对知识贫乏物品做选择性增强，免微调提升 Recall@1 与上下文效率。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/genrec_a_preference_oriented_generative_framework_for_large_scale_recommend/">GenRec: A Preference-Oriented Generative Framework for Large-Scale Recommendation</a></strong></dt>
<dd>
<p>提出 Page-wise NTP + 线性 Token Merger + GRPO-SR 偏好对齐的工业级生成式召回框架，京东线上点击 +9.5%、成交 +8.7%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/itemrag_item_based_retrieval_augmented_generation_for_llm_based_recommendat/">ItemRAG: Item-Based Retrieval-Augmented Generation for LLM-Based Recommendation</a></strong></dt>
<dd>
<p>将 RAG 从粗粒度用户历史检索转向细粒度物品级检索，融合共购与语义信息为每个物品生成摘要，标准与冷启场景全面超越用户级 RAG 基线。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/l2rec_towards_dual_view_understanding_of_llms_for_personalized_recommendati/">L2Rec: Towards Dual-View Understanding of LLMs for Personalized Recommendation</a></strong></dt>
<dd>
<p>在 LLM 参数空间用双视图个性化 MoE (LoRA) 分别适配行为/语义视图，ACF 融合，四数据集 N@10 提升 3.87%–8.02%，线上 CTR +9.24%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/agent_conversational_rec/meituan_merchant_business_diagnosis_via_policy_guided_dual_process_user_sim/">Meituan Merchant Business Diagnosis via Policy-Guided Dual-Process User Simulation</a></strong></dt>
<dd>
<p>从行为轨迹挖掘决策策略作为共享对齐层，锚定 LLM 语义推理与 ML 统计拟合双过程分支，群体模拟误差降至 8.80%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/modular_representation_compression_adapting_llms_for_efficient_and_effectiv/">Modular Representation Compression: Adapting LLMs for Efficient and Effective Recommendations</a></strong></dt>
<dd>
<p>揭示 LLM 中间层表示优势 (MRA) 现象，提出模块化解耦压缩框架 MARC，将 4096 维 LLM 表示压缩至 128 维，线上 A/B 测试 eCPM 提升 2.82%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/multimodal_large_language_models_with_adaptive_preference_optimization_for_/">Multimodal Large Language Models with Adaptive Preference Optimization for Sequential Recommendation</a></strong></dt>
<dd>
<p>面向 MLLM 序列推荐，提出难度感知重加权 (HaRS) 与噪声正则分布优化 (NoDO) 改进 DPO，在三个数据集上 NDCG@3 相对提升 7.39%–11.02%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multi_task_reranking/one_pass_any_order_position_invariant_listwise_reranking_for_llm_based_reco/">One Pass, Any Order: Position-Invariant Listwise Reranking for LLM-Based Recommendation</a></strong></dt>
<dd>
<p>用结构化注意力掩码+共享位置框架在架构层面消除 LLM 列表重排的候选顺序敏感性，单次前向即达近完美排列稳健性</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/personalized_deep_research_a_user_centric_framework_dataset_and_hybrid_eval/">Personalized Deep Research: A User-Centric Framework, Dataset, and Hybrid Evaluation for Knowledge Discovery</a></strong></dt>
<dd>
<p>将用户画像注入深度研究检索-推理闭环，构建四任务个性化基准与混合评测，个性化指标大幅领先商业系统。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/promax_exploring_the_potential_of_llm_derived_profiles_with_distribution_sh/">ProMax: Exploring the Potential of LLM-derived Profiles with Distribution Shaping for Recommender Systems</a></strong></dt>
<dd>
<p>将 LLM 画像视为检索信号，用监督/自监督双重分布塑形引导基座推荐模型，全库排序下 NDCG@10 最高提升 8.44%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/prompt_unknown_promotion_attacks_against_llm_based_sequential_recommender_s/">Prompt-Unknown Promotion Attacks against LLM-based Sequential Recommender Systems</a></strong></dt>
<dd>
<p>黑盒LLM序列推荐下的提示蒸馏+双重投毒商品推广攻击，HR@50最高提升约3.9倍</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/rethinking_semantic_collaborative_integration_why_alignment_is_not_enough/">Rethinking Semantic–Collaborative Integration: Why Alignment Is Not Enough</a></strong></dt>
<dd>
<p>论证语义-协同表征仅部分共享，全局对齐会压制视图私有信号；提出互补融合视角与诊断指标，最小融合在稀疏基准上超越对齐范式。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/sigma_a_semantic_grounded_instruction_driven_generative_multi_task_recommen/">SIGMA: A Semantic-Grounded Instruction-Driven Generative Multi-Task Recommender at AliExpress</a></strong></dt>
<dd>
<p>多视图语义对齐+混合 SID/ID 分词+指令驱动多任务 SFT+三步生成与自适应概率融合，AliExpress 线上 GMV +7.84%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/agent_conversational_rec/task_aware_automated_user_profile_generation_for_recommendation_simulation_/">Task-Aware Automated User Profile Generation for Recommendation Simulation Using Large Language Models</a></strong></dt>
<dd>
<p>提出免训练三阶段自动画像生成框架 APG4RecSim，在 Full-Ranking 式判别/排序/评分任务上 nDCG@10 提升约 7%、JSD 降低约 8%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/unifying_search_and_recommendation_in_llms_via_gradient_multi_subspace_tuni/">Unifying Search and Recommendation in LLMs via Gradient Multi-Subspace Tuning</a></strong></dt>
<dd>
<p>GEMS 用多子空间梯度分解缓解 S&amp;R 梯度冲突，并用零空间投影保留通用知识，在 Qilin/Amazon 上大幅超越 PEFT 与统一 S&amp;R 基线。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/agent_conversational_rec/when_and_how_to_ask_dynamic_preference_elicitation_strategies_for_conversat/">When and How to Ask: Dynamic Preference Elicitation Strategies for Conversational Recommendation</a></strong></dt>
<dd>
<p>提出 InPE 标注数据集与 COPE 混合专家架构，将对话推荐建模为结构化动作空间上的序贯决策，实现阶段感知的偏好诱导策略动态切换。</p>
</dd>
</dl>
