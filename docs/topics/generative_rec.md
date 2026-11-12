# ✨ 生成式推荐与离散语义 ID (Generative Recommendation & Semantic IDs)

> **研究范畴**：RQ-VAE/VQ 残差量化码本、TIGER 自回归生成式检索、扩散/流匹配生成推荐 (Diffusion/Flow Rec)

当前方向共收录 **31** 篇顶会/顶刊论文解读：

<dl>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/a_cascaded_generative_approach_for_e_commerce_recommendations/">A Cascaded Generative Approach for e-Commerce Recommendations</a></strong></dt>
<dd>
<p>级联 LLM 生成货架主题与 RAG 约束关键词驱动检索，教师-学生蒸馏落地，线上加购 +2.7%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/adaptive_autoguidance_for_item_side_fairness_in_diffusion_recommender_syste/">Adaptive Autoguidance for Item-Side Fairness in Diffusion Recommender Systems</a></strong></dt>
<dd>
<p>用自适应自引导网络逐步学习引导权重，配合流行度正则化，在扩散推荐中低成本提升物品侧公平性。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/bear_towards_beam_search_aware_optimization_for_recommendation_with_large_l/">BEAR: Towards Beam-Search-Aware Optimization for Recommendation with Large Language Models</a></strong></dt>
<dd>
<p>揭示 SFT 训练与束搜索解码的训练-推理不一致，提出束搜索感知正则 BEAR，四数据集平均提升 12.50%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/beyond_item_ids_scaling_short_form_video_recommendation_via_semantic_native/">Beyond Item IDs: Scaling Short-Form-Video Recommendation via Semantic-Native Long Sequence Modeling</a></strong></dt>
<dd>
<p>用深度截断 Bi-gram 语义ID替代正交Video ID，并以无参时间折叠压缩超长序列，在十亿用户短视频平台实现L=2000序列建模与线上显著增益。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/beyond_static_best_of_n_bayesian_list_wise_alignment_for_llm_based_recommen/">Beyond Static Best-of-N: Bayesian List-wise Alignment for LLM-based Recommendation</a></strong></dt>
<dd>
<p>提出 BLADE 贝叶斯动态估计框架，融合静态先验与实时 rollout 证据构建自演化目标，突破静态 BoN 对齐的性能上界，在 NDCG/Recall/公平性/多样性上全面超越基线。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/bi_level_optimization_for_generative_recommendation_bridging_tokenization_a/">Bi-Level Optimization for Generative Recommendation: Bridging Tokenization and Generation</a></strong></dt>
<dd>
<p>将 tokenizer 与 recommender 重构为双层优化问题，用元学习+梯度手术对齐语义ID与推荐目标，全库排序下稳定超越 SOTA。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/agent_conversational_rec/bridging_passive_and_active_enhancing_conversation_starter_recommendation_v/">Bridging Passive and Active: Enhancing Conversation Starter Recommendation via Active Expression Modeling</a></strong></dt>
<dd>
<p>用对抗分布对齐+语义离散化把用户主动查询引入 Conversation Starter 推荐，打破曝光-点击闭环，离线 AUC +6.72%、线上特征渗透率 +0.54%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/card_non_uniform_quantization_of_visual_semantic_unit_for_generative_recomm/">CARD: Non-Uniform Quantization of Visual Semantic Unit for Generative Recommendation</a></strong></dt>
<dd>
<p>将文本/视觉/协同信号渲染为卡片式视觉语义单元，并用可学习可逆非均匀变换的 NU-RQ-VAE 量化，缓解码本失衡与生成偏置。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/case_cadence_aware_set_encoding_for_large_scale_next_basket_repurchase_reco/">CASE: Cadence-Aware Set Encoding for Large-Scale Next Basket Repurchase Recommendation</a></strong></dt>
<dd>
<p>用日历时间二值信号+共享多尺度时序CNN+ISAB集合编码做下一篮复购推荐，全库候选排序下离线与生产均显著提升</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/differentiable_semantic_id_for_generative_recommendation/">Differentiable Semantic ID for Generative Recommendation</a></strong></dt>
<dd>
<p>用 Gumbel 噪声+不确定性衰减实现可微语义ID联合训练，缓解码本坍缩，Full-Ranking 下 NDCG@10 稳定提升</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multimodal_rec/discrete_preference_learning_for_personalized_multimodal_generation/">Discrete Preference Learning for Personalized Multimodal Generation</a></strong></dt>
<dd>
<p>用模态专属LightGCN+RQ-VAE将连续偏好量化为离散token注入LLM/扩散生成器，配合跨模态一致性奖励实现个性化图文生成。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/dual_diffusional_generative_fashion_recommendation/">Dual-Diffusional Generative Fashion Recommendation</a></strong></dt>
<dd>
<p>双扩散 Transformer 联合生成时尚单品图像与属性级文本，实现可解释个性化穿搭推荐，PFITB/GOR 双任务全面领先。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/multi_task_reranking/equip_pre_ranking_with_target_attention_by_residual_quantization/">Equip Pre-ranking with Target Attention by Residual Quantization</a></strong></dt>
<dd>
<p>用残差量化把 Target Attention 近似搬进粗排，以码本查表替代逐候选交互，淘宝线上 CTR/CVR/GMV 全面显著提升</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/fave_flow_based_average_velocity_establishment_for_sequential_recommendatio/">FAVE: Flow-based Average Velocity Establishment for Sequential Recommendation</a></strong></dt>
<dd>
<p>用语义锚先验替代高斯噪声并学习全局平均速度场，实现单步流匹配生成式序列推荐，全库排序下NDCG@20相对提升9.90%且推理加速20倍以上。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/cross_domain_multi_scenario/fedmm_federated_collaborative_signal_quantization_for_multi_market_ctr_pred/">FedMM: Federated Collaborative Signal Quantization for Multi-Market CTR Prediction</a></strong></dt>
<dd>
<p>用双层 RQ-VAE 码本做联邦传输介质，全局码本聚合跨市场协同信号、本地码本保留市场特异性，在 XMarket 2/3/4 市场 CTR 任务上以 AUC 超越联邦与集中式 MMR 基线。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/gated_bidirectional_linear_attention_for_generative_retrieval/">Gated Bidirectional Linear Attention for Generative Retrieval</a></strong></dt>
<dd>
<p>提出线性时间双向注意力 GBLA，以 Conv1D+Key Gating+Gated RMSNorm 三件套替换编码器自注意力，混合 1:2 堆叠在 Yandex Music 上持平双向 SA 质量，L=32768 单层提速 8.2×。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/genrec_a_preference_oriented_generative_framework_for_large_scale_recommend/">GenRec: A Preference-Oriented Generative Framework for Large-Scale Recommendation</a></strong></dt>
<dd>
<p>提出 Page-wise NTP + 线性 Token Merger + GRPO-SR 偏好对齐的工业级生成式召回框架，京东线上点击 +9.5%、成交 +8.7%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/cold_start_rec/genrecedit_adapting_model_editing_for_generative_recommendation_with_cold_s/">GenRecEdit: Adapting Model Editing for Generative Recommendation with Cold-Start Items</a></strong></dt>
<dd>
<p>将NLP模型编辑范式迁移至生成式推荐，通过位置级迭代编辑与One-One触发机制注入冷启动物品SID模式，仅需9.5%重训时间即显著提升冷启推荐。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/gflowgr_fine_tuning_generative_recommendation_frameworks_with_generative_fl/">GFlowGR: Fine-tuning Generative Recommendation Frameworks with Generative Flow Networks</a></strong></dt>
<dd>
<p>将GR重构为多步集合生成任务，用GFlowNets做token级微调对齐物品效用，三数据集双骨干一致提升并落地淘宝搜索广告。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/learning_decomposed_contextual_token_representations_from_pretrained_and_co/">Learning Decomposed Contextual Token Representations from Pretrained and Collaborative Signals for Generative Recommendation</a></strong></dt>
<dd>
<p>DECOR 通过分解嵌入融合保留冻结预训练码本语义、上下文化 token 组合动态消歧，在 Amazon 三数据集 Full-Ranking 下 NDCG@10 最高提升 14%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/lwgr_lagrangian_constrained_personalized_world_knowledge_for_generative_rec/">LWGR: Lagrangian-Constrained Personalized World Knowledge for Generative Recommendation</a></strong></dt>
<dd>
<p>用 OPQ 并行码本生成用户个性化软指令提取 LLM 世界知识，并以拉格朗日原始-对偶约束可控融合进生成式推荐，离线最高 +11.23%，线上广告收入 +1.35%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/mitigating_collaborative_semantic_id_staleness_in_generative_retrieval/">Mitigating Collaborative Semantic ID Staleness in Generative Retrieval</a></strong></dt>
<dd>
<p>提出模型无关的 SID 对齐更新，将新日志重建的协同语义 ID 映射回旧词表以支持热启动微调，严格时序评估下 Recall@500 提升且训练算力降低约 8-9 倍。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/mviger_multi_view_variational_integration_of_complementary_knowledge_for_ge/">MVIGER: Multi-View Variational Integration of Complementary Knowledge for Generative Recommender</a></strong></dt>
<dd>
<p>将异构模板-索引组合视为多视图，用类别隐变量变分整合互补知识，三数据集全库排序显著超越生成式基线。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/next_interest_flow_a_generative_pre_training_paradigm_for_recommender_syste/">Next Interest Flow: A Generative Pre-training Paradigm for Recommender Systems by Modeling All-domain Movelines</a></strong></dt>
<dd>
<p>将用户意图建模为隐流形上的连续演化轨迹，用切空间正交与测地平滑约束生成式预训练，并以双向对齐+TSP桥接判别式精排，淘宝线上CTCVR+11.6%。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/rclrec_reverse_curriculum_learning_for_modeling_sparse_conversions_in_gener/">RCLRec: Reverse Curriculum Learning for Modeling Sparse Conversions in Generative Recommendation</a></strong></dt>
<dd>
<p>逆向课程前缀+课程质量感知损失为稀疏转化目标注入实例级中间监督，离线全库排序显著提升并线上A/B +2.09%广告收入。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/request_only_optimization_for_recommendation_systems/">Request-Only Optimization for Recommendation Systems</a></strong></dt>
<dd>
<p>以用户请求为训练样本单元，通过数据/系统/模型协同设计消除请求内特征冗余，训练吞吐提升最高570%，同等算力下模型FLOPs扩展7倍。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/sequential_rec/rote_coarse_to_fine_multi_level_rotary_time_embedding_for_sequential_recomm/">RoTE: Coarse-to-Fine Multi-Level Rotary Time Embedding for Sequential Recommendation</a></strong></dt>
<dd>
<p>将时间戳分解为年/月/日多粒度并借 RoPE 旋转注入 Q/K，即插即用提升 SASRec/RPG，NDCG@5 最高 +20.11%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/semantic_ids_for_recommender_systems_at_snapchat_use_cases_technical_challe/">Semantic IDs for Recommender Systems at Snapchat: Use Cases, Technical Challenges, and Design Choices</a></strong></dt>
<dd>
<p>Snapchat 工业级 SID 落地实践：STE 全码本反传与多模态融合缓解码本坍缩，启发式桶内消歧解决 SID-Item 碰撞，离线与线上 A/B 双验证。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/sigma_a_semantic_grounded_instruction_driven_generative_multi_task_recommen/">SIGMA: A Semantic-Grounded Instruction-Driven Generative Multi-Task Recommender at AliExpress</a></strong></dt>
<dd>
<p>多视图语义对齐+混合 SID/ID 分词+指令驱动多任务 SFT+三步生成与自适应概率融合，AliExpress 线上 GMV +7.84%</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/generative_rec/universal_item_tokenization_for_transferable_generative_recommendation/">Universal Item Tokenization for Transferable Generative Recommendation</a></strong></dt>
<dd>
<p>基于MLLM与树状码本的通用物品分词器，融合多模态重建与协同对齐，实现跨域可迁移生成式推荐。</p>
</dd>
<dt><strong><a href="../../venues/SIGIR/2026/llm_rec/verifiable_reasoning_for_llm_based_generative_recommendation/">Verifiable Reasoning for LLM-based Generative Recommendation</a></strong></dt>
<dd>
<p>提出 reason-verify-recommend 范式，用混合验证器+代理偏好预测+单调性正则，在四个数据集上显著提升生成式推荐效果。</p>
</dd>
</dl>
