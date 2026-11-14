---
hide:
  - toc
---

# 🏛️ ACL / EMNLP (RecSys & CRS) 2025 推荐系统论文解读

41 篇 ACL / EMNLP (RecSys & CRS) 2025 论文解读，覆盖 7 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (5)</h2>
<dl>
<dt><strong><a href="generative_rec/a_generative_framework_for_personalized_sticker_retrieval/">A Generative Framework for Personalized Sticker Retrieval</a></strong></dt>
<dd>
<p>PEARL 用用户组嵌入+LLM意图排序加权解码，实现微信表情包个性化生成式检索，离线 MRR@10 提升 15%、线上 CTR +7.12%。</p>
</dd>
<dt><strong><a href="generative_rec/enhancing_time_awareness_in_generative_recommendation/">Enhancing Time Awareness in Generative Recommendation</a></strong></dt>
<dd>
<p>将绝对时间戳与目标相对间隔verbalize进提示，并叠加全局转移图与免训练趋势重排，Full-Ranking下R@5最高提升15.4%。</p>
</dd>
<dt><strong><a href="generative_rec/gram_generative_recommendation_via_semantic_aware_multi_granular_late_fusio/">GRAM: Generative Recommendation via Semantic-aware Multi-granular Late Fusion</a></strong></dt>
<dd>
<p>将层次/协同语义翻译为 LLM 词表内文本 ID，并用多粒度提示+后期融合保留长文本信息，Full-Ranking 下 R@5 提升 11.5–16.0%。</p>
</dd>
<dt><strong><a href="generative_rec/recbase_generative_foundation_model_pretraining_for_zero_shot_recommendatio/">RecBase: Generative Foundation Model Pretraining for Zero-Shot Recommendation</a></strong></dt>
<dd>
<p>用课程学习增强的 RQ-VAE 将跨域物品统一离散为层级概念 ID，从零预训练 1.5B 自回归推荐基座，在 8 个未见数据集零样本 AUC 全面超越 7B LLM。</p>
</dd>
<dt><strong><a href="generative_rec/recgpt_a_foundation_model_for_sequential_recommendation/">RecGPT: A Foundation Model for Sequential Recommendation</a></strong></dt>
<dd>
<p>纯文本驱动+FSQ统一分词+混合双向因果注意力的序列推荐基础模型，零样本跨域显著超越10%数据微调的ID基线。</p>
</dd>
</dl>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (19)</h2>
<dl>
<dt><strong><a href="llm_rec/a_text_based_recommender_system_that_leverages_explicit_affective_state_pre/">A Text-Based Recommender System that Leverages Explicit Affective State Preferences</a></strong></dt>
<dd>
<p>提出情感-认知状态显式偏好推荐任务，用 GPT-4o 从书评挖掘 AC 语句构建数据集，Transformer 融合长短期偏好与 AC 描述，全库排序 HR@10 达 15.6%。</p>
</dd>
<dt><strong><a href="llm_rec/copl_collaborative_preference_learning_for_personalizing_llms/">CoPL: Collaborative Preference Learning for Personalizing LLMs</a></strong></dt>
<dd>
<p>用用户-回复二部图协同过滤学习用户表征，结合 MoLE 路由个性化奖励模型，并支持无优化未见用户适配，在 UltraFeedback-P 等基准上显著超越个性化奖励模型基线。</p>
</dd>
<dt><strong><a href="llm_rec/cove_compressed_vocabulary_expansion_makes_better_llm_based_recommender_sys/">CoVE: Compressed Vocabulary Expansion Makes Better LLM-based Recommender Systems</a></strong></dt>
<dd>
<p>将物品映射为扩展词表唯一 token，用 logits 直接打分替代文本生成，并以哈希压缩嵌入表，实现最高 62% 精度提升与约 100× 推理加速。</p>
</dd>
<dt><strong><a href="llm_rec/decoding_in_latent_spaces_for_efficient_inference_in_llm_based_recommendati/">Decoding in Latent Spaces for Efficient Inference in LLM-based Recommendation</a></strong></dt>
<dd>
<p>用训练样本末层隐状态聚合出物品潜在表征，以 L2 距离匹配替代自回归语言空间解码，推理提速 10 倍以上且性能不降反升。</p>
</dd>
<dt><strong><a href="llm_rec/easyrec_simple_yet_effective_language_models_for_recommendation/">EasyRec: Simple yet Effective Language Models for Recommendation</a></strong></dt>
<dd>
<p>用 LLM 生成用户/物品画像并以对比学习对齐协同信号，训练轻量文本编码器实现零样本推荐，Full-Ranking 下 Recall@20 最高提升 32%。</p>
</dd>
<dt><strong><a href="llm_rec/generative_product_recommendations_for_implicit_superlative_queries/">Generative Product Recommendations for Implicit Superlative Queries</a></strong></dt>
<dd>
<p>提出 SUPERB 四级标注体系与深思式属性生成，用 LLM 列表级重排处理隐式最高级商品查询，nDCG@10 相对 BM25 提升约 18%。</p>
</dd>
<dt><strong><a href="llm_rec/kerl_knowledge_enhanced_personalized_recipe_recommendation_using_large_lang/">KERL: Knowledge-Enhanced Personalized Recipe Recommendation using Large Language Models</a></strong></dt>
<dd>
<p>用 FoodKG 子图作上下文 + 多 LoRA 适配器，统一完成约束食谱推荐、烹饪步骤生成与微量营养素估计，KGQA 基准 F1 达 0.973。</p>
</dd>
<dt><strong><a href="llm_rec/knowledge_graph_retrieval_augmented_generation_for_llm_based_recommendation/">Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation</a></strong></dt>
<dd>
<p>用跳场知识子图索引+流行度选择性检索+GNN软提示，将KG结构知识注入冻结LLM，MovieLens-1M ACC 相对提升58.6%且推理仅慢0.14s</p>
</dd>
<dt><strong><a href="llm_rec/lamp_val_large_language_models_empower_personalized_valuation_in_auction/">LaMP-Val: Large Language Models Empower Personalized Valuation in Auction</a></strong></dt>
<dd>
<p>用 LLM 从用户评论语义中做个性化估值，构建 23k 估值数据集 + 多样化指令 SFT，并提出 PU/PV 闭环拍卖指标，PU 较 GPT-4 提升约 6.5 倍。</p>
</dd>
<dt><strong><a href="llm_rec/maps_motivation_aware_personalized_search_via_llm_driven_consultation_align/">MAPS: Motivation-Aware Personalized Search via LLM-Driven Consultation Alignment</a></strong></dt>
<dd>
<p>用 LLM+MoAE 将售前咨询动机与查询/商品/历史双向对齐，Commercial 与 Amazon 上排序 HR@10 提升约 20%/35%。</p>
</dd>
</dl>
<p><a href="llm_rec/">查看全部19篇「大语言模型推荐 (LLM4Rec)」论文 →</a></p>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (8)</h2>
<dl>
<dt><strong><a href="agent_conversational_rec/a_survey_on_llm_powered_agents_for_recommender_systems/">A Survey on LLM-powered Agents for Recommender Systems</a></strong></dt>
<dd>
<p>系统综述 LLM Agent 推荐的三范式（推荐/交互/仿真）与四模块架构，梳理数据集与评测协议并指出统一评测缺失。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/empowering_retrieval_based_conversational_recommendation_with_contrasting_u/">Empowering Retrieval-based Conversational Recommendation with Contrasting User Preferences</a></strong></dt>
<dd>
<p>用 LLM 抽取并扩增 like/dislike 对比偏好，以三路相似度线性打分做稠密检索式对话推荐，INSPIRED 上 Recall@10 相对提升 99.72%。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/enhancing_recommendation_explanations_through_user_centric_refinement/">Enhancing Recommendation Explanations through User-Centric Refinement</a></strong></dt>
<dd>
<p>提出 RefineX 多智能体推理期精炼框架，以 plan-then-refine 与分层反思将 PETER 解释在事实性/个性化/连贯性三维度大幅提升。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/exploring_the_potential_of_llms_as_personalized_assistants_dataset_evaluati/">Exploring the Potential of LLMs as Personalized Assistants: Dataset, Evaluation, and Analysis</a></strong></dt>
<dd>
<p>提出 HiCUPID 个性化助手基准（1.5k 合成用户、17k token 长对话、单/多跳 QA），并蒸馏 Llama-3.2 评估器对齐人类偏好，揭示 SFT+DPO 最优而纯 DPO 崩溃。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/iagent_llm_agent_as_a_shield_between_user_and_recommender_systems/">iAgent: LLM Agent as a Shield between User and Recommender Systems</a></strong></dt>
<dd>
<p>提出 user-agent-platform 新范式，用个体指令感知 LLM Agent 在用户与平台间做重排护盾，四数据集平均提升 16.6%。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/muse_a_multimodal_conversational_recommendation_dataset_with_scenario_groun/">Muse: A Multimodal Conversational Recommendation Dataset with Scenario-Grounded User Profiles</a></strong></dt>
<dd>
<p>首个多模态对话推荐数据集，用场景锚定用户画像的多智能体框架合成 7K 段图文对话，并验证三种 MLLM 的可学习性。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/refining_text_generation_for_realistic_conversational_recommendation_via_di/">Refining Text Generation for Realistic Conversational Recommendation via Direct Preference Optimization</a></strong></dt>
<dd>
<p>用 DPO 对齐 LLM 生成的对话摘要与物品推荐信息，使评分器更准，Tabidachi 上 HR@1 提升 21.3%。</p>
</dd>
<dt><strong><a href="agent_conversational_rec/towards_personalized_conversational_sales_agents_contextual_user_profiling_/">Towards Personalized Conversational Sales Agents: Contextual User Profiling for Strategic Action</a></strong></dt>
<dd>
<p>提出 CSALES 对话销售任务与 CSUSER 真实画像用户模拟器，CSI 智能体通过上下文画像驱动统一动作空间，SR 提升约 15-20pt、SWR 提升约 5-10pt。</p>
</dd>
</dl>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (2)</h2>
<dl>
<dt><strong><a href="multimodal_rec/icg_improving_cover_image_generation_via_mllm_based_prompting_and_personali/">ICG: Improving Cover Image Generation via MLLM-based Prompting and Personalized Preference Alignment</a></strong></dt>
<dd>
<p>用 MLLM 元令牌+双路适配器把个性化上下文注入扩散模型，并以多奖励对齐实现无标签封面生成，下游 MMGCN 推荐 NDCG@10 提升 19.1%。</p>
</dd>
<dt><strong><a href="multimodal_rec/premise_matching_based_prediction_for_accurate_review_recommendation/">PREMISE: Matching-based Prediction for Accurate Review Recommendation</a></strong></dt>
<dd>
<p>以多尺度多字段语义匹配分数替代跨模态融合表征，用于多模态评论有用性排序，MAP 较 GBDT 提升 5–17 点且训练提速 42%。</p>
</dd>
</dl>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl>
<dt><strong><a href="cold_start_rec/scholar_inbox_personalized_paper_recommendations_for_scientists/">Scholar Inbox: Personalized Paper Recommendations for Scientists</a></strong></dt>
<dd>
<p>面向科研论文的开放获取推荐平台，以显式点赞/点踩训练加权逻辑回归，结合 Scholar Maps 与主动学习破解冷启动，在 80 万评分数据集上 GTE-Large 达 AUC 86.75。</p>
</dd>
</dl>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (1)</h2>
<dl>
<dt><strong><a href="poi_spatiotemporal_rec/triptailor_a_real_world_benchmark_for_personalized_travel_planning/">TripTailor: A Real-World Benchmark for Personalized Travel Planning</a></strong></dt>
<dd>
<p>构建50万真实POI与近4000条真人行程的个性化旅行规划基准，提出可行性/合理性/个性化三维评估与奖励模型对比框架，揭示SOTA LLM行程超越真人比例不足10%。</p>
</dd>
</dl>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (5)</h2>
<dl>
<dt><strong><a href="debias_fairness_industrial/a_federated_framework_for_llm_based_recommendation/">A Federated Framework for LLM-based Recommendation</a></strong></dt>
<dd>
<p>FELLRec 用动态聚合+课程升温平衡客户端性能，并以分层卸载降低 LLM 联邦推荐端侧开销。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/adtec_a_unified_benchmark_for_evaluating_text_quality_in_search_engine_adve/">AdTEC: A Unified Benchmark for Evaluating Text Quality in Search Engine Advertising</a></strong></dt>
<dd>
<p>首个公开的搜索广告文案质量多维评测基准，定义五任务并对比 PLM/LLM/人类评估者表现。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/bias_beware_the_impact_of_cognitive_biases_on_llm_driven_product_recommenda/">Bias Beware: The Impact of Cognitive Biases on LLM-Driven Product Recommendations</a></strong></dt>
<dd>
<p>首次将人类认知偏差作为黑盒重排攻击注入商品描述，揭示 LLM 推荐器对 social proof 等偏差高度敏感且难以防御。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/richer_output_for_richer_countries_uncovering_geographical_disparities_in_g/">Richer Output for Richer Countries: Uncovering Geographical Disparities in Generated Stories and Travel Recommendations</a></strong></dt>
<dd>
<p>在10万旅行请求与20万故事生成上量化5个LLM的地理不平等：穷国推荐独特性低、地理实体少、故事更多苦难与悲伤，且与人均GDP显著相关。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/stereotype_or_personalization_user_identity_biases_chatbot_recommendations/">Stereotype or Personalization? User Identity Biases Chatbot Recommendations</a></strong></dt>
<dd>
<p>揭示 LLM 聊天机器人无论用户显式还是隐式透露种族身份都会产生刻板推荐，且在被追问时系统性掩盖种族影响，跨 7 个主流模型、4 个种族、12 万条合成请求验证。</p>
</dd>
</dl>
</div>
