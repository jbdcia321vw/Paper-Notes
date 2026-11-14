# ❄️ 冷启动与长尾推荐

**RecSys 2025** · **9** 篇论文解读

> **方向说明**：Strict Zero-shot 新用户/新物品冷启动、元学习 (Meta-Learning)、少样本适应与长尾物品扶持

📌 **同领域跨会议浏览：** [RecSys 2026 (4)](../../2026/cold_start_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/cold_start_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/cold_start_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (1)](../../../ACL_EMNLP/2025/cold_start_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/cold_start_rec/) · [CIKM 2025 (4)](../../../CIKM/2025/cold_start_rec/) · [CIKM 2026 (1)](../../../CIKM/2026/cold_start_rec/) · [ECIR 2025 (1)](../../../ECIR/2025/cold_start_rec/) · [ICDE 2025 (1)](../../../ICDE/2025/cold_start_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/cold_start_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/cold_start_rec/) · [KDD 2025 (1)](../../../KDD/2025/cold_start_rec/) · [KDD 2026 (2)](../../../KDD/2026/cold_start_rec/) · [SIGIR 2026 (2)](../../../SIGIR/2026/cold_start_rec/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/cold_start_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/cold_start_rec/) · [WSDM 2025 (1)](../../../WSDM/2025/cold_start_rec/) · [WSDM 2026 (1)](../../../WSDM/2026/cold_start_rec/) · [WWW (TheWebConf) 2025 (2)](../../../WWW/2025/cold_start_rec/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/cold_start_rec/)

🔥 **高频主题：** 召回 ×6 · 重排 ×2

<dl>
<dt><strong><a href="do_we_really_need_specialization_evaluating_generalist_text_embeddings_for_/">Do We Really Need Specialization? Evaluating Generalist Text Embeddings for Zero-Shot Recommendation and Search</a></strong></dt>
<dd>
<p>实证通用文本嵌入(GTE)零样本即超越领域微调模型，并以PCA有效维度揭示空间利用率与压缩收益。</p>
</dd>
<dt><strong><a href="exploring_the_potential_of_llms_for_serendipity_evaluation_in_recommender_s/">Exploring the Potential of LLMs for Serendipity Evaluation in Recommender Systems</a></strong></dt>
<dd>
<p>用 LLM 模拟用户对推荐结果做惊喜度（serendipity）打分，在淘宝与 MovieLens 两个真实用户调研数据集上做元评估，最优配置 Pearson 相关达 21.5%，显著超越传统代理指标。</p>
</dd>
<dt><strong><a href="item_centric_exploration_for_cold_start_problem/">Item-centric Exploration for Cold Start Problem</a></strong></dt>
<dd>
<p>提出以物品为中心的 Beta 后验过滤控制，在排序后剔除“错配受众”曝光，线上探索曝光降 20% 而满意度升 50%、可推荐语料扩 10%。</p>
</dd>
<dt><strong><a href="let_it_go_not_quite_addressing_item_cold_start_in_sequential_recommendation/">Let It Go? Not Quite: Addressing Item Cold Start in Sequential Recommendations with Content-Based Initialization</a></strong></dt>
<dd>
<p>冻结内容嵌入+有界可训练 delta 校正，兼顾冷启动物品语义结构与热物品任务适配，三数据集冷启动 NDCG@10 最高提升 20.9%。</p>
</dd>
<dt><strong><a href="llm_recg_a_semantic_bias_aware_framework_for_zero_shot_sequential_recommend/">LLM-RecG: A Semantic Bias-Aware Framework for Zero-Shot Sequential Recommendation</a></strong></dt>
<dd>
<p>揭示 LLM 语义嵌入的域语义偏置，用域间紧致+域内多样双熵损失与源域序列模式注意力实现零样本跨域序列推荐，R@10 平均提升超 10%。</p>
</dd>
<dt><strong><a href="on_inherited_popularity_bias_in_cold_start_item_recommendation/">On Inherited Popularity Bias in Cold-Start Item Recommendation</a></strong></dt>
<dd>
<p>揭示生成式冷启动模型继承热模型流行度偏置，提出基于向量模长重标定的免训练后处理去偏方法，在三个多媒体数据集上显著提升物品公平性与曝光多样性。</p>
</dd>
<dt><strong><a href="reading_between_the_lines_a_study_of_thematic_bias_in_book_recommender_syst/">Reading Between the Lines: A Study of Thematic Bias in Book Recommender Systems</a></strong></dt>
<dd>
<p>用 BERTopic 为 Book-Crossing 打主题标签，构建数据-推荐-用户三阶段主题偏置评估框架，揭示长尾小众读者被主流主题同质化的现象。</p>
</dd>
<dt><strong><a href="recommendation_is_a_dish_better_served_warm/">Recommendation Is a Dish Better Served Warm</a></strong></dt>
<dd>
<p>系统量化冷启动阈值：物品侧训练交互扫描+用户侧推理历史截断，揭示 6–15 条交互的冷热分界并批判 p-core 过滤的任意性。</p>
</dd>
<dt><strong><a href="socripple_a_two_stage_framework_for_cold_start_video_recommendations/">SocRipple: A Two-Stage Framework for Cold-Start Video Recommendations</a></strong></dt>
<dd>
<p>社交种子+嵌入KNN涟漪扩散的两阶段冷启视频召回框架，Recall@200 提升至 12.8–13.2%，线上冷启曝光 +36% 且互动率持平。</p>
</dd>
</dl>
