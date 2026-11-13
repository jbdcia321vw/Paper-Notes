---
hide:
  - toc
---

# 📖 ACM TORS 2026 推荐系统论文解读

8 篇 ACM TORS 2026 论文解读，覆盖 6 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (1)</h2>
<dl>
<dt><strong><a href="generative_rec/diffusion_recommender_models_and_the_illusion_of_progress_a_concerning_stud/">Diffusion Recommender Models and the Illusion of Progress: A Concerning Study of Reproducibility and a Conceptual Mismatch</a></strong></dt>
<dd>
<p>复现 9 个 SIGIR&#x27;23-24 扩散推荐模型，仅 25% 结果可复现，调优简单基线全面反超，并揭示 DDPM 与 top-n 推荐任务的概念错配。</p>
</dd>
</dl>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (1)</h2>
<dl>
<dt><strong><a href="llm_rec/music_recommendation_with_large_language_models_challenges_opportunities_an/">Music Recommendation with Large Language Models: Challenges, Opportunities, and Evaluation</a></strong></dt>
<dd>
<p>系统梳理 LLM 驱动音乐推荐在用户建模、物品建模与自然语言请求三方面的变革，并提出 NLP 启发的评估分类体系与 MRS 成功/风险维度框架。</p>
</dd>
</dl>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (1)</h2>
<dl>
<dt><strong><a href="agent_conversational_rec/collab_rec_an_llm_based_agentic_framework_for_balancing_recommendations_in_/">Collab-Rec: An LLM-based Agentic Framework for Balancing Recommendations in Tourism</a></strong></dt>
<dd>
<p>三角色 LLM 智能体 + 确定性 Moderator 多轮约束精炼，在 900 条旅游查询上兼顾约束满足与长尾多样性。</p>
</dd>
</dl>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl>
<dt><strong><a href="multimodal_rec/adopting_state_of_the_art_pretrained_audio_representations_for_music_recomm/">Adopting State-of-the-Art Pretrained Audio Representations for Music Recommender Systems</a></strong></dt>
<dd>
<p>系统评测 9 个 MIR 预训练音频表征在 5 类推荐器上的热/冷启动表现，揭示 MIR 性能无法直接迁移到推荐任务。</p>
</dd>
</dl>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (1)</h2>
<dl>
<dt><strong><a href="poi_spatiotemporal_rec/point_of_interest_recommendation_pitfalls_and_viable_solutions/">Point of Interest Recommendation: Pitfalls and Viable Solutions</a></strong></dt>
<dd>
<p>系统梳理POI推荐在数据集、算法与评估三维度的20项陷阱，并提出六方向研究议程与陷阱-方案映射表</p>
</dd>
</dl>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (3)</h2>
<dl>
<dt><strong><a href="debias_fairness_industrial/balancing_fairness_and_high_match_rates_in_reciprocal_recommender_systems_a/">Balancing Fairness and High Match Rates in Reciprocal Recommender Systems: A Nash Social Welfare Approach</a></strong></dt>
<dd>
<p>将双边匹配推荐建模为推荐机会分配，用 Nash 社会福利交替优化实现近无嫉妒的公平推荐，并以 Sinkhorn 加速至大规模场景。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/reproducibility_in_recommender_systems_a_survey/">Reproducibility in Recommender Systems: A Survey</a></strong></dt>
<dd>
<p>对 RecSys 可复现性赛道 2020–2025 年 51 篇论文的结构化元分析，揭示其方法论高度同质化、以“扩展”替代“严格复现”的实践落差。</p>
</dd>
<dt><strong><a href="debias_fairness_industrial/robustness_analysis_of_pareto_based_joint_evaluation_of_fairness_and_releva/">Robustness Analysis of Pareto-based Joint Evaluation of Fairness and Relevance in Recommender Systems</a></strong></dt>
<dd>
<p>提出 DPFR：以测试集真实标签构建相关性-公平性帕累托前沿，用模型得分到前沿参考点的距离联合评估推荐系统，比现有联合度量更稳健。</p>
</dd>
</dl>
</div>
