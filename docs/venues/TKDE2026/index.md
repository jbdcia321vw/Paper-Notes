---
hide:
  - toc
---

<style>
.recsys-hero {
  text-align: center;
  padding: 2.6rem 1.5rem 2.2rem;
  margin-bottom: 2rem;
  border-radius: 16px;
  background: linear-gradient(135deg, rgba(63,81,181,0.10) 0%, rgba(33,150,243,0.06) 50%, rgba(156,39,176,0.08) 100%);
  border: 1px solid rgba(63,81,181,0.16);
}
.recsys-hero h1 {
  font-size: 2.2rem;
  font-weight: 800;
  margin: 0 0 0.6rem 0;
  letter-spacing: -0.02em;
}
.recsys-hero p {
  font-size: 1.05rem;
  opacity: 0.82;
  max-width: 740px;
  margin: 0 auto 1.4rem;
}
.recsys-stats {
  display: flex;
  justify-content: center;
  gap: 1rem;
  flex-wrap: wrap;
}
.recsys-stat-pill {
  padding: 0.45rem 1.1rem;
  border-radius: 999px;
  background: var(--md-default-bg-color);
  border: 1px solid rgba(128,128,128,0.2);
  font-size: 0.9rem;
  font-weight: 600;
  box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}
.recsys-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 1rem;
  margin: 1.2rem 0 2.2rem;
}
.recsys-card {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: 1.15rem 1.25rem;
  border-radius: 12px;
  background: var(--md-default-bg-color);
  border: 1px solid rgba(128,128,128,0.18);
  text-decoration: none !important;
  color: var(--md-default-fg-color) !important;
  transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}
.recsys-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 8px 24px rgba(0,0,0,0.08);
  border-color: var(--md-primary-fg-color);
}
.recsys-card-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.45rem;
}
.recsys-card-title {
  font-size: 1.02rem;
  font-weight: 700;
}
.recsys-badge {
  font-size: 0.75rem;
  padding: 0.15rem 0.55rem;
  border-radius: 999px;
  background: rgba(63,81,181,0.12);
  color: var(--md-primary-fg-color);
  font-weight: 600;
  white-space: nowrap;
}
.recsys-card-desc {
  font-size: 0.84rem;
  opacity: 0.75;
  line-height: 1.45;
}
</style>

# 📖 IEEE TKDE 2026 推荐系统论文解读汇总

- **英文全称**: IEEE Transactions on Knowledge and Data Engineering
- **级别定位**: CCF-A · 知识与数据工程顶刊 · 当前共收录 **0** 篇笔记

## 按推荐系统细分方向浏览

<div class="recsys-grid">
  <a class="recsys-card" href="collaborative_filtering/">    <div class="recsys-card-head">      <span class="recsys-card-title">🤝 协同过滤与双塔召回</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">矩阵分解、双塔向量内积检索、ANN 索引量化、解耦表征与负采样理论</div>  </a>
  <a class="recsys-card" href="graph_rec/">    <div class="recsys-card-head">      <span class="recsys-card-title">🕸️ 图神经网络与知识图谱推荐</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">LightGCN 谱图卷积改进、超图推荐 (Hypergraph)、异构图与知识图谱推理 (KGRec)</div>  </a>
  <a class="recsys-card" href="self_supervised_denoising/">    <div class="recsys-card-head">      <span class="recsys-card-title">🔄 自监督对比学习与交互去噪</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">图/序列对比数据增广 (SGL/SimGCL)、隐式反馈假阳性噪声过滤 (Denoising) 与鲁棒表征</div>  </a>
  <a class="recsys-card" href="sequential_rec/">    <div class="recsys-card-head">      <span class="recsys-card-title">⏱️ 序列与会话推荐</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测</div>  </a>
  <a class="recsys-card" href="generative_rec/">    <div class="recsys-card-head">      <span class="recsys-card-title">✨ 生成式推荐与离散语义 ID</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">RQ-VAE/VQ 残差量化码本、TIGER 自回归生成式检索、扩散/流匹配生成推荐 (Diffusion/Flow Rec)</div>  </a>
  <a class="recsys-card" href="llm_rec/">    <div class="recsys-card-head">      <span class="recsys-card-title">🧠 大语言模型推荐 (LLM4Rec)</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">LLM 指令微调推荐、In-Context 列表重排、语义表征增强、大模型向轻量推荐模型知识蒸馏</div>  </a>
  <a class="recsys-card" href="agent_conversational_rec/">    <div class="recsys-card-head">      <span class="recsys-card-title">🗣️ 对话推荐与智能体仿真</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">多轮主动追问对话推荐 (CRS)、基于 LLM Agent 的用户行为模拟沙盒、可解释性推荐理由生成</div>  </a>
  <a class="recsys-card" href="multimodal_rec/">    <div class="recsys-card-head">      <span class="recsys-card-title">🧩 多模态推荐与内容理解</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">视觉-文本-音频特征融合与解耦、多模态图对齐、短视频/微视频推荐与图文电商匹配</div>  </a>
  <a class="recsys-card" href="social_group_rec/">    <div class="recsys-card-head">      <span class="recsys-card-title">👥 社交与群组推荐</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">社交信任网络传播、社交同质性去噪、群组偏好共识聚合 (Group Rec) 与影响力建模</div>  </a>
  <a class="recsys-card" href="cold_start_rec/">    <div class="recsys-card-head">      <span class="recsys-card-title">❄️ 冷启动与长尾推荐</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">Strict Zero-shot 新用户/新物品冷启动、元学习 (Meta-Learning)、少样本适应与长尾物品扶持</div>  </a>
  <a class="recsys-card" href="cross_domain_multi_scenario/">    <div class="recsys-card-head">      <span class="recsys-card-title">🌐 跨域与多场景推荐</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">跨域知识迁移 (CDR)、重叠/非重叠用户对齐、多业务场景联合建模 (STAR/PEPNet 等)</div>  </a>
  <a class="recsys-card" href="poi_spatiotemporal_rec/">    <div class="recsys-card-head">      <span class="recsys-card-title">📍 时空轨迹与 POI 本地生活推荐</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">Next-POI 兴趣点推荐、时空序列轨迹建模、LBS 本地生活服务与移动出行推荐</div>  </a>
  <a class="recsys-card" href="ctr_feature_interaction/">    <div class="recsys-card-head">      <span class="recsys-card-title">🎯 CTR/CVR 预估与特征交叉</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">显式/隐式高阶特征交叉网络 (DCN/EulerNet)、转化率延迟反馈 (Delayed Feedback) 与概率校准</div>  </a>
  <a class="recsys-card" href="multi_task_reranking/">    <div class="recsys-card-head">      <span class="recsys-card-title">⚖️ 多任务学习与列表级重排</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">MMoE/PLE 多任务多目标精排、帕累托最优、上下文列表重排 (Re-ranking) 与粗精排级联一致性</div>  </a>
  <a class="recsys-card" href="rl_bandit_rec/">    <div class="recsys-card-head">      <span class="recsys-card-title">🎮 强化学习与动态探索推荐</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">Contextual Bandit 探索与利用 (E&E)、Offline RL 长期留存与生命周期价值 (LTV) 策略优化</div>  </a>
  <a class="recsys-card" href="debias_fairness_industrial/">    <div class="recsys-card-head">      <span class="recsys-card-title">🛡️ 去偏公平、可信隐私与工业架构</span>      <span class="recsys-badge">0 篇</span>    </div>    <div class="recsys-card-desc">曝光/流行度/位置因果去偏 (Causal Rec)、双边公平性、联邦防投毒推荐、工业级全链路架构与评测基准</div>  </a>
</div>
