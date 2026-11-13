---
hide:
  - toc
---

# 🏛️ ACM MM 2026 推荐系统论文解读

3 篇 ACM MM 2026 论文解读，覆盖 1 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (3)</h2>
<dl>
<dt><strong><a href="multimodal_rec/beyond_modality_harmony_orthogonal_purification_and_topology_guided_moe_for/">Beyond Modality Harmony: Orthogonal Purification and Topology-Guided MoE for Conflict-Aware Multimodal Recommendation</a></strong></dt>
<dd>
<p>以纯协同锚点为几何参考轴正交分解并保能截断模态噪声，配合解耦 Sigmoid 路由 MoE 与冲突感知 safe-SSL，在 Amazon 三数据集 8:1:1 划分下 R@20 最高提升 6.55%。</p>
</dd>
<dt><strong><a href="multimodal_rec/beyond_noisy_signals_dual_level_denoising_for_multi_modal_sequential_recomm/">Beyond Noisy Signals: Dual-Level Denoising for Multi-modal Sequential Recommendation</a></strong></dt>
<dd>
<p>图拉普拉斯低通滤波去特征噪声 + FFT 可学习频域滤波去序列噪声，四数据集 Full-Ranking 全面 SOTA，R@20 最高 +19.33%。</p>
</dd>
<dt><strong><a href="multimodal_rec/one_graph_multiple_gains_single_high_quality_item_item_graph_for_multimodal/">One Graph, Multiple Gains: Single High-Quality Item-Item Graph for Multimodal Recommendation</a></strong></dt>
<dd>
<p>构建一次高质量物品-物品图，经三元闭包重加权后复用于表征、交互图与优化三阶段，四数据集全指标SOTA且训练更快更省显存。</p>
</dd>
</dl>
</div>
