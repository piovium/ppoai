# PPO-CTDE-Belief-SePoT-P2SRO Runtime 重构清单

## 0. 大路线是否改变

**结论：没有改变主路线，但改变了 SePoT 的 runtime 实现路线。**

当前仍然坚持的大路线是：

- **PPO**：策略优化主训练框架
- **CTDE**：训练时可用更多信息，执行时仍按在线决策运行
- **Belief**：保留显式 / 近显式的 belief 建模，不回退成纯无记忆 policy
- **SePoT**：保留 selective planning / tactical search 这条路线
- **P2SRO**：保留 population / PSRO 式对手池与自博弈演化路线

**改变的不是算法总路线，而是 SePoT 在 runtime 里的实现方式。**

当前准备淘汰 / 判死刑的是：

- template JSON reconstruction scaffold
- 每个搜索分支都 `json.loads -> patch -> json.dumps -> env.reset(state_json=...)`
- posterior update 里按 hypothesis 反复 reconstruct/reset 的 runtime 实现路径

保留并继续推进的是：

- root-heavy selective expansion
- lazy / factorized / partial belief propagation
- tactical-prioritized branching
- search/value head 在 runtime 稳定后再继续训练蒸馏

---

## 1. 当前判断摘要

### 已确认事实

- `depth / root_top_k / belief_samples` 已经明显压小，但 `fallback_rate` 仍接近 95%。
- 当前主问题已不是“树太宽 / 太深”，而是**单分支固定成本过高**。
- 当前代码里最重的固定成本来自 runtime SePoT 对 **template reconstruction** 的依赖。参见 `PublicStateReconstructor.reconstruct()` 以及 `PublicBeliefUpdater._advance_opponent_range()` 中的 reconstruct/reset 路径。
- 当前主路径里仍有两个方向性问题：
  - `choose_action()` 还保留 `_should_search_for_policy(...)` 这层 policy veto
  - `choose_action()` 主路径仍调用 `_select_root_candidate_indices(...)`，没有真正接上 `_prioritized_root_indices(...)`
- 当前项目里确实存在需要训练的 search head / search value head，但这主要影响**搜索强度**，不是当前这种接近 95% 的 runtime fallback。

### 因此得到的结论

- **不再继续把“调 timeout / 调 top-k / 调 depth”当主路线。**
- **正式进入：runtime 去 template 化 + lazy belief 重构。**

---

## 2. 总体阶段路线图

| Phase | 名称 | 目标 | 是否必须 | 当前状态 |
|---|---|---|---|---|
| 0 | 观测修正 | 修正脏统计，避免误判 | 必须 | 未做 |
| 1 | 主路径纠偏 | 去掉 policy veto，接上 prioritized root | 必须 | 未做 |
| 2 | runtime 去 template 化 | 去掉 template reconstruction 主依赖 | 必须 | 未做 |
| 3 | lazy / factorized belief | 把 posterior update 变成真正可运行 | 必须 | 未做 |
| 4 | 限制性 combo 扩展 | 处理一部分同回合多步组合 | 可选但建议 | 未做 |
| 5 | runtime 稳定后重训 | 重新蒸馏 search heads | 必须（后置） | 未做 |

---

## 3. Phase 0：观测修正

### 目标

把现在的统计修干净，避免继续被假指标带偏。

### 要做的事

- [ ] 修正 fallback 时的 `root_candidate_count` 统计
  - 不再默认把 `len(transformed_policy)` 记成 `root_candidate_count`
  - 区分：真实 root 候选数 vs fallback 时 legal action 数
- [ ] 给 fallback reason 做明确分桶
  - [ ] `template_unavailable`
  - [ ] `reconstruction_failed`
  - [ ] `timeout_before_root_scores`
  - [ ] `timeout_during_posterior`
  - [ ] `timeout_during_root_reconstruct`
  - [ ] `no_root_candidates`
- [ ] 给 root 搜索的各阶段打 timing
  - [ ] trigger 判定耗时
  - [ ] root candidate 生成耗时
  - [ ] root branch reconstruct 耗时
  - [ ] posterior update 耗时
  - [ ] leaf value 耗时
- [ ] 区分 success-only 和 fallback-only 的统计
  - [ ] `average_root_k_success_only`
  - [ ] `average_root_k_fallback_only`
  - [ ] `average_latency_success_only`
  - [ ] `average_latency_fallback_only`

### 为什么这样改

- 当前 `average_root_k` 被 fallback 污染，已经不能直接解释为“真实搜索展开宽度”。
- 不先修观测，后面任何“优化成功 / 失败”的判断都可能是错的。

### 改完前的问题

- 看不清到底死在 root 前、posterior 中、还是 template 重建本身。
- 很容易误以为是 root 太宽，实际上可能是 fallback 统计脏了。

### 改完后的收益

- 指标第一次变得可解释。
- 能精准识别下一刀应该砍哪里。

### 风险 / 注意事项

- [ ] 日志不要过量，最好有 summary 汇总
- [ ] 输出字段名固定，避免后续脚本反复重写

### 验收标准

- [ ] 能看见 success/fallback 分离后的 root_k
- [ ] 能看见 fallback_reason 分布
- [ ] 能看见阶段耗时分解

---

## 4. Phase 1：主路径纠偏（不做大重构）

### 目标

先把方向错误的地方纠正，不让 policy 继续压过 search。

### 要做的事

- [ ] 去掉或弱化 `_should_search_for_policy(...)` 的 veto 作用
  - 触发搜索后，不允许 policy 直接否决 tactical search
- [ ] `choose_action()` 主路径改为使用 `_prioritized_root_indices(...)`
  - 不再继续用 `_select_root_candidate_indices(...)`
- [ ] 在 prioritized root 上加 semantic dedup
  - 同语义但槽位/副本等价的动作不重复占 root 配额
- [ ] tactical pool 和 prior pool 并列，不从属
  - [ ] tactical 候选保底
  - [ ] prior 只做排序 / 辅助裁剪

### 为什么这样改

- 当前主路径仍然是 policy gate + 旧 root 选择。
- 这会把 SePoT 拉回“policy 主导、search 辅助”的错误关系。

### 为什么不跳过这一步

- 因为这些逻辑你已经部分写进文件里了，只是没真正接到主路径。
- 先接上，才能避免后面大重构后仍然沿着错误关系运行。

### 改完前的问题

- tactical state 可能被 policy 提前 veto
- prioritized branching 逻辑写了但没生效

### 改完后的收益

- search 和 policy 的分工更符合论文式 prior + planning
- tactical state 真正能进入搜索
- root 预算不再被语义等价动作浪费

### 风险 / 注意事项

- [ ] trigger_rate 可能变化，不要把它当核心指标
- [ ] 要重点看 `search_acted` 和 tactical 样本质量

### 验收标准

- [ ] `search_acted` 样本数增加
- [ ] tactical 局面不再明显漏搜
- [ ] semantic 重复 root 候选减少

---

## 5. Phase 2：runtime 去 template 化（主轴）

### 目标

正式淘汰 template reconstruction 这条 runtime 实现路线。

### 要做的事

- [ ] root 分支不再从 template scaffold 重建
- [ ] 引入显式 `BranchState`
  - [ ] `current_context`
  - [ ] `current_sampled_state_json`
  - [ ] `current_public_belief`
  - [ ] `current_tracker`
  - [ ] `sampled_self_hypothesis`
  - [ ] `sampled_opponent_hypothesis`
  - [ ] `history_tail`
- [ ] root candidate 从真实 root state 直接起步
- [ ] rollout 沿 `BranchState` 继续推进，不再回模板
- [ ] 尽可能减少 `env.reset(state_json=...)` 次数

### 为什么这样改

- 现在树已经砍小了，fallback 还是接近 95%，说明主要不是树的规模问题。
- template reconstruction 是当前单分支固定成本的核心来源。

### 为什么不继续靠调参

- 调参只是在减少“要跑多少次重建”，不是减少“单次重建有多重”。
- 这条路已经试过很多轮，证据足够了。

### 改完前的问题

- root 搜索还没真正开始，就先把预算烧在 state rebuild 上

### 改完后的收益

- root path 固定成本显著下降
- timeout 主因有机会从“reconstruct 太贵”转移到“真正搜索本身”

### 风险 / 注意事项

- [ ] env 是否有可靠 clone / reset 语义要核清楚
- [ ] branch state 必须始终对齐 sampled hidden world
- [ ] 不能混回 live root context 的 hidden truth

### 验收标准

- [ ] `timeout_before_root_scores` 显著下降
- [ ] `average_latency_ms` 不恶化
- [ ] `search_acted` 明显上升

---

## 6. Phase 3：lazy / factorized / partial belief

### 目标

让 belief update 真正能在 runtime 活下来，而不是 nominal lazy。

### 要做的事

- [ ] 默认不做 full posterior advance
- [ ] 引入三层 update 策略
  - [ ] Keep-sampled
  - [ ] Partial update（只更公开可观测因子）
  - [ ] Exact-ish update（仅在必要时）
- [ ] 扩充 transition / posterior cache
  - [ ] `pre_public_belief key`
  - [ ] `semantic action key`
  - [ ] `sampled hypotheses key`
  - [ ] `observation delta`
- [ ] `_advance_opponent_range()` 不再对所有 hypotheses reconstruct/reset

### 为什么这样改

- 在线 POMDP 规划真正贵的通常是 posterior update，不是 leaf value。
- belief 近似是 runtime 可行性的必要条件，不是可有可无的优化。

### 为什么不做 full exact posterior

- runtime 预算承受不了
- 会继续把系统拖回几乎全 fallback

### 改完前的问题

- 就算 root 去了 template，posterior 也可能继续吃掉大部分预算

### 改完后的收益

- belief sample 小时，成功率会显著更可提升
- search 才真正像 selective planning，而不是“半重建器”

### 风险 / 注意事项

- [ ] belief 近似会带来 value 偏差
- [ ] 需要观察强度损失是否可接受
- [ ] cache 键必须足够稳定，避免错复用

### 验收标准

- [ ] fallback rate 从接近 95% 显著下降
- [ ] belief sample 不再一升就炸

---

## 7. Phase 4：限制性 combo 扩展（处理同回合多步）

### 目标

解决一部分“先 A 再 B / 先 B 再 A”有差异的组合问题，但不把整回合全展开。

### 当前现状

- 当前 runtime 是 root-heavy
- root 玩家再次行动时，多数情况直接 collapse 到 leaf
- 所以一部分同回合顺序敏感组合主要靠 leaf 近似

### 要做的事

- [ ] 只在满足条件时，多展开一步 self action
  - [ ] root 玩家重新拿到行动权
  - [ ] 第一步显著改变资源 / 目标状态
  - [ ] 语义上属于高 swing 动作
- [ ] 第二步只看 very small prioritized set
- [ ] 不展开第三步
- [ ] 对顺序等价动作继续 semantic dedup

### 为什么这样改

- 你担心的是 order-sensitive combo，不是要把整回合所有排列都搜完。
- 所以应该做“一步补洞”，不是“全回合扩树”。

### 为什么不全展开

- runtime 会重新炸掉
- 前面所有降成本工作会被抵消

### 改完前的问题

- 同回合多步组合主要靠 leaf 猜

### 改完后的收益

- 一部分关键顺序差异能被显式看到
- 战术样本质量更合理

### 风险 / 注意事项

- [ ] 触发条件若过宽，会重新引爆 runtime
- [ ] 必须保持为“一步限制性扩展”

### 验收标准

- [ ] combo 样本质量提高
- [ ] runtime 不明显回炸

---

## 8. Phase 5：runtime 稳定后重训

### 目标

在 runtime 结构稳定后，再重新训练 search heads / search value。

### 要做的事

- [ ] 重新生成适配新 runtime 的 search teacher 分布
- [ ] 重新训练 / 蒸馏
  - [ ] `search_option_logits`
  - [ ] `search_state_value`
- [ ] 重看 search distill loss 和 runtime 指标联动

### 为什么这样改

- 当前训练确实已经在学 search-related heads
- 但这些 heads 现在学的是旧 runtime 分布
- runtime 结构一变，最好重训

### 为什么不现在就重训

- runtime fallback 还接近 95% 时，teacher / runtime 分布都还不稳定
- 现在重训只会把不稳定行为蒸馏进去

### 改完前的问题

- search head 学的是旧实现分布

### 改完后的收益

- 新 runtime 才可能真正获得强度提升，而不是只有可运行性提升

### 风险 / 注意事项

- [ ] 训练指标短期可能抖动
- [ ] 不要把训练抖动误判成路线错误

### 验收标准

- [ ] runtime 成功率先稳定
- [ ] 重训后再看强度是否提升

---

## 9. 明确不再作为主线的路线

- [ ] 不再把“单纯调高 timeout”当主线
- [ ] 不再把“继续砍 depth / root_top_k / belief_samples”当主线
- [ ] 不再让 policy 增加对 search 的 veto 权
- [ ] 不做“全回合 combo 全展开”

---

## 10. 风险总表

### 技术风险

- [ ] env clone / reset 语义不够稳，影响 branch-state 设计
- [ ] branch-state 和 sampled hidden world 可能错位
- [ ] cache 键设计不稳，导致错复用
- [ ] lazy belief 近似可能带来价值偏差

### 项目风险

- [ ] 改到一半 runtime 结构变了，但统计和训练代码没同步修
- [ ] 误把 trigger_rate 当主目标，而不是 success / fallback / quality
- [ ] 过早重训，浪费算力和时间

---

## 11. 建议实施顺序（严格按顺序）

1. [ ] Phase 0：观测修正
2. [ ] Phase 1：主路径纠偏
3. [ ] Phase 2：runtime 去 template 化
4. [ ] Phase 3：lazy / factorized belief
5. [ ] Phase 4：限制性 combo 扩展
6. [ ] Phase 5：runtime 稳定后重训

**不要跳步，不要把 2/3/4/5 混成一个超大补丁。**

---

## 12. 最终目标定义

### 第一目标：先活

- [ ] fallback 不再接近 95%
- [ ] success 样本明显增加
- [ ] 统计可解释

### 第二目标：再稳

- [ ] runtime 延迟可控
- [ ] tactical state 能稳定进入搜索
- [ ] belief update 不再成为主要瓶颈

### 第三目标：后强

- [ ] 重新训练 search heads 后，搜索不只是“能跑”，而是“有实际强度收益”

---

## 13. 当前版本的项目立场（供后续核对）

**一句话版：**

> 我们没有放弃 PPO-CTDE-Belief-SePoT-P2SRO 这条总路线；
> 我们放弃的是 SePoT 在 runtime 中依赖 template reconstruction 的那条实现路径。

