---
title: Gateway Analysis
status: current
owner: gateway
scope: topic-routing-and-user-query-analysis
code_paths:
  - src/hivememory/gateway/analysis/
  - src/hivememory/gateway/context.py
  - src/hivememory/gateway/topic_context.py
  - src/hivememory/engines/gateway/
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/architecture/boundaries.md
last_reviewed: 2026-07-29
---

# Gateway 话题与查询分析

Gateway 的分析层要回答两个不同的问题：这条消息属于哪段对话连续性，以及下游应如何理解和检索它。旧一步 LLM router 把两者连同记忆价值判断塞进一次输出，看似能够“一次计算，多处复用”，实际却把多个失败域绑定在一起：话题选择错误会污染 query rewrite，某个字段解析失败也可能让整份决策不可用。

当前设计保留共享上下文和单次提交，但拆开能力边界：Topic Router 只选择话题；User Query Analysis 在话题已确定后统一生成意图、重写、关键词、记忆信号和检索计划。拆分不是追求 Engine 数量，而是让不同问题拥有独立输入、不变量和 fallback。

## 1. 分析链路

```text
raw message + identity
  -> L1 entry interception
  -> candidate TopicSnapshot list
  -> Topic Router
  -> selected TopicData（已有话题时）
  -> simple chat defaults
     或 UserQueryAnalysisResolver
  -> GatewayDecision
```

L1 只做确定性快速判断。系统命令属于[命令文档](./commands.md)；简单寒暄命中后，workflow 跳过 Query Understanding，直接使用 `CHAT/SKIP/SKIP` 默认值。普通消息进入完整分析。

## 2. 两阶段话题上下文

Gateway 不拥有话题数据。`GlobalBusGatewayContextProvider` 通过 Patchouli 公共 route 分两次读取：

1. `prepare_candidate_topics()` 获取不可变 `TopicSnapshot` tuple，用于候选菜单和路由；
2. `prepare_routed_topic()` 只为最终选中的已有话题获取 `TopicData`；`NEW_TOPIC` 直接返回 `None`。

Provider 负责读取和类型验证，不做决策。Patchouli 若返回非 tuple、非 `TopicSnapshot` 或非 `TopicData`，属于契约违约并直接失败；route 不可用等能力失败则转换为 `RecoverableGatewayError`，由 workflow 决定是否降级。

这种设计避免 Gateway 缓存第二份权威话题状态，也避免把所有活跃话题的完整 transcript 送入模型。代价是路由后可能多一次读取，但失败可以局部降级为“没有 routed topic context”，而不必推翻已经完成的话题选择。

## 3. Topic Router：只做选择

`TopicRouterEngine` 接收原始消息和候选 `TopicSnapshot`，只允许返回：

- 候选列表中的一个 `topic_id`；
- `NEW_TOPIC`，并在模型成功响应时同时提供非空 title 与 summary。

模型返回候选列表外 ID、非 object、缺少 target 或 `NEW_TOPIC` 缺少标题/摘要时，Engine 抛出 `TopicRouterError`。调用和解析异常也统一转为该预期能力错误，再由 workflow 作为 recoverable failure 降级到 `NEW_TOPIC`。

当 Topic Router 配置为禁用时，Engine 不调用模型并返回默认 `NEW_TOPIC`。当它启用但没有 LLM service 时，Engine 报告能力不可用，workflow 同样降级。两条路径的业务结果接近，但观测原因不同。

Topic Router 不生成关键词、不判断意图、不读完整 `TopicData`，也不决定是否检索。否则它会重新退化为旧的一步 router。

## 4. User Query Analysis 的稳定边界

`UserQueryAnalysisResolver` 接收冻结的 `UserQueryAnalysisContext`：原始消息、身份、候选话题、路由结果以及可选的 routed `TopicData`。它必须一次性返回完整的 `UserQueryAnalysisResult`：

```text
intent_type
rewritten_query
search_keywords
memory_write_signal
retrieval_plan
```

这个结果是“不可拆分提交”的原因不是所有字段必须由同一个模型产生，而是 Gateway state 不能先写入一半分析、再让下游观察到中间态。Resolver 内部将来可以替换 Engine 或改变并发方式，只要保持输出语义。

### 4.1 第一代 Resolver

当前 LLM Resolver 分三层工作：

```text
Tier 0：本地规则
  -> 显式“记住 / 以后 / remember / from now on” => WRITE + WRITE
  -> 与当前话题最近一条 user_query 重复 => memory signal SKIP

Tier 1：一次 QueryUnderstanding LLM 调用
  -> intent_type
  -> rewritten_query
  -> search_keywords
  -> memory_write_signal
  -> 私有 sub_intents

Tier 2：纯函数派生 retrieval plan
  -> CHAT / WRITE => SKIP
  -> 无关键词 => DENSE
  -> 有关键词 => HYBRID
```

Tier 0 只覆盖低成本、高确定性的规则，并在与模型结果冲突时覆盖对应字段。重复检测会移除空白并忽略大小写，但只比较 routed topic 最近一个 block 的用户输入。

Tier 1 的 `QueryUnderstandingEngine` 共享一次 LLM 调用，同时产生意图、重写、关键词和记忆初判。这是第一代实现的成本/延迟取舍，不是永久规定。Engine 还能解析 `sub_intents`，但 Resolver 当前没有把它投影到公共 `UserQueryAnalysisResult`，因此没有下游消费者，也不能被描述为已支持复合意图执行。

Tier 2 不再调用模型。检索模式是 Gateway 根据已提交分析派生的提示，不代表 Patchouli 已经执行检索。

## 5. 三种保守默认值不能混为一谈

当前代码存在三条不同的非模型路径：

| 场景 | Intent | 记忆信号 | Retrieval | 说明 |
|:---|:---|:---|:---|:---|
| L1 简单寒暄 | `CHAT` | `SKIP` | `SKIP`, `top_k=0` | 明确规则命中 |
| Runtime 未装配 LLM 或分析被禁用 | `RAG` | `WRITE` | `HYBRID` | `FallbackUserQueryAnalysisResolver` 的配置级保守结果 |
| LLM Resolver 运行中超时/可恢复失败 | `RAG` | `UNKNOWN` | `HYBRID` | workflow 的 Step fallback |

第二、三行对 `memory_write_signal` 的取值不同。这是当前实现事实，不应在文档中抹平：装配级 fallback 倾向保留写入机会，运行时失败则明确表示未知。下游必须按枚举语义处理，不能把所有 fallback 都当成 `WRITE`。

这处不对称也属于后续需要评估的设计矛盾。如果没有产品数据证明“无 LLM 时默认 WRITE”更合理，应统一语义或至少让配置明确表达策略；在改变前，当前文档不替代码虚构统一行为。

## 6. `GatewayDecision` 的含义

- `target_topic_id` 是话题路由建议，话题实体仍由 Patchouli 管理；
- `rewritten_query` 与 `search_keywords` 是检索输入，不是新的用户原文；
- `intent_type` 是入口主意图，不是 Alice 任务图；
- `memory_write_signal` 是用户输入阶段的初判，不是持久化命令；
- `retrieval_plan` 是依赖中立的模式、预算和权重提示，不包含具体 Retriever 或索引实现。

`RetrievalPlan` 当前可以携带 dense/sparse weight，但下游对权重的消费仍有限。不能因为字段存在，就宣称稀疏/稠密融合在所有链路中都严格按该权重执行。

## 7. 第一代实现的技术债

### 7.1 记忆价值判断时机过早

Resolver 只看到用户输入和已有话题上下文，看不到 Alice 的最终回答、MTP WRITE/UPDATE、工具结果或 Patchouli finalize 证据。它最多能判断“这条输入看起来是否值得记录”，不能决定完整 turn 的长期价值。最终记忆生成必须继续由 Patchouli 在更丰富证据下完成。

### 7.2 共享调用需要数据再拆分

一次 Query Understanding 调用可能产生字段耦合，但直接拆成多个串行 Engine 也会增加延迟、成本和新的不一致。合理的演化依据应是能力级耗时、解析失败率和字段质量观测，而不是仅凭职责名称就预设固定 Engine 数量。

### 7.3 规则仍是硬编码启发式

WRITE 表达式和简单寒暄正则直接写在代码中，语言覆盖有限；自定义入口规则尚没有完整配置与热注入路径。规则应保持窄而确定，不能不断扩张为另一套难以解释的自然语言分类器。

### 7.4 复合意图尚未形成公共能力

私有 `sub_intents` 目前在 Resolver 投影时丢弃，`COMPOSITE` 也没有稳定的下游任务/决策契约。相关设想仍属于计划，不得从 Engine 私有字段反推当前已支持复合任务执行。

### 7.5 未消费字段与配置边界仍需收敛

`RetrievalPlan` 已经携带 mode、top-k 与 dense/sparse weight，但 Patchouli prepare 当前主要消费是否跳过和 top-k，具体融合权重仍由 Retrieval 自身配置决定。`UserQueryAnalysisConfig` 也同时承载 Resolver 开关、模型覆盖和 context 截断等实现细节，尚未形成“稳定公共策略”与“Resolver 私有参数”的清晰分层。

这些问题不能通过继续增加 DTO 字段解决。第二代调整前应先观察 rewrite 重复率、keywords 质量、memory signal 分布/实际物化结果和 `COMPOSITE` 样本，再决定删除无消费者字段、下沉私有配置、扩展 Patchouli 消费，或拆分独立能力。观测只保存计数、耗时和脱敏摘要，不应把原始用户输入塞进 RuntimeEvent。复合意图的协议与样本门禁见 [v0.6.0 复合意图分解计划](../plans/v0.6.0-composite-intent-decomposition.md)。

## 8. 设计矛盾检查

修改分析能力时检查：

1. Topic Router 是否开始输出意图、关键词或记忆判断，重新制造一步 router？
2. Query Analysis 是否绕过 Context Provider 自行读取 Patchouli 私有状态？
3. Engine 新字段是否真的进入公共 Resolver 结果并有下游消费者？
4. 是否把 `memory_write_signal` 写成最终持久化决定？
5. 是否把模型解析错误、候选越界或类型违约静默改成任意默认值？
6. 是否未经数据验证就把共享调用拆成更慢的串行调用？
7. 是否把配置级 fallback 与运行时失败 fallback 描述成相同语义？
8. 是否把 RetrievalPlan 的存在误写为具体检索实现已经完全消费？
9. 新字段是否有明确消费者和所有者，还是只扩大了第一代共享调用的私有输出？
10. 第二代方案是否有可复现样本和质量指标支撑，而不是只按类名拆分 Engine？

## 9. 验证入口

- `tests/unit/gateway/test_phase3d_context_provider.py`
- `tests/unit/gateway/test_phase3e_analysis_resolver.py`
- `tests/unit/engines/gateway/test_topic_router.py`
- `tests/unit/engines/gateway/test_query_understanding.py`
- `tests/unit/engines/gateway/test_interceptors.py`
- `tests/unit/patchouli/test_phase3f_gateway_decision.py`
