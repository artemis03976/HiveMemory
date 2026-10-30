---
title: Gateway Analysis
status: current
owner: gateway
scope: topic-routing-and-user-query-analysis
code_paths:
  - src/hivememory/gateway/analysis/
  - src/hivememory/gateway/context.py
  - src/hivememory/gateway/topic_context.py
  - src/hivememory/gateway/workflow/topology.py
  - src/hivememory/engines/gateway/
  - src/hivememory/prompts/gateway.py
  - src/hivememory/infrastructure/llm/base.py
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/architecture/boundaries.md
related_docs:
  - docs/workspace/README.md
last_reviewed: 2026-10-30
---

# Gateway 话题与查询分析

Gateway 的分析层要回答两个不同的问题：这条消息属于哪段对话连续性，以及下游应如何理解和检索它。旧一步 LLM router 把两者连同记忆价值判断塞进一次输出，看似能够“一次计算，多处复用”，实际却把多个失败域绑定在一起：话题选择错误会污染 query rewrite，某个字段解析失败也可能让整份决策不可用。

当前设计保留共享上下文和单次提交，但拆开能力边界：Topic Router 只选择话题；User Query Analysis 在话题已确定后统一生成意图、重写、关键词、记忆信号和检索计划。拆分不是追求 Engine 数量，而是让不同问题拥有独立输入、不变量和 fallback。

## 1. 分析链路

```text
raw message + IdentityScope
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

Provider 负责读取和类型验证，不做决策。Patchouli 若返回非 tuple、非 `TopicSnapshot` 或非 `TopicData`，属于契约违约并直接失败；中立存储/资源不可用才转换为 `RecoverableGatewayError`，由 workflow 决定是否降级。缺失 route、授权拒绝和 handler 程序错误原样传播，不能当作没有话题。

这种设计避免 Gateway 缓存第二份权威话题状态，也避免把所有活跃话题的完整 transcript 送入模型。代价是路由后可能多一次读取，但失败可以局部降级为“没有 routed topic context”，而不必推翻已经完成的话题选择。

调用方另传最近 `gateway.context_preparation.session_turn_limit` 轮（默认 3）的用户/助手文本，即 `session_turns`。两个调用方各自取得这些轮次：任务进程取当前会话的最近轮次，每轮附上会话记录的落位话题；[Import Bus](../system/import-bus.md) 取请求 `recent_turns` 的最后若干轮，只有最近一轮带外部轮次标识时，才附上它经 Patchouli 交互状态查询得到的真实落点（该轮已应用成功时才有）。`GatewaySessionTurn` 不携带工具事件、身份或权限；其中的 `topic_id` 只供话题路由失败时续接上一轮（第 3 节），不渲染进模型提示词。Topic 菜单和路由后辅助背景仍按既有路径读取，会话文本才是指代消解的对话来源。该输入不进入 `GatewayDecision` 或 InteractionPayload。

## 3. Topic Router：只做选择

`TopicRouterEngine` 接收原始消息、候选 `TopicSnapshot` 和会话 `session_turns`，只允许返回：

- 候选列表中的一个 `topic_id`；
- `NEW_TOPIC`，并在模型成功响应时同时提供非空 title 与 summary。

模型返回候选列表外 ID、非 object、缺少 target 或 `NEW_TOPIC` 缺少标题/摘要时，Engine 抛出 `TopicRouterError`。LLM 服务报告的提供商失败（`ModelProviderError`，含超窗与提供商没有返回文本）和模型 JSON 解析失败也转为该能力错误，再由 workflow 按下文的兜底落位；未知程序错误、投影失败与取消继续传播。QueryUnderstanding 使用同样的窄分类，不把全部 Exception 当作模型不可用；两个 Engine 都不识别提供商 SDK 的异常类型。

话题路由失败（能力错误、步骤超时或整次请求 deadline 在这一步耗尽）时，workflow 的兜底是续接上一轮：`session_turns` 最近一轮的 `topic_id` 仍在本次候选话题中时，落位到该话题，标题与摘要为空；会话的第一轮、上一轮落位未知、该话题已被驱逐或结算而不在候选中，或候选话题读取本身已降级为空时，按 `NEW_TOPIC`（标题与摘要为空）兜底。同一会话的相邻轮次大多仍在同一话题上，续接比每次失败都新建无标题话题归错的代价小；代价是用户恰在故障期间换话题时，这一轮会被归入旧话题。只认候选中的话题，保证兜底不会落到已经不存在的话题上。这一兜底在 Gateway 内实现，任务进程与 Import Bus 共用。

当 Topic Router 配置为禁用时，Engine 不调用模型，正常返回默认 `NEW_TOPIC`；这不是失败，不走兜底，也就不续接上一轮。当它启用但没有 LLM 服务或 Gateway 请求参数（组件未配置模型）时，Engine 报告能力不可用，workflow 走上述兜底。两条路径的观测原因不同，业务结果也可能不同。

Topic Router 不生成关键词、不判断意图、不读完整 `TopicData`，也不决定是否检索。否则它会重新退化为旧的一步 router。

## 4. User Query Analysis 的稳定边界

`UserQueryAnalysisResolver` 接收冻结的 `UserQueryAnalysisContext`：原始消息、已从完整 `IdentityScope` 派生的 `ActorIdentity`、候选话题、路由结果、可选的 routed `TopicData` 与会话 `session_turns`。候选话题与 `TopicData` 已由带 scope 的 Context Provider 取得，因此查询分析本身不重新承担 Workspace ownership 校验。它必须一次性返回完整的 `UserQueryAnalysisResult`：

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
  -> 与当前会话上一轮 user_message 重复 => memory signal SKIP

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

Tier 0 只覆盖低成本、高确定性的规则，并在与模型结果冲突时覆盖对应字段。重复检测会移除空白并忽略大小写，但只比较当前会话的最后一轮用户输入；没有会话轮次时不回退到 Topic。

Topic Router 与 QueryUnderstanding 的模型消息都只有 system 与当前输入两条：最近会话轮次与话题菜单一样渲染为编号列表（`prompts/gateway.py` 的 `render_session_turns`，每轮一个条目，`User:` 与 `Assistant:` 各占一行，多行内容续行缩进），放在 system 提示词的“当前会话最近轮次”段落中，并注明只用于理解指代与省略、不需要回答。会话轮次不作为对话消息发送：以用户/助手消息提供时，真实模型会接着对话回答用户而不是完成分析，JSON 输出随之为空，两项分析都会降级。消息经组合根注入的共享 LLM 服务以完整调用发出。请求参数来自由 `shared.llm.gateway` 转换的组件参数，两个 Engine 都追加 JSON 输出要求；配置了各自的 `model_override` 时只替换模型，凭证与采样参数沿用组件参数（[System 组合根](../system/composition.md#13-llm-bundle)）。Tier 1 的 `QueryUnderstandingEngine` 共享一次 LLM 调用，同时产生意图、重写、关键词和记忆初判。这是第一代实现的成本/延迟取舍，不是永久规定。Engine 还能解析 `sub_intents`，但 Resolver 当前没有把它投影到公共 `UserQueryAnalysisResult`，因此没有下游消费者，也不能被描述为已支持复合意图执行。

路由到已有话题时，QueryUnderstanding 的输入另带该话题的辅助背景：标题、话题摘要与状态摘要，以及最近若干个逻辑块（`context_block_limit`，默认 3）的用户消息与回复正文，各截断到 `context_text_limit` 字符（默认 200）。这两段文字由逻辑块从本轮事件读取，不另存；思考与工具过程不进入 Gateway 的提示词。

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

这些问题不能通过继续增加 DTO 字段解决。第二代调整前应先观察 rewrite 重复率、keywords 质量、memory signal 分布/实际物化结果和 `COMPOSITE` 样本，再决定删除无消费者字段、下沉私有配置、扩展 Patchouli 消费，或拆分独立能力。观测只保存计数、耗时和脱敏摘要，不应把原始用户输入塞进 RuntimeEvent。复合意图的协议与样本门禁见[复合意图分解 Idea](../ideas/composite-intent-decomposition.md)，该方向当前未排期。

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

- `tests/unit/gateway/test_context_provider.py`
- `tests/unit/gateway/test_workflow.py`（话题路由失败时续接上一轮）
- `tests/unit/gateway/test_analysis_resolver.py`
- `tests/unit/engines/gateway/test_topic_router.py`
- `tests/unit/engines/gateway/test_query_understanding.py`
- `tests/unit/engines/gateway/test_interceptors.py`
- `tests/unit/patchouli/test_service_gateway_decision.py`
