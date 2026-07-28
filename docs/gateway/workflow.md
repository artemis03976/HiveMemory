---
title: Gateway Workflow
status: current
owner: gateway
scope: fixed-workflow-state-control-and-fallbacks
code_paths:
  - src/hivememory/gateway/service.py
  - src/hivememory/gateway/workflow/
  - src/hivememory/gateway/runtime/core.py
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
  - docs/contracts/error-model.md
last_reviewed: 2026-07-28
---

# Gateway 固定工作流

Gateway workflow 解决的不是“如何自由编排任意 Agent 任务”，而是“如何把每条入口消息可靠地收敛为唯一、完整、可验证的决策终态”。因此它选择固定拓扑、显式分支和单一状态提交点，而不是动态 DAG、隐式中间件链或由 LLM 决定下一步。

这个取舍来自旧一步 router 的教训：当模型在一次返回中同时决定话题、意图、关键词和记忆价值时，系统看似少了一些步骤，却失去了局部超时、独立降级、边界验证和错误定位。固定 workflow 把不确定能力放进受控 Step，让流程本身保持确定。

## 1. 装配与所有权

`GatewayRuntime` 在启动时装配：

```text
GatewayContextProvider
RuleInterceptor + CommandRegistry
SystemCommandDispatcher
TopicRouterEngine
UserQueryAnalysisResolver
        |
        v
build_gateway_workflow()
        |
        v
GatewayWorkflow
```

`GatewayWorkflow` 是请求级执行协调者，`GatewayExecutionState` 只由它持有。Engine、Provider、Resolver 和 Dispatcher 可以替换，但不能改变公共终态结构；拓扑若发生变化，应显式修改 `build_gateway_workflow()` 和当前文档，而不是让某个 Engine 私下跳转到另一步。

`GatewayService.process()` 会先收敛请求 timeout：调用方可以给出更短的 `request_timeout_ms`，但不能用它扩大配置中的 `default_request_timeout_ms`。随后 Service 将消息、`Identity`、入口模式、取消信号和有效 deadline 交给 workflow。

## 2. 当前固定拓扑

```text
entry_interception
  |
  +-- system_command --> command_dispatch --> finalize command outcome
  |
  `-- decision -------> candidate_topics_preparation
                         -> topic_routing
                         -> routed_topic_preparation
                         -> simple_chat_defaults
                            或 user_query_analysis
                         -> finalize decision outcome
```

### 2.1 入口分支

`entry_interception` 执行低成本确定性判断。在 `ACTIVE_CHAT` 中，它可以把 slash command 标记为 `system_command`；在 `PASSIVE_MEMORY` 中，`allow_system=false`，因此 command dispatch 在结构上不可达。

简单寒暄也可以被 L1 规则命中，但它不会提前结束整个 decision flow。Gateway 仍准备候选话题并完成话题路由，只在最后以 `simple_chat_defaults` 代替 LLM 查询分析。这保持了话题连续性，同时避免为明确的寒暄调用查询分析模型。

### 2.2 Decision 前缀

Decision 分支固定经过三个步骤：

1. `candidate_topics_preparation` 通过 Patchouli 公共 route 读取轻量 `TopicSnapshot`；
2. `topic_routing` 只选择候选话题或 `NEW_TOPIC`；
3. `routed_topic_preparation` 仅在选中已有话题后读取完整 `TopicData`。

两阶段准备避免为了做候选选择而加载所有话题全文，也避免 Query Analysis 自行再查一套上下文。它牺牲了一次潜在 route round trip，换取了更小的模型输入、更明确的数据所有权和可独立降级的边界。

## 3. Step 是一次受控提交

每个 `GatewayWorkflowStep` 都遵循相同协议：

```text
state.snapshot()
  -> select_input(snapshot)
  -> invoke(typed input)
  -> project(output)
  -> GatewayStepResult
  -> state._apply_step_result()
```

这个结构把一次外部调用分为“读什么、调用什么、写什么”三个可审查阶段。

- `select_input` 只能读取当前只读快照，不能直接修改 execution state；
- `invoke` 接收类型化输入，不得到整个可变状态；
- `project` 把能力输出转换为允许提交的字段；
- `_apply_step_result()` 是唯一写入口，先校验字段集合，再提交结果。

`raw_message`、`identity` 和 `ingress_mode` 是初始化字段，任何 Step 都不能覆盖。未知字段会被拒绝；`flow_end_reason` 只能设置一次；状态完成后不允许继续提交。这不是数据库事务或跨子系统 rollback，而是请求内的单写入口：它防止 Engine 在调用途中留下半份 Gateway 状态。

`GatewayStepResult.is_fallback` 与 `fallback_reason` 只用于观测本次提交，不进入公共 `GatewayDecision`。下游不应通过猜测 fallback 原因改变业务行为。

## 4. 终态投影

内部 state 不会直接泄露给 System 或 transport。只有 `finalize()` 能构造公共结果：

- `system_command` 分支必须来自 `ACTIVE_CHAT`，必须已有 `CommandExecutionResult`，且不能包含查询分析；
- decision 分支不得包含 command result，必须已有 `topic_id` 和完整 `UserQueryAnalysisResult`；
- 成功构造后 state 标记为 `completed`，再次 finalize 或提交都会失败。

最终结果只有：

```text
GatewayCommandOutcome
  -> command_execution_result

GatewayDecisionOutcome
  -> target_topic_id
  -> new_topic_title / new_topic_summary
  -> rewritten_query / search_keywords
  -> intent_type / memory_write_signal
  -> retrieval_plan
```

这一投影刻意不包含候选话题、完整 `TopicData`、L1 命中原因、Step 顺序、fallback 原因或 trace。它们是 Gateway 私有执行事实，不是跨子系统契约。

## 5. 取消与 deadline

请求控制遵循两个原则：取消优先传播，deadline 限制整条链路。

`cancel_event` 在 Step 开始前和能力调用等待期间都会检查。取消发生时，当前 invocation task 被取消并等待收尾，workflow 抛出 `GatewayCancelledError`。取消不是能力退化，因此不能转为本地 fallback 或普通 decision。

请求 deadline 从 workflow 开始时计算。每个 Step 的实际等待时间是“Step timeout”与“整次请求剩余时间”的较小值。若整次 deadline 在某个可降级 Step 中耗尽，该 Step 先提交 fallback，后续带 fallback 的 Step 不再调用能力，而是继续提交保守默认值，直到形成完整 decision；若当前 Step 没有 fallback，则抛出 `GatewayTimeoutError`。

因此，“整次超时”并不总等于“没有结果”。Gateway 只在每个剩余字段都有安全默认值时继续完成；command dispatch、入口不变量或没有 fallback 的能力不能被伪装成成功。

## 6. 当前 fallback 矩阵

| Step | 可恢复失败后的提交 | 设计理由 |
|:---|:---|:---|
| `entry_interception` | 无 fallback | 入口分类失败会影响分支合法性，不能猜测 |
| `command_dispatch` | 无 workflow fallback | Dispatcher 自身把解析、权限和 handler 失败投影为结构化命令终态 |
| `candidate_topics_preparation` | 空 `CandidateTopics` | 仍可创建新话题，不伪造已有话题 |
| `topic_routing` | `NEW_TOPIC`，标题/摘要为空 | 选错已有话题比新建话题风险更高 |
| `routed_topic_preparation` | `None` | 查询分析可以在缺少历史上下文时继续 |
| `simple_chat_defaults` | 原文、`CHAT`、写入 `SKIP`、检索 `SKIP` | 寒暄不值得因分析能力失败阻塞入口 |
| `user_query_analysis` | 原文、`RAG`、写入 `UNKNOWN`、`HYBRID` 和默认 `top_k` | 保留检索机会，不把未知误判为明确写入或跳过 |

只有 `TimeoutError` 与 `RecoverableGatewayError` 会触发 Step fallback。类型错误、输入选择错误、投影错误、未知状态字段和 finalize 不变量失败都会原样失败。这个边界非常重要：fallback 用来隔离能力不可用，不用来掩盖代码或契约错误。

## 7. 观测不是控制面

Workflow 发布 started、step completed、completed、cancelled 和 failed RuntimeEvent。Step 事件会记录 `step_id`、序号、耗时、是否 fallback 和原因；终态事件记录 outcome kind 与总耗时。

RuntimeEvent sink 失败不得改变 Gateway 结果。观测字段也不进入公共 outcome。关于 RuntimeEvent 的统一边界见 [System 可观测性](../system/observability.md)。

## 8. 当前限制

- workflow 是进程内一次性执行，不持久化中间状态，也不能在重启后恢复；
- 当前没有 Step retry、熔断、并行分支或动态插件拓扑；
- fallback 不会回滚已经完成的外部只读调用，command 副作用也没有跨子系统事务；
- `NEW_TOPIC` fallback 可以没有标题和摘要，下游必须把它视为保守路由，而不是完整的模型生成元数据；
- 取消依赖被调用协程正确响应 asyncio cancellation，不能保证终止外部提供商已经接收的请求；
- RuntimeEvent 可以解释发生了 fallback，但公共调用方不会直接得到逐 Step 诊断。

这些限制与未来 Job Queue、动态任务编排不是同一问题。只有当 Gateway 入口决策本身出现稳定的并行需求时，才应扩展当前拓扑；不能为了未来 Agent 任务图而提前把 Gateway 变成通用工作流引擎。

## 9. 设计矛盾检查

修改 workflow 时检查：

1. 新 Step 是否真的属于入口决策，而不是 Patchouli/Alice/System 的执行职责？
2. 是否绕过 `_apply_step_result()` 直接修改 execution state？
3. 是否让 Engine 读取完整可变 state，或让下游依赖私有 snapshot？
4. 是否把普通编程错误也转成了 fallback，从而掩盖契约违约？
5. 是否让调用方 timeout 大于系统默认值，破坏运行预算？
6. 是否在 `PASSIVE_MEMORY` 中重新开放 command 分支？
7. 是否在取消后继续提交结果，或把 RuntimeEvent 当成控制信号？
8. 是否新增了无法在 `finalize()` 中证明完整性的第三种半终态？

## 10. 验证入口

- `tests/unit/gateway/test_phase3a_system.py`
- `tests/unit/gateway/test_phase3b_contracts.py`
- `tests/unit/gateway/test_phase3c_workflow.py`
- `tests/unit/gateway/test_phase3d_context_provider.py`
- `tests/unit/gateway/test_phase3f_request_control.py`
- `tests/unit/system/application/test_gateway_chat_flow.py`
- `tests/unit/system/services/passive/test_passive_gateway_mode.py`
