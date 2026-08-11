---
title: 复合意图分解计划
status: planned
owner: gateway
target: unscheduled
scope: composite-intent-contract-and-execution
code_paths:
  - src/hivememory/gateway/
  - src/hivememory/engines/gateway/
related_contracts:
  - docs/gateway/analysis.md
  - docs/gateway/workflow.md
  - docs/contracts/subsystem-contracts.md
last_reviewed: 2026-08-10
---

# 复合意图分解计划

**文档状态**: Planned  
**目标阶段**: Unscheduled  
**适用范围**: `gateway/`、`engines/gateway/intent_decomposer.py`、`patchouli/`、`alice/`
**前置条件**: 当前固定单主意图 workflow 与公共 `GatewayDecision` 保持稳定  
**核心目标**: 为复合意图分解、多分支消费和合并语义建立清晰边界。具体算法、Prompt、执行策略仍需在实现前以真实样本收敛。

---

## 1. 定位

当前第一代 Query Understanding 可以把主意图分类为 `COMPOSITE`，也会在 Engine 私有结果中解析 `sub_intents`；但 Resolver 不把 `sub_intents` 投影进公共 `UserQueryAnalysisResult`，固定 workflow 仍然只提交一个 `GatewayDecision`。代码中没有旧稿曾设想的 `CompositePlaceholder`、`is_composite` 或 `composite_deferred` 公共状态，也没有多分支执行、合并和持久化样本集。

这不是“分解已经实现但暂时关闭”，而是只有一个仍未被消费的分类信号。计划的作用是先固定复合 envelope、下游所有权和 fallback，再决定是否保留当前私有 `sub_intents`、建立独立 decomposer 或收集更多样本；不能为了临时支持复合输入而输出不稳定的 `list[GatewayState]`，也不能让 Patchouli、Alice 或尚未立项的后台执行机制被迫适配半成品分支结构。

本计划已经移出 v0.6.0 发布范围，当前不绑定具体版本。Phase C0 可以作为非阻塞研究推进；只有真实样本证明单主意图路径存在稳定缺口，且 envelope、消费所有权与 fallback 能够形成可验收契约后，才重新进入路线图排期。

---

## 2. 已确定设计决策

1. 复合意图识别与入口级分解属于 Gateway，但 branch 的业务执行仍由对应下游所有者完成。
2. 是否引入独立 `IntentDecomposerStep` 应由质量与延迟数据决定，不能从当前私有 `sub_intents` 直接反推固定实现结构。
3. 对应 engine 原语为 `IntentDecomposerEngine`，放在 `engines/gateway/intent_decomposer.py`。
4. `IntentDecomposer` 可以调用 LLM，但必须被固定 workflow 包裹，不能演化为自由 Agent Router。
5. `IntentDecomposer` 只负责边界识别、子意图类型、顺序、依赖、置信度，不做 topic routing、query rewrite、retrieval planning 或执行。
6. 多分支结果不能直接暴露为 `list[GatewayState]`。
7. 必须先定义 composite envelope 和下游消费协议，再实现 LLM 分解。
8. 低置信度、解析失败、合并失败、下游不支持时必须能回退到当前单主意图路径。
9. 系统指令 + 普通聊天/检索的混合输入需要稳定协议，不能只靠入口命令短路一刀切。
10. 新 Prompt 文本继续由 i18n 模板统一管理，JSON parsing / schema 解析复用项目已有通用工具。

---

## 3. 当前基线与样本门禁

当前基线是：Query Understanding 仍只调用一次 LLM；`COMPOSITE` 和私有 `sub_intents` 只参与本次分析，不进入公共决策；下游继续消费单一 `GatewayDecision`。在本计划形成稳定协议前，必须保持以下约束：

- 不把 Engine 私有 `sub_intents` 当作已提交的公共结果；
- 不返回 `list[GatewayState]`；
- 不触发多分支 Patchouli/Alice 或后台长任务调用；
- 不把没有持久化事实支撑的“deferred”字段添加到公共协议；
- `COMPOSITE` 仍按单主意图兼容路径执行，必要时由 Alice 在自然语言层处理。

实施前应建立可脱敏的 golden dataset 或等价评估材料，至少包含：

- raw message
- intent classifier 输出
- topic route
- routed topic context 摘要
- final single-intent fallback 行为
- 用户是否追问或修正

这些材料用于判断是否值得拆分独立 Engine，以及确定过度分解、漏分解和 fallback 的验收阈值。RuntimeEvent 可以记录计数、耗时和分类摘要，但不能把完整用户输入当成默认观测 payload。

---

## 4. 计划范围

本计划需要解决以下问题，但具体实现仍需由数据与协议评审收敛：

1. **IntentDecomposer**
   - TODO: 设计 prompt。
   - TODO: 设计置信度阈值。
   - TODO: 设计过度分解/漏分解评估样本。

2. **Composite Envelope**
   - TODO: 定义 `CompositeGatewayDecision`。
   - TODO: 定义 `parent_state`、`branches`、`merge_policy`、`fallback_policy`。

3. **下游消费协议**
   - TODO: 定义 Patchouli 如何消费多 QUERY/WRITE 分支。
   - TODO: 定义 Alice 如何消费需要协作或规划的分支。
   - `FUTURE_JOB` 分支保持 unsupported；只有真实长任务机制独立立项后才增加消费者。

4. **执行与合并语义**
   - TODO: 定义串行/并行策略。
   - TODO: 定义依赖关系。
   - TODO: 定义失败回退。
   - TODO: 定义多分支结果展示。

5. **系统指令混合输入**
   - TODO: 定义 `/command + 普通请求` 的拆分与响应协议。
   - TODO: 定义 command result 与 assistant response 是否共享一次 SSE stream。

---

## 5. 数据模型草案

以下模型是协议草案，不代表当前实现。

### 5.1 BranchIntentKind

当前公共 `IntentType` 是 `RAG / WRITE / CHAT / COMPOSITE / UNKNOWN`，不应为了草案直接改写其值或大小写。分解后的 branch 若需要区分命令与当前不支持的后台工作，应建立新的 branch-local 枚举，并在 envelope 评审时决定是否进入公共契约：

```python
class BranchIntentKind(str, Enum):
    QUERY = "query"                 # 从当前 RAG 投影
    WRITE = "write"
    CHAT = "chat"
    SYSTEM_COMMAND = "system_command"
    FUTURE_JOB = "future_job"
    UNKNOWN = "unknown"
```

说明：

- 当前 `COMPOSITE` 只应作为 parent 分类信号，不应作为分解后的 branch 类型。
- `FUTURE_JOB` 只是现有 route placeholder，不代表已经存在对应队列或版本计划。

### 5.2 SubIntent

```python
class SubIntent(BaseModel):
    intent_type: BranchIntentKind
    content: str
    original_span: str | None = None
    order: int = 0
    depends_on: list[int] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    route_target: RouteTarget | None = None
```

设计决策：

- `content` 是子意图文本，可以做粗粒度指代消解。
- 精细 query rewrite 仍由分支内 `QueryUnderstanding` 完成。
- `depends_on` 使用 `order` 引用，具体调度语义 TODO。
- `route_target` 是否由 decomposer 直接绑定仍是开放问题。

### 5.3 CompositeIntent

```python
class CompositeIntent(BaseModel):
    sub_intents: list[SubIntent]
    decomposition_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    is_fallback: bool = False
    reason: str = ""
```

设计决策：

- 低置信度时应回退到当前单主意图路径。
- fallback 单子意图不等价于成功分解，需要 `is_fallback=True`。

### 5.4 CompositeGatewayDecision

```python
class CompositeGatewayDecision(BaseModel):
    parent_state_id: str
    parent_summary: dict[str, Any]
    branches: list[GatewayBranchDecision]
    merge_policy: MergePolicy
    fallback_policy: FallbackPolicy
```

TODO:

- 定义 `GatewayBranchDecision`。
- 定义是否允许 branch 内持有 `GatewayState` 快照。
- 定义 envelope 如何序列化给下游。

---

## 6. Route Target 草案

```python
class RouteTargetType(str, Enum):
    PATCHOULI_QUERY = "patchouli_query"
    PATCHOULI_WRITE = "patchouli_write"
    ALICE_RUN = "alice_run"
    SYSTEM_COMMAND = "system_command"
    FUTURE_JOB = "future_job"
    CHAT_ONLY = "chat_only"


class RouteTarget(BaseModel):
    type: RouteTargetType
    payload: dict[str, Any] = Field(default_factory=dict)
```

设计决策：

- Gateway 不直接执行下游业务。
- Route target 是消费提示，不是执行结果。
- `FUTURE_JOB` 只作为后续版本预留，不在 Phase C1 的协议评审前执行。

---

## 7. 分支执行模型 TODO

需要决定：

1. 每个 `SubIntent` 是否独立重走话题准备与查询分析，还是复用 parent 的冻结入口上下文。
2. 多个 QUERY 子意图是否分别检索后交给 Alice 合成。
3. 多个 WRITE 子意图是否分别进入 Patchouli finalize，还是合并为一次 memory write signal。
4. 分支是否共享 `CandidateTopics`。
5. 分支是否共享 `RoutedTopicContext`，还是每个分支重新 topic routing。
6. 分支失败是否影响 parent response。
7. 分支结果如何在 SSE 中展示。

当前倾向：

- Phase C1 先定义 envelope 和消费协议，不执行并发分支。
- Phase C2 再实现或扩展 LLM decomposition。
- 真正并发分支执行后置到 Phase C3 或更后版本。

---

## 8. 下游消费边界 TODO

### 8.1 Patchouli

TODO:

- QUERY 分支如何生成 retrieval request。
- WRITE 分支如何影响 memory write signal。
- topic prepare 是 parent 级还是 branch 级。
- 多分支 memory refs 如何合并。

约束：

- Patchouli 不应接收裸 `list[GatewayState]`。
- Patchouli 只接收稳定 envelope 或派生 prepare decision。

### 8.2 Alice

TODO:

- 多分支 QUERY 是否由 Alice 统一回答。
- 需要协作的分支是否转为 Alice execution plan。
- Alice 是否需要知道 branch dependency。

### 8.3 Frontend

TODO:

- 多分支结果如何展示。
- 分支执行中/失败/回退如何展示。
- command result + assistant response 是否同一 stream。

---

## 9. Fallback 策略

本计划必须保留以下 fallback：

| 场景 | 行为 |
| --- | --- |
| decomposer 关闭 | 退回当前单主意图路径 |
| decomposer 解析失败 | 退回当前单主意图路径 |
| 分解置信度低 | 退回当前单主意图路径 |
| 子意图置信度低 | 可丢弃、合并或整体回退，策略 TODO |
| 下游不支持 envelope | 退回当前单主意图路径 |
| 合并失败 | 返回 parent 单主意图 response 或要求用户澄清，策略 TODO |

不允许：

- 抛未捕获异常中断 chat。
- 把未验证的 `list[GatewayState]` 直接交给 Patchouli/Alice。
- 在低置信度下强行多分支执行。

---

## 10. Golden Dataset TODO

Phase C0 应从当前 `COMPOSITE` 分类、私有 `sub_intents`、fallback 与用户后续修正中建立脱敏样本，并在新增公共 envelope 前整理 golden dataset。当前代码没有 `composite_deferred` 字段，不能把不存在的占位状态作为采样前提。

样本类别：

- 看似复合但不应拆。
- 明确并列查询。
- 查询 + 写入。
- 查询 + 系统指令。
- 修改/更新历史上下文。
- 需要规划或长期后台执行的复杂请求。
- 低置信度需要澄清的请求。

评价指标 TODO：

- 过度分解率。
- 漏分解率。
- 子意图边界准确率。
- route target 准确率。
- fallback 触发合理性。

---

## 11. 实现计划

### Phase C0: 基线测量与样本集

目标：先证明现有共享调用在哪些复合输入上真正失败。

实现范围：

1. 为 `COMPOSITE` 分类率、私有 `sub_intents` 非空率、fallback 和字段质量建立摘要指标；
2. 建立脱敏 golden dataset，覆盖不应拆分、并列查询、查询加写入、命令混合和低置信度输入；
3. 决定保留共享 Query Understanding、增加独立 decomposer，还是先移除无消费者字段。

验收：

- 指标不泄露原始消息与上下文；
- 方案选择有可复现样本依据；
- 未通过门禁时不新增公共 composite contract。

### Phase C1: Composite Envelope 与消费协议

目标：先定义输出 envelope 和下游消费协议。

实现范围：

1. 设计 `CompositeGatewayDecision` 或等价 envelope。
2. 明确 `parent_state`、`branches`、`merge_policy`、`fallback_policy`。
3. 定义 Patchouli 与 Alice 的消费边界；`FUTURE_JOB` 保持 unsupported fallback。
4. 定义多分支排序、依赖、失败、用户确认和展示策略。

验收：

- 下游消费方能识别 envelope。
- 低置信度或下游不支持时可回退。
- 不以 `list[GatewayState]` 作为公开协议。

### Phase C2: IntentDecomposer

目标：在 envelope 和 dataset 基础上引入真实分解。

实现范围：

1. 在 Phase C0 证明确有收益后，引入可开关的 `IntentDecomposerStep` 或扩展现有 Resolver 私有实现。
2. 调用 `IntentDecomposerEngine`。
3. 设计窄 prompt：边界、类型、顺序、依赖、置信度。
4. 引入 `confidence_threshold`。
5. 使用 Phase C0 的样本构建并持续维护 golden dataset。

验收：

- 能区分“看似复合但不该拆”和“确实需要拆”。
- 分解失败或关闭时行为等价当前单主意图路径。
- 分解结果不绕过 Phase C1 envelope。

---

## 12. 开放问题

1. `SubIntent` 是否直接绑定 `route_target`，还是只绑定 intent type，由后续 PlannerRouter 决定目标？
2. 多个 QUERY 子意图是分别检索后交给 Alice 合成，还是先在 Gateway 层生成统一 retrieval plan？
3. WRITE 子意图应在 Gateway 阶段生成 memory write request，还是只生成 `memory_write_signal` 供 Patchouli finalize 使用？
4. 复合输入中的系统指令是否允许和普通聊天共享一次响应，还是拆成 command result + chat result 两条事件？
5. 本计划是否需要 lightweight planner，还是只做 decomposition？
6. IntentDecomposer prompt、置信度阈值和 golden dataset 如何设计，才能避免过度分解与漏分解？
7. 多分支并行时，`CandidateTopics` 和 `RoutedTopicContext` 是否克隆，还是保证只读共享？
8. Branch trace 如何合并到 parent trace？
9. 低置信度时是直接单主意图 fallback，还是先向用户澄清？

---

## 13. 当前结论

复合意图分解必须建立在稳定的单主意图 workflow、下游公共契约和可验证样本之上。当前实现只有 `COMPOSITE` 分类与不对外提交的私有 `sub_intents`，没有 `CompositeGatewayDecision`、branch 调度、merge policy 或多分支消费能力。

因此，下一步不是直接让 LLM 输出更多字段，而是先完成 Phase C0 的证据门禁，再冻结 envelope 与消费协议。其余内容都属于计划，不应被当前 Gateway、Patchouli、Alice 或 Frontend 文档写成已经生效的能力。
