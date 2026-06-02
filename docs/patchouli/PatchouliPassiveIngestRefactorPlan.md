# Patchouli Passive Ingest 结构化改造方案

## 1. 文档目标

本文件用于整理 **被动 ingest 模式接入当前 TurnRecord / TurnEvent / AgentAction / TraceItem 结构化主链** 的实施方案。

当前 transcript / `LogicalBlock` 重构已经基本完成主动模式（Kernel / MTP）主链收束，但被动模式仍停留在旧时代的 observer 拼接逻辑：

- `system.ingest()` 只接收 `user / assistant`
- `ObserverSessionBuffer` 只拼接 `assistant_message`
- 感知层仍为被动模式保留 `assistant_message` fallback

因此，本次改造的目标不是仅修补几个字段，而是：

- 让被动模式与主动模式一样，直接产出结构化 `InteractionPayload`
- 让被动模式能够表达 `tool_call / tool_result`
- 将 observer 会话编排从 `TheEye` 中独立出来
- 最终删除感知层中专为被动模式保留的 legacy fallback

---

## 2. 当前实现现状

## 2.1 当前入口

被动模式入口在 [`system.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/patchouli/system.py) 的 `ingest()`：

- `role == "user"`：
  - 调用 `eye.ingest_user_async()`
  - 若上一轮被 flush，则将 payload 提交给 `kernel.submit_interaction()`
  - 再调用 `kernel.handle_hot()` 返回检索结果
- `role == "assistant"`：
  - 调用 `eye.ingest_assistant()`
  - 仅进入缓冲，不立即提交
- 其他角色：
  - 直接返回 `ignored`

当前入口的限制是：

- 不能接收 `tool_call`
- 不能接收 `tool_result`
- 不能接收更通用的事件元数据（如 `action_id / tool_name / tool_args / status`）

## 2.2 当前被动缓冲链路

被动会话聚合发生在 [`observer_buffer.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/gateway/observer_buffer.py)：

- `accept_user(content, gaze_result)`
- `accept_assistant(content)`
- `flush()`
- `_build_payload()`

当前内部状态只有：

- `_user_content`
- `_assistant_parts`
- `_gaze_result`

当前输出 payload 只有：

- `user_message`
- `assistant_message`
- `identity`
- `rewritten_query`
- `worth_saving`
- `mtp_traces=[]`
- `write_focus=None`
- `update_focus=None`

也就是说，被动模式当前并不会构建：

- `assistant_final_text`
- `turn_events`
- `actions`
- `semantic_traces`

## 2.3 当前感知层 fallback

在 [`semantic_flow_perception_layer.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/perception/semantic_flow_perception_layer.py) 中，当前仍保留双路径：

- 结构化路径：
  - 使用 `payload.turn_events`
  - `ActionReducer.reduce()`
  - `TraceReducer.reduce()`
- fallback 路径：
  - 若 `turn_events` 为空，则使用 `payload.assistant_message`

这意味着被动模式还没有真正接入当前主链，而只是被兼容层托底。

## 2.4 当前职责混杂

`TheEye` 当前除了负责 `gaze()` 之外，还承担：

- observer buffer 池管理
- session flush
- idle timeout flush
- observer idle scheduler

这会导致：

- `TheEye` 职责继续膨胀
- 被动 ingest 状态机无法独立演进
- `gaze` 分析逻辑和 session 编排逻辑耦合

---

## 3. 当前问题判断

本阶段需要解决的问题可归纳为 5 类：

### 3.1 被动模式没有结构化事件流

当前被动模式只保留：

- 一条 `user_message`
- 一条拼接后的 `assistant_message`

这会导致：

- 无法描述 `tool_call`
- 无法描述 `tool_result`
- 无法形成 `TurnEvent -> AgentAction -> TraceItem`
- 无法让 generation / history 真正共享同一套结构化基础

### 3.2 被动模式无法表达 function call agent

本轮重构的核心判断之一是：

- MTP 与普通 json function call 本质上都是工具调用
- 主动/被动两种模式应该共享同一套视图逻辑

但当前 `system.ingest()` 对 `tool` 事件直接无能力承接，这使得被动模式天然落后于主动模式。

### 3.3 Observer 编排与 TheEye 耦合过深

`TheEye` 应该专注于：

- query 理解
- intent 判断
- 路由目标话题

而 observer 编排更像：

- session turn 管理
- idle timeout 策略
- flush 触发与 payload 构建

这两类职责继续耦合，会让后续改造越来越困难。

### 3.4 下一轮 user 到来时的路由归属存在语义风险

当前 `user2` 到来时：

- `eye.ingest_user_async()` 会先 flush 出 `turn1`
- 但 `system.ingest()` 提交 `turn1` 时使用的是 `user2` 的 `gaze_result.target_topic`

这意味着上一轮 payload 的 topic 归属可能被下一轮路由结果污染。

### 3.5 测试仍围绕旧 payload 语义

现有被动模式测试主要验证：

- `assistant_message` 拼接
- `user -> assistant -> user` 时自动 flush
- idle timeout

但尚未验证：

- `turn_events`
- `assistant_final_text`
- `tool_call / tool_result`
- 上一轮独立 `target_topic`

---

## 4. 目标形态

本次改造完成后，被动模式应达到以下状态：

### 4.1 与主动模式共享同一份 payload 结构

被动模式应直接构造：

- `user_message`
- `assistant_final_text`
- `turn_events`
- `identity`
- `rewritten_query`
- `worth_saving`
- `mtp_traces=[]`
- `write_focus=None`
- `update_focus=None`

其中：

- `assistant_message` 不再作为主字段
- `turn_events` 成为唯一结构化真相源

### 4.2 与主动模式共享同一条 perception 主链

目标态下：

- `SemanticFlowPerceptionLayer.ingest_payload()` 不再区分“主动结构化路径”和“被动 fallback 路径”
- 主动/被动都统一走：
  - `turn_events`
  - `ActionReducer`
  - `TraceReducer`
  - `TurnRecord`
  - `LogicalBlock`

### 4.3 被动模式可表达工具调用

被动模式至少要能表达以下原子事件：

- `user_message`
- `assistant_message`
- `tool_call`
- `tool_result`

从而支持：

- 普通 function call agent
- 外部工具返回
- 多段 assistant + tool 混合消息流

### 4.4 TheEye 只保留 gaze 职责

目标态下：

- `TheEye`：只做 `gaze()`
- 被动会话状态机与 idle 监控：迁到独立组件

---

## 5. 推荐设计

## 5.1 组件拆分

建议新增一个独立组件，例如：

- `PassiveObserverIngressor`

建议放置位置：

- [`src/hivememory/patchouli/passive_ingest/`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/patchouli/)
  - `ingressor.py`
  - `observer_turn_buffer.py`
  - `models.py`（如需要）

### 5.1.1 `TheEye`

保留职责：

- `gaze()`
- `gaze fallback`
- 预检索前的 query 理解

移除职责：

- observer buffer 池
- ingest_user / ingest_assistant
- flush_session / flush_idle_sessions
- idle monitor scheduler

### 5.1.2 `PassiveObserverIngressor`

负责：

- 接收外部离散事件
- 调用 `TheEye.gaze()` 为新一轮 user 建立 route 元数据
- 管理多个 session 的 `MessageTurnBuffer`
- 执行：
  - Next User Turn flush
  - Idle Timeout flush
  - Explicit EOF flush
- 产出结构化 `(payload, target_topic)` 对

### 5.1.3 `MessageTurnBuffer`

负责：

- 单 session / 单轮的事件聚合
- 缓存 user / assistant / tool 事件
- 持有该轮的 `gaze_result`
- 持有该轮的 `target_topic`
- flush 时构建结构化 `InteractionPayload`

## 5.2 数据模型建议

### 5.2.1 `MessageTurnBuffer` 内部状态

建议内部至少缓存：

```python
class MessageTurnBuffer:
    _identity: Identity
    _state: MessageBufferState

    _user_content: str | None
    _assistant_parts: list[str]
    _turn_events: list[TurnEvent]

    _gaze_result: EyeGazeResult | None
    _target_topic: str | None
    _last_activity: float
```

说明：

- `_assistant_parts` 仍可保留，用于快速构建 `assistant_final_text`
- `_turn_events` 作为最终真相源
- `_target_topic` 必须和本轮绑定，不能在 flush 时从“当前下一轮”推断

### 5.2.2 事件接收接口

建议接口如下：

```python
accept_user(content, gaze_result)
accept_assistant(content)
accept_tool_call(content, *, action_id=None, tool_name=None, tool_kind=None, tool_args=None, target=None)
accept_tool_result(content, *, action_id=None, status=None, render_as="plain")
flush() -> tuple[InteractionPayload, str] | None
```

### 5.2.3 `TurnEvent` 映射规则

被动模式事件映射建议如下：

- 外部 user 消息
  - `TurnEvent(kind="user_message", role="user")`
- 外部 assistant 文本
  - `TurnEvent(kind="assistant_message", role="assistant")`
- 外部 function call / tool invocation
  - `TurnEvent(kind="tool_call", role="assistant")`
- 外部 function return / tool result
  - `TurnEvent(kind="tool_result", role="system" 或 "user")`

其中：

- `action_id` 用于把 `tool_call` 和对应 `tool_result` 绑定起来
- `tool_kind / tool_name / tool_args / status / target / render_as` 直接落在 `TurnEvent` 上

## 5.3 `assistant_final_text` 的构建规则

被动模式下：

- `assistant_final_text` 应由本轮所有 `assistant_message` 事件顺序拼接得到
- `tool_call / tool_result` 不进入 `assistant_final_text`

这与主动模式的语义是一致的：

- `assistant_final_text` 表示最终自然语言回复
- 工具调用过程由 `turn_events` 表达

## 5.4 `target_topic` 的归属规则

必须明确：

- `target_topic` 属于“本轮 user 发起时的 route 决策”
- 它不是 flush 时动态计算的值

因此，在 `accept_user(content, gaze_result)` 时就要把：

- `gaze_result.target_topic`

缓存到该轮 buffer 中。

当下一轮 user 到来时：

- 先 flush 上一轮，返回 `(previous_payload, previous_target_topic)`
- 再初始化新一轮 `(new_user, new_gaze_result, new_target_topic)`

---

## 6. `system.ingest` 接口改造建议

当前签名：

```python
async def ingest(
    role: str,
    content: str,
    user_id: str,
    agent_id: str = "omni_doll",
    session_id: Optional[str] = None,
    context: Optional[List[StreamMessage]] = None,
) -> Dict[str, Any]:
```

这套签名对于 tool 事件已经不够用了。

建议两种方案：

### 6.1 方案 A：扩展 `ingest()` 参数

扩展为：

```python
async def ingest(
    role: str,
    content: str,
    user_id: str,
    agent_id: str = "omni_doll",
    session_id: Optional[str] = None,
    context: Optional[List[StreamMessage]] = None,
    action_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    tool_kind: Optional[str] = None,
    tool_args: Optional[dict[str, Any]] = None,
    target: Optional[str] = None,
    status: Optional[str] = None,
    render_as: str = "plain",
) -> Dict[str, Any]:
```

并支持：

- `role in {"user", "assistant", "tool_call", "tool_result"}`

优点：

- 改动局部
- 与当前调用方式最接近

缺点：

- 参数会继续膨胀
- 后续扩展协议会比较难看

### 6.2 方案 B：新增统一事件输入模型

建议新增：

```python
class PassiveIngressEvent(BaseModel):
    role: Literal["user", "assistant", "tool_call", "tool_result"]
    content: str
    action_id: str | None = None
    tool_name: str | None = None
    tool_kind: str | None = None
    tool_args: dict[str, Any] | None = None
    target: str | None = None
    status: str | None = None
    render_as: str = "plain"
```

由系统提供：

```python
async def ingest_event(
    event: PassiveIngressEvent,
    user_id: str,
    agent_id: str = "omni_doll",
    session_id: Optional[str] = None,
    context: Optional[List[StreamMessage]] = None,
) -> Dict[str, Any]:
```

推荐判断：

- 若项目准备长期承接外部 agent 接入，推荐方案 B
- 若希望先最小改动落地，可先做方案 A，再在下一阶段收敛到事件模型

本轮更推荐方案 B。

---

## 7. 具体实施步骤

## 7.1 Phase P1：拆出被动 ingest orchestration

目标：

- 从 `TheEye` 中拆出 observer session 管理职责

要处理的文件：

- [`patchouli/eye.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/patchouli/eye.py)
- 新增：
  - `patchouli/passive_ingest/ingressor.py`
  - `patchouli/passive_ingest/observer_turn_buffer.py`

具体工作：

- 新增 `PassiveObserverIngressor`
- 将下列方法迁出 `TheEye`：
  - `ingest_user(_async)`
  - `ingest_assistant()`
  - `flush_session()`
  - `flush_idle_sessions()`
  - `flush_all_pending_sessions()`
  - idle monitor 相关方法
- `TheEye` 仅保留 `gaze()`

完成标志：

- `PatchouliSystem.ingest()` 不再直接调用 `TheEye.ingest_user_async()`
- `TheEye` 不再持有 `MessageTurnBufferManager`

## 7.2 Phase P2：升级 MessageTurnBuffer 为结构化事件缓冲器

目标：

- 被动模式开始构建 `turn_events`

要处理的文件：

- `message_turn_buffer.py`
- [`core/models/interaction.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/core/models/interaction.py)（如需补帮助方法）

具体工作：

- 新增 `_turn_events`
- 新增：
  - `accept_tool_call()`
  - `accept_tool_result()`
- `accept_user()` 写入 `TurnEvent(kind="user_message")`
- `accept_assistant()` 写入 `TurnEvent(kind="assistant_message")`
- `accept_tool_call()` 写入 `TurnEvent(kind="tool_call")`
- `accept_tool_result()` 写入 `TurnEvent(kind="tool_result")`
- flush 时计算：
  - `assistant_final_text`
  - `turn_events`
  - `target_topic`

完成标志：

- 被动模式 flush 后 payload 已包含 `turn_events`

## 7.3 Phase P3：扩展 `system.ingest` 承接 tool 事件

目标：

- `system.ingest` 能表达 function call agent 的完整输入

要处理的文件：

- [`patchouli/system.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/patchouli/system.py)
- 新增 `PassiveIngressEvent`（若采用方案 B）

具体工作：

- 支持 `tool_call`
- 支持 `tool_result`
- user/assistant/tool 全都进入 `PassiveObserverIngressor`
- 明确返回值策略：
  - `user` 仍可返回检索结果
  - `assistant/tool_*` 返回 `buffered`

完成标志：

- `PatchouliSystem.ingest()` 不再把非 `user/assistant` 一律忽略

## 7.4 Phase P4：切 perception 单路径

目标：

- 删除专为被动模式保留的 fallback

要处理的文件：

- [`semantic_flow_perception_layer.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/perception/semantic_flow_perception_layer.py)
- [`models.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/perception/models.py)

具体工作：

- 移除 `assistant_message` 作为 ingest 主来源
- 要求 `turn_events` 始终存在
- 被动模式也通过：
  - `ActionReducer.reduce(payload.turn_events)`
  - `TraceReducer.reduce(actions)`

完成标志：

- perception 只保留单一路径
- `assistant_message` 彻底降级或删除

## 7.5 Phase P5：补齐被动模式结构化测试

目标：

- 用结构化测试替代旧的 `assistant_message` 拼接测试心智

建议新增测试类型：

### P5-A `MessageTurnBuffer` 单测

- `user -> assistant -> flush` 产出 `turn_events`
- `user -> tool_call -> tool_result -> assistant -> flush`
- 多段 assistant 拼接为 `assistant_final_text`
- `action_id` 能正确传递到 `TurnEvent`

### P5-B 路由正确性测试

- `user1 -> assistant1 -> user2`
- flush 出来的 `turn1` 必须使用 `user1` 的 `target_topic`
- 而不是 `user2` 的

### P5-C `system.ingest` 集成测试

- `user / assistant / tool_call / tool_result` 混合输入
- 最终提交给 `kernel.submit_interaction()` 的 payload 含：
  - `turn_events`
  - `assistant_final_text`
  - 正确的 `target_topic`

### P5-D perception 主链测试

- 被动 payload 不再走 `assistant_message` fallback
- 被动模式也能产出：
  - `actions`
  - `semantic_traces`
  - `TurnRecord`

---

## 8. 风险与注意事项

## 8.1 不要在这一轮同时重构外部协议适配层

本阶段目标是统一内部数据流，不是一次性支持所有上游框架。

建议第一版只覆盖：

- `user`
- `assistant`
- `tool_call`
- `tool_result`

不要在同一阶段引入更多稀有角色。

## 8.2 不要让 `TraceItem` 回到被动入口构建

被动模式只应负责构建：

- `assistant_final_text`
- `turn_events`

`actions` 和 `semantic_traces` 仍应在 perception 层统一派生。

否则会重新出现主动/被动双套 reducer。

## 8.3 不要保留“被动专用 turn_events 语义”

被动模式的 `TurnEvent.kind` 必须与主动模式完全一致：

- `user_message`
- `assistant_message`
- `tool_call`
- `tool_result`

不要再引入：

- `legacy_tool`
- `observer_result`
- `json_function_call`

这类被动专用命名，否则统一设计会再次失败。

## 8.4 `target_topic` 必须与轮次绑定

这是本次改造最重要的行为修复点之一。

若继续在 flush 时使用“当前新 user 的 gaze 结果”，会导致：

- 上一轮被提交到错误话题
- topic history 与 generation 上下文错位

---

## 9. 推荐落地顺序

为了控制风险，建议按以下顺序推进：

1. 拆出 `PassiveObserverIngressor`
2. 升级 `MessageTurnBuffer` 支持 `turn_events`
3. 让 `system.ingest` 支持 `tool_call/tool_result`
4. 让被动模式 payload 改填 `assistant_final_text + turn_events`
5. 删除 perception fallback
6. 清理 `assistant_message` 残留与旧测试

这样做的原因：

- 先拆职责，再换数据模型，改动面更可控
- 先让被动模式具备结构化能力，再删 fallback，风险最低

---

## 10. 一句话结论

这次被动 ingest 改造的本质，不是“给 observer buffer 多补几个字段”，而是：

- 把被动模式从“字符串拼接缓冲器”
- 提升为“结构化 turn 构建器”

只有这样，主动模式与被动模式才能真正共享：

- `TurnEvent`
- `AgentAction`
- `TraceItem`
- `TurnRecord`
- `LogicalBlock`

并最终删除当前所有专为被动模式保留的额外兼容代码。
