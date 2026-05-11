# Patchouli Transcript 双视图重构设计草案

**版本**: Draft 1\
**状态**: 设计中\
**适用范围**: `patchouli` 对话主链路、`perception` 感知层、`generation` 记忆生成链路

***

## 1. 背景与问题

在当前实现中，`PatchouliSystem` 会在一轮递归生成结束后，通过 `_reconstruct_raw_assistant_text()` 重建 `assistant_message`，再将其作为 `InteractionPayload` 的一部分提交到感知层。

这套设计最初建立在一个旧前提上：

- MTP 指令由 assistant 发出
- MTP 执行结果也被伪造为 assistant 文本的一部分
- 因此只要“拼接 assistant 文本”，就能得到一份完整的原始回复

但当前链路已经发生变化：

- MTP 指令仍由 assistant 发出
- MTP 执行结果已经改为通过 `role=user` 的 `[System MTP Execution Result]` 消息回填
- assistant 侧天然更接近“clean response”
- 历史消息视图和记忆生成视图，已经不再适合继续共用一个 `raw_response` 字符串

这导致以下几个问题：

- `_reconstruct_raw_assistant_text()` 的语义已经过时
- `raw_response` 不再能真实代表“完整的一轮执行过程”
- `MTPLogParser` 仍被主路径依赖，用字符串清洗承担结构化职责
- `LogicalBlock` 中虽然已经保留了 `semantic_traces`、`raw_response`、`clean_response` 等字段，但后续历史回放和记忆生成链路主要只消费 `clean_response`
- `state_summary`、`semantic_traces` 在进入 Generation 视图时被裁掉

归根结底，当前链路的问题不是单个函数实现错误，而是：

- 一个字段同时被期待服务于多种完全不同的语义目标
- 但这些目标本质上应该拆分为两套视图

***

## 2. 设计目标

本次重构的目标如下：

1. 明确区分两套 transcript 视图
2. 以结构化事件流替代“拼接字符串”作为单一真相源
3. 将 `MTPLogParser` 从主路径解析器降级为兼容 fallback
4. 让 `state_summary` 和 `semantic_traces` 正式进入记忆生成视图
5. 保持迁移过程可渐进，不一次性破坏现有接口

具体来说，需要满足两种不同的消费方：

### 2.1 历史消息视图

用于下一轮 Agent 对话上下文构建。

其要求是：

- 尽量还原模型当时“看到了什么”
- 保留 assistant 发出的 MTP 指令
- 保留系统回填的 MTP 执行结果
- 保持消息顺序与角色一致

### 2.2 记忆生成视图

用于 Generation Engine 构建记忆提取上下文。

其要求是：

- 不保留完整工具返回正文
- 保留动作摘要，以反映 Agent 做了什么
- 保留最终自然语言回复
- 保留 `state_summary`
- 保留 `write_focus` / `update_focus`

***

## 3. 当前链路梳理

当前主链路如下：

```text
PatchouliSystem.chat/chat_stream
    -> LoopExecutor 递归生成
    -> _reconstruct_raw_assistant_text(messages)
    -> InteractionPayload.assistant_message
    -> SemanticFlowPerceptionLayer.ingest_payload()
    -> MTPLogParser.parse(assistant_message)
    -> LogicalBlock(raw_response, clean_response, semantic_traces)
    -> PerceptionContextConverter / LogicalBlock.to_stream_messages()
    -> GenerationRequest(context_messages)
```

### 3.1 当前运行时事实

在 `LoopExecutor` 中：

- assistant 消息中保存的是模型生成内容，可能包含 MTP 指令 `⟪ ... ⟫`
- MTP 执行结果通过 `role=user` 的系统消息回填

因此，当前 assistant 侧天然并不包含完整执行结果。

### 3.2 当前感知层事实

在 `SemanticFlowPerceptionLayer.ingest_payload()` 中：

- `assistant_message` 被送入 `MTPLogParser.parse()`
- parser 输出 `clean_text + fallback_traces`
- 最终构建 `LogicalBlock`

因此主路径仍在依赖“从字符串里反推结构”。

### 3.3 当前消费链路事实

在两条主要消费链路中：

- 历史消息构建主要消费 `clean_response`
- Generation 上下文构建也主要消费 `clean_response`

这意味着：

- `raw_response` 存在，但不是真正的主消费字段
- `semantic_traces` 存在，但没有真正进入 Generation transcript
- `state_summary` 没有进入 Generation 的主输入

***

## 4. 核心判断：需要两个视图，但可以共享一份底层数据

本次重构的核心判断如下：

- 历史消息视图和记忆生成视图必须分离
- 但两者不需要维护两套不同来源的数据
- 它们应该从同一份结构化的“回合执行记录”中派生

换句话说：

- 不再以 `raw_response` 字符串作为真相源
- 改为以“本轮结构化事件流”作为真相源

在此模型下：

- 历史消息视图从事件流重放
- 记忆生成视图从事件流降维

***

## 5. 新的数据模型草案

### 5.1 TurnEvent：单一事件模型

建议新增 `TurnEvent`，表示一轮交互中可重放的最小事件。

```python
class TurnEvent(BaseModel):
    kind: Literal[
        "assistant_text",
        "mtp_command",
        "mtp_result",
    ]
    sequence: int
    role: Literal["assistant", "user"]
    content: str

    verb: Optional[str] = None
    target: Optional[str] = None
    args: Optional[dict[str, Any]] = None
    status: Optional[str] = None

    include_in_history: bool = True
    include_in_generation: bool = False
```

字段说明：

- `kind`: 事件类型
- `sequence`: 在本轮中的顺序，用于稳定重放
- `role`: 重放时的消息角色
- `content`: 原始文本内容
- `verb/target/args/status`: 结构化 MTP 元信息
- `include_in_history`: 是否进入历史消息视图
- `include_in_generation`: 是否直接进入记忆生成视图，默认不进入

### 5.2 TurnRecord：运行时的一轮交互记录

`TurnRecord` 在本草案中**不默认表示一个长期独立存在的领域对象**。  
它首先被定义为一个**运行时 DTO（Data Transfer Object）**，用于承载：

- `LoopExecutor` 在递归生成期间采集到的本轮结构化事件
- `PatchouliSystem` 后处理阶段提交给感知层的结构化内容
- 从“字符串重建”迁移到“事件流直传”过程中的桥接职责

也就是说，`TurnRecord` 的主要语义是：

- 运行时记录
- 传输对象
- 事件真相源容器

建议将当前“单一 assistant\_message”改造为更明确的一轮记录：

```python
class TurnRecord(BaseModel):
    user_query: str
    rewritten_query: Optional[str] = None
    identity: Identity

    assistant_final_text: str = ""
    events: list[TurnEvent] = Field(default_factory=list)

    semantic_traces: list[TraceItem] = Field(default_factory=list)

    worth_saving: Optional[bool] = None
    write_focus: Optional[Any] = None
    update_focus: Optional[Any] = None
```

字段职责：

- `assistant_final_text`: 最终自然语言回复，等价于当前 `loop_result.final_text`
- `events`: 本轮完整结构化事件流，是真相源
- `semantic_traces`: 从 `events` 降维得到的摘要动作
- `write_focus` / `update_focus`: 保持现有控制信号设计不变

### 5.3 TurnRecord 与 LogicalBlock 的关系

`TurnRecord` 和 `LogicalBlock` 的字段会有较高重叠，这是**设计上有意暴露出来的问题域**，其目的是提醒后续实现时必须明确两者的层级关系。

两者应理解为：

- `TurnRecord`: 运行时/协议层的一轮结构化原始记录
- `LogicalBlock`: 感知层/Buffer 层的可归档、可调度、可管理语义块

从工程视角看，可将两者关系概括为：

```text
LogicalBlock = TurnRecord + Perception Metadata
```

其中 `Perception Metadata` 主要指：

- `worth_saving`
- `priority`
- `write_focus`
- `update_focus`
- `total_tokens`
- 未来可能追加的话题管理、flush、folding 相关元信息

为避免“两份模型、两份字段、两套真相源”的问题，本草案明确提出两种可选方案。

### 5.4 方案一：TurnRecord 作为运行时 DTO，LogicalBlock 作为感知层领域对象

这是**短期推荐方案**，适合分阶段平滑落地。

#### 设计原则

- `TurnRecord` 仅存在于运行时主链路：
  - `LoopExecutor`
  - `PatchouliSystem._chat_post_process`
  - `InteractionPayload`
- 到达感知层后，立刻转换为 `LogicalBlock`
- `TurnRecord` 不进入 Buffer
- `TurnRecord` 不作为长期存储模型

#### 优点

- 改动风险较低
- 可以先完成“结构化事件流成为真相源”这一步
- 不需要立刻大改 `LogicalBlock` 的访问方式

#### 缺点

- 仍然存在一次字段搬运
- 运行时模型与感知层模型会有一段时间重叠

#### 方案一中的 LogicalBlock 目标形态

在该方案下，`LogicalBlock` 仍保持扁平字段结构，但优先消费来自 `TurnRecord` 的内容：

```python
class LogicalBlock(BaseModel):
    user_query: str
    rewritten_query: Optional[str] = None
    identity: Identity

    assistant_final_text: str = ""
    turn_events: list[TurnEvent] = Field(default_factory=list)
    semantic_traces: list[TraceItem] = Field(default_factory=list)

    worth_saving: Optional[bool] = None
    write_focus: Optional[Any] = None
    update_focus: Optional[Any] = None

    # 兼容字段
    raw_response: str = ""
    clean_response: str = ""
```

兼容字段说明：

- `raw_response`: 兼容旧数据或调试用途，不再作为真相源
- `clean_response`: 短期内保留，语义上等价于 `assistant_final_text`

### 5.5 方案二：将 TurnRecord 内嵌进 LogicalBlock

这是**长期推荐方案**，适合在主链路稳定后进一步收敛模型职责。

#### 设计原则

- `TurnRecord` 作为内容真相源被保留
- `LogicalBlock` 只承担感知层附加元信息
- 通过 `LogicalBlock.turn` 组合而非字段复制

目标结构示例：

```python
class LogicalBlock(BaseModel):
    turn: TurnRecord

    worth_saving: Optional[bool] = None
    priority: str = "NORMAL"
    write_focus: Optional[Any] = None
    update_focus: Optional[Any] = None
    total_tokens: int = 0

    # 兼容字段，可逐步移除
    raw_response: str = ""
    clean_response: str = ""
```

#### 优点

- 内容字段只有一份，避免复制
- 分层最清晰
- 更符合“TurnRecord 是内容真相，LogicalBlock 是感知层容器”的语义

#### 缺点

- 改动面较大
- 当前大量 `block.user_query` 风格访问需要改成 `block.turn.user_query`
- 需要系统性调整上下游 builder 与测试

### 5.6 推荐路径

本草案建议采用以下收敛路径：

#### 短期

先落地**方案一**：

- 把 `TurnRecord` 作为运行时 DTO 引入
- 让结构化事件流先跑通
- 不急于大改 `LogicalBlock`

#### 长期

待双视图链路稳定后，再评估是否演进到**方案二**：

- 将 `TurnRecord` 内嵌进 `LogicalBlock`
- 收敛重复字段
- 进一步清理兼容字段

换句话说：

- 方案一解决“先把真相源统一”
- 方案二解决“模型结构最终如何收敛”

***

## 6. 两个视图的职责划分

### 6.1 历史消息视图

用途：

- 用于 `PerceptionContextConverter`
- 用于下一轮 prompt 组装

输入：

- `LogicalBlock.user_query`
- `LogicalBlock.turn_events`
- 多智能体 `identity`

输出：

- OpenAI 风格 messages

保留内容：

- assistant 自然语言片段
- MTP 指令
- MTP 返回结果

不直接依赖：

- `raw_response`

### 6.2 记忆生成视图

用途：

- 用于 `LibrarianCore -> GenerationRequest -> GenerationEngine`

输入：

- `state_summary`
- `LogicalBlock.user_query`
- `LogicalBlock.semantic_traces`
- `LogicalBlock.assistant_final_text`
- `write_focus` / `update_focus`

输出：

- 适合记忆提取的 transcript 或结构化 `GenerationContext`

保留内容：

- 动作摘要
- 最终自然语言回复
- 话题状态摘要

丢弃内容：

- `READ` 返回正文
- `SEARCH` 菜单结果
- XML / `<mtp_response>` 全量文本

***

## 7. 新增两个 Builder

### 7.1 HistoryTranscriptBuilder

职责：

- 从 `LogicalBlock.turn_events` 渲染可重放的历史消息
- 替代当前 `PerceptionContextConverter.blocks_to_messages()` 中基于 `clean_response` 的简单拼接逻辑

目标接口：

```python
class HistoryTranscriptBuilder:
    def build_messages(
        self,
        blocks: list[LogicalBlock],
        current_agent_id: str,
    ) -> list[dict[str, str]]:
        ...
```

渲染规则：

1. 先输出 `user_query`
2. 再按 `sequence` 依次输出 `turn_events`
3. 对 assistant 事件应用多智能体身份前缀
4. `mtp_result` 保持 `role=user`
5. 老数据回退到 `clean_response`

推荐输出示例：

```text
User: 帮我查一下上次写的认证逻辑
Assistant: 我先查找相关实现。⟪ SEARCH | * | query="auth logic" ⟫
User: [System MTP Execution Result]
<mtp_response ...>
Assistant: 我找到了两个相关入口...
```

### 7.2 GenerationTranscriptBuilder

职责：

- 为 Generation Engine 生成去噪但保留动作语义的 transcript
- 替代当前 `LogicalBlock.to_stream_messages()` 在 Kernel 模式下的二元压缩

目标接口：

```python
class GenerationTranscriptBuilder:
    def build_context(
        self,
        blocks: list[LogicalBlock],
        state_summary: str,
    ) -> GenerationContext:
        ...
```

推荐的 `GenerationContext`：

```python
class GenerationTurn(BaseModel):
    user_query: str
    trace_summaries: list[str] = Field(default_factory=list)
    assistant_final_text: str = ""
    identity: Identity

class GenerationContext(BaseModel):
    state_summary: str = ""
    turns: list[GenerationTurn] = Field(default_factory=list)
```

推荐 transcript 示例：

```text
[Topic State]
用户正在重构 system 模块，已抽离 MessageAssembler，当前聚焦 transcript 双视图设计。

[Turn 1]
User: 需要把历史消息和生成视图分开
Actions:
- SEARCH "MTP log parser"
- READ semantic_flow_perception_layer
Assistant: 我已经梳理出当前链路中的两个主要断点...
```

***

## 8. Trace 的单一真相源策略

当前 `semantic_traces` 有两种来源：

- `KoakumaRuntime` 直接记录的 `TraceItem`
- `MTPLogParser` 从 assistant 字符串中回退解析出的 `fallback_traces`

这会形成“双来源”问题。

建议的新原则：

- `turn_events` 是唯一主数据源
- `semantic_traces` 是从 `turn_events` 派生出的摘要视图
- `MTPLogParser` 仅作为老数据或缺失事件时的 fallback

建议新增一个 reducer：

```python
class MTPTraceReducer:
    @staticmethod
    def from_events(events: list[TurnEvent]) -> list[TraceItem]:
        ...
```

映射规则：

- `READ` -> 记录目标别名
- `SEARCH` -> 记录 query
- `RUN` -> 记录 tool + status
- `WRITE/UPDATE` -> 不进入 trace，继续作为 focus

***

## 9. 对现有模块的重构建议

### 9.1 LoopExecutor

职责调整：

- 不再只负责写回 `messages`
- 同时直接采集本轮 `turn_events`

建议采集点：

- 每次追加 assistant 侧带 MTP 的文本时，生成 `mtp_command` 事件
- 每次追加 `[System MTP Execution Result]` 时，生成 `mtp_result` 事件
- 递归自然结束后，将最终自然语言结果写入 `assistant_final_text`

目标：

- 不再从 `messages` 反推“本轮发生了什么”
- 而是在运行时直接记录结构化过程

### 9.2 PatchouliSystem

职责调整：

- `_chat_post_process()` 不再依赖 `_reconstruct_raw_assistant_text()` 作为主输入
- 改为提交：
  - `assistant_final_text`
  - `turn_events`
  - `semantic_traces`

建议：

- `_reconstruct_raw_assistant_text()` 降级为兼容函数
- 或在结构化事件流稳定后彻底移除

### 9.3 SemanticFlowPerceptionLayer

当前逻辑：

```python
clean_text, fallback_traces = MTPLogParser.parse(payload.assistant_message)
```

建议改为双路径：

```python
if payload.turn_events:
    assistant_final_text = payload.assistant_final_text
    semantic_traces = (
        payload.semantic_traces
        or MTPTraceReducer.from_events(payload.turn_events)
    )
else:
    assistant_final_text, fallback_traces = MTPLogParser.parse(payload.assistant_message)
    semantic_traces = payload.mtp_traces or fallback_traces
```

并构建：

- `assistant_final_text`
- `turn_events`
- `semantic_traces`

兼容字段：

- `raw_response`
- `clean_response`

### 9.4 PerceptionContextConverter

建议：

- 不再直接从 `clean_response` 生成历史消息
- 改为委托 `HistoryTranscriptBuilder`

### 9.5 LibrarianCore

建议：

- 不再通过 `_blocks_to_messages()` 仅生成 `StreamMessage[]`
- 改为引入 `GenerationTranscriptBuilder`

这样可以显式把：

- `state_summary`
- `semantic_traces`
- `assistant_final_text`

输入到 Generation 视图中

### 9.6 GenerationRequest / GenerationEngine

当前：

- 只接收 `context_messages`

建议演化：

```python
class GenerationRequest(BaseModel):
    context: GenerationContext
    write_focus: Optional[WriteFocus] = None
    update_focus: Optional[UpdateFocus] = None
```

然后在 `GenerationEngine` 中将 `GenerationContext` 渲染为 transcript。

***

## 10. 兼容策略

为降低一次性重构风险，建议采用兼容字段过渡：

### 10.1 保留但降级的字段

- `InteractionPayload.assistant_message`
- `InteractionPayload.mtp_traces`
- `LogicalBlock.raw_response`
- `LogicalBlock.clean_response`

这些字段短期内仍保留，但优先级下降。

### 10.2 新字段优先级

新增字段后，读取优先顺序建议如下：

1. `turn_events`
2. `assistant_final_text`
3. `semantic_traces`
4. 旧字段 fallback

### 10.3 parser 的新定位

`MTPLogParser` 只用于：

- 老数据兼容
- turn events 缺失时的回退
- 数据修复场景

它不应继续承担主路径结构化解析职责。

***

## 11. 分阶段实施方案

### Phase 1：统一真相源

目标：

- 建立 `TurnEvent`
- 让运行时直接产出结构化事件流

工作项：

- 新增 `TurnEvent`
- 扩展 `InteractionPayload`
- 扩展 `LogicalBlock`
- `LoopExecutor` 直接采集 `turn_events`
- `PatchouliSystem._chat_post_process()` 提交结构化字段

收益：

- `_reconstruct_raw_assistant_text()` 脱离主路径
- 统一事件真相源

### Phase 2：重构历史视图

目标：

- 建立 `HistoryTranscriptBuilder`
- 用真实事件流替代 `clean_response` 压平回放

工作项：

- 新增 `HistoryTranscriptBuilder`
- `PerceptionContextConverter.blocks_to_messages()` 改为委托 builder
- 多智能体身份前缀逻辑迁入 builder

收益：

- 历史消息真正保留 MTP 指令和返回结果

### Phase 3：重构记忆生成视图

目标：

- 将 `state_summary` 和 `semantic_traces` 纳入 Generation 主输入

工作项：

- 新增 `GenerationTranscriptBuilder`
- 新增 `GenerationContext`
- 改造 `LibrarianCore`
- 改造 `GenerationRequest`
- 改造 `GenerationEngine` transcript 渲染逻辑

收益：

- 记忆生成看到完整语义上下文，而不是只有 user/assistant 简化对话

### Phase 4：清理旧字段

目标：

- 清理兼容层，收敛模型职责

工作项：

- 将 `raw_response` 改为调试字段或移除
- 将 `clean_response` 收敛为 `assistant_final_text`
- 将 `assistant_message` 彻底退为兼容入口
- 将 `MTPLogParser` 收敛为 fallback 工具

***

## 12. 最小可实施版本建议

为了降低风险，第一阶段建议只做以下最小切片：

1. 定义 `TurnEvent`
2. 在 `LoopExecutor` 运行时直接记录 `turn_events`
3. 在 `InteractionPayload` 中增加：
   - `assistant_final_text`
   - `turn_events`
4. 在感知层优先消费结构化字段
5. 保留 `MTPLogParser` 作为 fallback

暂时不做：

- `GenerationEngine` 大范围签名修改
- `LogicalBlock` 旧字段立即移除
- 一次性废弃 `clean_response`

这样可以先完成“统一真相源”的关键一步，再分阶段重构两个 builder。

***

## 13. 仍待确认的问题

在正式编码前，仍需确认以下几个设计点：

### 13.1 assistant\_final\_text 的边界

是否严格等于 `loop_result.final_text`？

建议：

- 是
- 它只表示最终自然语言回复
- 不再包含任何系统回填内容

### 13.2 mtp\_result 是否需要进入 history 视图

建议：

- 是
- 否则下轮模型看到的上下文与运行时不一致

### 13.3 raw\_response 是否保留

建议：

- 短期保留，作为兼容字段或调试缓存
- 长期不再作为主路径字段

### 13.4 semantic\_traces 是否继续允许直接透传

建议：

- 短期允许
- 长期改为从 `turn_events` 派生

### 13.5 worth\_saving 与 WRITE/UPDATE 的优先级

当前存在 `worth_saving=False` 误伤 `MTP_WRITE/MTP_UPDATE` 的风险。

建议：

- 对携带 `write_focus` / `update_focus` 的 block，不应被 `worth_saving=False` 过滤
- 或由 TriggerManager 单独豁免

***

## 14. Phase 1 实施方案清单

Phase 1 的目标不是一次性完成双视图重构，而是先完成最关键的一步：

- 让“结构化事件流”成为新的主真相源
- 让感知层优先消费结构化字段
- 让 `_reconstruct_raw_assistant_text()` 退出主路径

本阶段默认采用**方案一**：

- `TurnRecord` 只作为运行时 DTO
- `LogicalBlock` 仍作为感知层领域对象
- 暂不推进 `LogicalBlock.turn = TurnRecord` 的结构收敛

### 14.1 Phase 1 范围

#### In Scope

- 新增 `TurnEvent`
- 在运行时采集 `turn_events`
- 扩展 `InteractionPayload`
- 感知层优先消费 `assistant_final_text + turn_events`
- 将 `MTPLogParser` 降级为 fallback
- 保持现有 `LogicalBlock` 扁平结构不变

#### Out of Scope

- `HistoryTranscriptBuilder` 正式落地
- `GenerationTranscriptBuilder` 正式落地
- `GenerationRequest` / `GenerationEngine` 签名重构
- `LogicalBlock` 内嵌 `TurnRecord`
- 立即删除 `raw_response` / `clean_response` / `assistant_message`

### 14.2 需要修改的文件

建议按以下顺序推进：

1. 模型层
   - `src/hivememory/engines/perception/models.py`
2. 运行时链路
   - `src/hivememory/patchouli/kernel/runtime/loop_executor.py`
   - `src/hivememory/patchouli/system.py`
3. 感知层入口
   - `src/hivememory/engines/perception/semantic_flow_perception_layer.py`
4. 兼容解析与辅助逻辑
   - `src/hivememory/patchouli/mtp/log_parser.py`
   - 如有需要，可新增 `MTPTraceReducer` 所在模块
5. 测试
   - `tests/unit/system/test_chat_logic.py`
   - `tests/unit/patchouli/kernel/test_loop_executor_stream.py`
   - `tests/unit/patchouli/test_eye.py`
   - 与 perception ingest 相关的单测文件

### 14.3 数据契约调整

#### Step 1：新增 `TurnEvent`

文件：

- `src/hivememory/engines/perception/models.py`

工作项：

- 新增 `TurnEvent`
- 暂不新增独立 `TurnRecord` 类型到 Buffer 层
- 保持 `TraceItem`、`LogicalBlock`、`InteractionPayload` 兼容

最小字段建议：

```python
class TurnEvent(BaseModel):
    kind: Literal["assistant_text", "mtp_command", "mtp_result"]
    sequence: int
    role: Literal["assistant", "user"]
    content: str

    verb: Optional[str] = None
    target: Optional[str] = None
    status: Optional[str] = None
```

说明：

- Phase 1 不追求事件模型一步到位
- 只保留历史视图重放与 trace 降维所需的最小字段

#### Step 2：扩展 `InteractionPayload`

文件：

- `src/hivememory/engines/perception/models.py`

工作项：

- 新增 `assistant_final_text: str = ""`
- 新增 `turn_events: list[TurnEvent] = Field(default_factory=list)`
- 保留：
  - `assistant_message`
  - `mtp_traces`

推荐优先级：

- 新逻辑优先读 `assistant_final_text + turn_events`
- 老逻辑回退到 `assistant_message + mtp_traces`

#### Step 3：扩展 `LogicalBlock`

文件：

- `src/hivememory/engines/perception/models.py`

工作项：

- 新增 `assistant_final_text: str = ""`
- 新增 `turn_events: list[TurnEvent] = Field(default_factory=list)`
- 保留兼容字段：
  - `raw_response`
  - `clean_response`
  - `semantic_traces`

约束：

- `clean_response` 在 Phase 1 中应尽量与 `assistant_final_text` 对齐
- `raw_response` 仅作为兼容缓存或调试字段

### 14.4 运行时改造步骤

#### Step 4：在 `LoopExecutor` 中直接采集事件

文件：

- `src/hivememory/patchouli/kernel/runtime/loop_executor.py`

目标：

- 不再依赖事后从 `messages` 反推本轮过程
- 在执行时直接记录 `turn_events`

工作项：

- 在单轮执行上下文中增加 `turn_events` 累积容器
- 每次写入 assistant 侧带 MTP 文本时，追加事件：
  - `kind="mtp_command"` 或 `kind="assistant_text"`
  - `role="assistant"`
- 每次写入 `[System MTP Execution Result]` 时，追加事件：
  - `kind="mtp_result"`
  - `role="user"`
- 在循环正常收敛后，记录 `assistant_final_text`

实现建议：

- Phase 1 可先把事件容器挂在 loop result 或中间上下文对象上
- 不要求立刻抽独立 collector 类

验收标准：

- 一轮包含 MTP 调用的对话能产出稳定有序的 `turn_events`
- 没有 MTP 的纯 assistant 回复也能产出合理事件或至少产出 `assistant_final_text`

#### Step 5：调整 `PatchouliSystem._chat_post_process()`

文件：

- `src/hivememory/patchouli/system.py`

目标：

- 不再以 `_reconstruct_raw_assistant_text()` 作为主来源

工作项：

- 优先从 loop 执行结果中读取：
  - `assistant_final_text`
  - `turn_events`
  - `semantic_traces`
- 将这些字段写入 `InteractionPayload`
- `assistant_message` 改为兼容字段：
  - 可继续写入旧值
  - 但不再作为感知层主入口依赖

建议：

- 暂时保留 `_reconstruct_raw_assistant_text()`
- 在函数注释中明确其已降级为兼容路径

验收标准：

- `PatchouliSystem` 提交给感知层的 payload 在新链路下不依赖 `assistant_message`

### 14.5 感知层改造步骤

#### Step 6：调整 `SemanticFlowPerceptionLayer.ingest_payload()`

文件：

- `src/hivememory/engines/perception/semantic_flow_perception_layer.py`

目标：

- 优先消费结构化字段
- 仅在缺失时回退到 `MTPLogParser`

建议分支：

```python
if payload.turn_events:
    assistant_final_text = payload.assistant_final_text
    semantic_traces = payload.semantic_traces or reducer.from_events(payload.turn_events)
    clean_text = assistant_final_text
else:
    clean_text, fallback_traces = MTPLogParser.parse(payload.assistant_message)
    assistant_final_text = clean_text
    semantic_traces = payload.mtp_traces or fallback_traces
```

工作项：

- 在新分支下构建 `LogicalBlock.assistant_final_text`
- 在新分支下构建 `LogicalBlock.turn_events`
- `clean_response` 对齐为 `assistant_final_text`
- `raw_response` 先保留旧兼容写法

验收标准：

- 当 `turn_events` 存在时，感知层不再依赖 `MTPLogParser` 提供主结果
- 老 payload 仍能正常 ingest

#### Step 7：新增 `MTPTraceReducer`

文件：

- 推荐新增，例如：
  - `src/hivememory/patchouli/mtp/trace_reducer.py`
  - 或 `src/hivememory/engines/perception/trace_reducer.py`

目标：

- 统一 `semantic_traces` 的事件降维逻辑

最小职责：

- 从 `turn_events` 生成 `TraceItem[]`
- Phase 1 只覆盖：
  - `SEARCH`
  - `READ`
  - `RUN`

可后置项：

- 更复杂的 `WRITE/UPDATE` 细化映射

验收标准：

- 新路径下 `semantic_traces` 可从事件流稳定生成
- 对老路径仍允许沿用已有 `payload.mtp_traces`

### 14.6 测试清单

#### 必补单测

1. `LoopExecutor` 事件采集
   - 有 MTP 指令时，产出有序 `turn_events`
   - 有 MTP 结果回填时，生成 `mtp_result` 事件
   - 无 MTP 时，`assistant_final_text` 正常

2. `PatchouliSystem` 后处理
   - `_chat_post_process()` 优先写入结构化字段
   - 老兼容字段仍存在

3. `SemanticFlowPerceptionLayer`
   - 有 `turn_events` 时不走 parser 主路径
   - 无 `turn_events` 时回退到 `MTPLogParser`
   - `LogicalBlock.clean_response == assistant_final_text`

4. `semantic_traces` 降维
   - `SEARCH/READ/RUN` 能从 `turn_events` 派生

#### 推荐回归测试

- `tests/unit/system/test_chat_logic.py`
  - 确认聊天主链路未回归
- `tests/unit/system/test_chat_stream_memory_refs_schema.py`
  - 确认 SSE 相关逻辑未被连带破坏
- `tests/unit/patchouli/kernel/test_loop_executor_stream.py`
  - 确认流式路径与事件采集协同正常

### 14.7 实施顺序建议

建议严格按下列顺序推进，避免一次性同时改太多层：

1. 先改 `models.py`
   - 加 `TurnEvent`
   - 扩展 `InteractionPayload`
   - 扩展 `LogicalBlock`
2. 再改 `loop_executor.py`
   - 让运行时真正产出结构化事件
3. 再改 `system.py`
   - 让 payload 提交新字段
4. 再改 `semantic_flow_perception_layer.py`
   - 让感知层优先消费新字段
5. 再补 `MTPTraceReducer`
   - 收敛 trace 生成逻辑
6. 最后补测试并回归

### 14.8 Phase 1 完成标准

满足以下条件，可认为 Phase 1 完成：

1. 一轮对话的结构化事件流已能从运行时稳定产出
2. 感知层已优先消费 `assistant_final_text + turn_events`
3. `MTPLogParser` 已退为 fallback
4. `_reconstruct_raw_assistant_text()` 不再是主路径依赖
5. 现有聊天主链路与 SSE 相关测试通过
6. 未引入对 Generation Engine 的大范围破坏性变更

### 14.9 Phase 1 结束后的预期状态

Phase 1 完成后，系统将进入一个更稳定的中间态：

- 真相源已从“拼接字符串”迁移为“结构化事件流”
- `LogicalBlock` 已具备承载双视图所需的核心字段
- 历史视图和生成视图尚未完全拆开，但数据基础已经具备
- 后续可以低风险进入 Phase 2：
  - 历史视图 builder 正式落地
  - `PerceptionContextConverter` 从 `clean_response` 迁移到 `turn_events`

---

## 15. Phase 2 实施方案清单

Phase 2 的目标不是处理记忆生成视图，而是**正式落地历史消息视图**。

在 Phase 1 完成后，系统已经具备以下基础：

- `LoopExecutor` 可以稳定产出 `turn_events`
- `InteractionPayload` 已携带 `assistant_final_text + turn_events`
- `LogicalBlock` 已能保存 `assistant_final_text + turn_events`
- `TurnEvent.render_as` 已提供轻量渲染提示

因此，Phase 2 的核心目标是：

- 新增 `HistoryTranscriptBuilder`
- 让历史消息构建从 `clean_response` 迁移到 `turn_events`
- 保持多智能体身份前缀逻辑继续生效
- 保持旧 block 与旧调用链可兼容

### 15.1 Phase 2 范围

#### In Scope

- 新增 `HistoryTranscriptBuilder`
- 为 `TurnEvent.render_as` 定义统一渲染规则
- 改造 `PerceptionContextConverter.blocks_to_messages()`
- 让 block 级结构化历史重放成为主路径
- 保留旧 `clean_response` 渲染作为 fallback

#### Out of Scope

- `GenerationTranscriptBuilder`
- `GenerationRequest` / `GenerationEngine` 的上下文签名调整
- `LogicalBlock.turn = TurnRecord` 的结构收敛
- 删除 `clean_response` / `raw_response`
- 调整 `LibrarianCore` 的 generation 输入格式

### 15.2 Phase 2 要解决的具体问题

当前 [`PerceptionContextConverter`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/perception/context_converter.py) 仍然是：

- 每个 block 输出一条 user 消息
- 再输出一条 assistant 消息
- assistant 内容直接来自 `clean_response`

这会丢掉：

- assistant 发出的 MTP 指令
- 系统回填的 MTP 执行结果
- `role=user` 的系统结果消息边界

Phase 2 需要把它改成“按事件流重放”的模式。

### 15.3 建议新增的模块

建议新增：

- `src/hivememory/engines/perception/history_transcript_builder.py`

职责：

- 从 `LogicalBlock` 列表构建历史消息视图
- 统一处理：
  - `turn_events` 顺序重放
  - `render_as` 前缀渲染
  - 多智能体身份前缀
  - fallback 路径

建议接口：

```python
class HistoryTranscriptBuilder:
    def build_messages(
        self,
        blocks: list[LogicalBlock],
        current_agent_id: str = "default",
    ) -> list[dict[str, str]]:
        ...
```

### 15.4 历史视图渲染规则

#### Block 级顺序

对每个 `LogicalBlock`：

1. 先输出 `user_query`
2. 若 `turn_events` 非空，则按 `sequence` 依次输出事件
3. 若 `turn_events` 为空，则回退到旧模式：
   - 输出 `clean_response`

#### Event 级角色映射

- `assistant_text` -> `role="assistant"`
- `mtp_command` -> `role="assistant"`
- `mtp_result` -> `role="user"`

#### `render_as` 渲染规则

`TurnEvent.content` 仍保存消息主体，builder 决定是否补系统前缀。

建议规则：

- `render_as="plain"`
  - 原样输出 `content`
- `render_as="system_mtp_result"`
  - 输出：
    - `[System MTP Execution Result]\n{content}`
- `render_as="system_ipc_return"`
  - 输出：
    - `[System IPC Return]\n{content}`

这样可以保证：

- `turn_events` 仍保持轻量
- 历史视图仍能重放系统消息边界

#### 多智能体身份前缀

沿用当前 `PerceptionContextConverter` 的语义，但只对 assistant 侧事件生效：

- 当 `event.role == "assistant"` 且
- `block.identity.agent_id` 不为 `default` / `omni_doll` 且
- `block.identity.agent_id != current_agent_id`

则在内容前加：

```text
[From: {agent_id}]
```

注意：

- 不应给 `mtp_result` 事件加身份前缀
- 因为它们表示系统回填消息，不是某个 agent 的自然发言

### 15.5 需要修改的文件

建议按以下顺序推进：

1. 新增 builder
   - `src/hivememory/engines/perception/history_transcript_builder.py`
2. 接入转换器
   - `src/hivememory/engines/perception/context_converter.py`
3. 如有必要，补模型辅助方法
   - `src/hivememory/engines/perception/models.py`
4. 测试
   - `tests/unit/engines/perception/test_history_transcript_builder.py`
   - `tests/unit/system/test_chat_logic.py`
   - 任何依赖 `blocks_to_messages()` 的现有测试

### 15.6 详细实施步骤

#### Step 1：实现 `HistoryTranscriptBuilder`

文件：

- `src/hivememory/engines/perception/history_transcript_builder.py`

最小职责：

- 遍历 `blocks`
- 输出 user 消息
- 若存在 `turn_events`，按顺序输出结构化事件
- 若不存在 `turn_events`，回退到 `clean_response`

建议内部拆分的方法：

- `_render_block()`
- `_render_event()`
- `_apply_agent_prefix()`
- `_apply_system_prefix()`

最小伪代码：

```python
def build_messages(blocks, current_agent_id="default"):
    messages = []
    for block in blocks:
        if block.user_query:
            messages.append({"role": "user", "content": block.user_query})

        if block.turn_events:
            for event in sorted(block.turn_events, key=lambda e: e.sequence):
                content = self._render_event(event)
                if event.role == "assistant":
                    content = self._apply_agent_prefix(content, block.identity, current_agent_id)
                messages.append({"role": event.role, "content": content})
        elif block.clean_response:
            content = self._apply_agent_prefix(block.clean_response, block.identity, current_agent_id)
            messages.append({"role": "assistant", "content": content})
    return messages
```

#### Step 2：接管 `PerceptionContextConverter.blocks_to_messages()`

文件：

- `src/hivememory/engines/perception/context_converter.py`

目标：

- `blocks_to_messages()` 不再手写 user/assistant 二元拼接
- 改为委托 `HistoryTranscriptBuilder`

建议方式：

- 保留现有公开签名不变
- 内部直接调用 builder

好处：

- 不破坏上游调用点
- 历史消息语义只在一个地方维护

#### Step 3：明确 fallback 策略

Phase 2 必须同时兼容：

- 新 block：有 `turn_events`
- 旧 block：没有 `turn_events`

建议 fallback 顺序：

1. `turn_events`
2. `clean_response`
3. 若极端情况下二者都空，则不输出 assistant 消息

注意：

- `raw_response` 不建议作为 Phase 2 的默认 fallback
- 因为其语义仍偏兼容/调试字段

#### Step 4：处理历史兼容中的多角色场景

现有多角色逻辑在 [`context_converter.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/perception/context_converter.py#L54-L64) 中只作用于 `clean_response`。

Phase 2 需要确认：

- assistant_text 与 mtp_command 都走同样的 agent prefix 逻辑
- mtp_result 不走该逻辑

验收标准：

- 非当前 agent 的历史指令与自然语言回复都带 `[From: ...]`
- 系统结果消息不被误标识成某个 agent 发言

#### Step 5：保守处理 `to_stream_messages()`

[`LogicalBlock.to_stream_messages()`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/perception/models.py#L387-L455) 当前仍是：

- `user_query + clean_response`

Phase 2 建议：

- 暂不修改它的主语义
- 保持它服务于旧路径或 generation 兼容
- 先只把历史消息主路径切到 `HistoryTranscriptBuilder`

这样可避免：

- 历史视图与 generation 视图在同一接口里再次混杂

### 15.7 测试清单

#### 必补单测

1. `HistoryTranscriptBuilder` 基础重放
   - 一个 block，无 `turn_events`
   - 回退到 `clean_response`

2. `HistoryTranscriptBuilder` 结构化重放
   - `assistant_text + mtp_command + mtp_result + assistant_text`
   - 顺序必须与 `sequence` 一致

3. `render_as` 前缀渲染
   - `system_mtp_result` 正确补 `[System MTP Execution Result]`
   - `system_ipc_return` 正确补 `[System IPC Return]`

4. 多智能体前缀
   - 非当前 agent 的 assistant 事件带 `[From: ...]`
   - `mtp_result` 不带 `[From: ...]`

5. 混合兼容场景
   - 一个旧 block + 一个新 block 混合输入时，输出顺序正确

#### 推荐回归测试

- 复跑 `tests/unit/system/test_chat_logic.py`
- 复跑任何依赖 `PerceptionContextConverter.blocks_to_messages()` 的测试
- 如存在话题历史回放相关测试，也应补跑

### 15.8 实施顺序建议

建议按下列顺序推进：

1. 先新增 `HistoryTranscriptBuilder`
2. 先写 builder 单测
3. 再让 `PerceptionContextConverter` 委托 builder
4. 再跑聊天主链路回归
5. 最后观察是否需要为 `models.py` 补少量辅助方法

这样可以把风险聚焦在：

- 历史视图渲染
- 而不波及 generation 链路

### 15.9 Phase 2 完成标准

满足以下条件，可认为 Phase 2 完成：

1. 历史消息主路径已基于 `turn_events` 构建
2. `render_as` 已能稳定还原系统结果消息前缀
3. 多智能体身份前缀与系统消息前缀不会互相污染
4. 没有 `turn_events` 的旧 block 仍能正常回退
5. `PerceptionContextConverter.blocks_to_messages()` 对外签名保持兼容
6. 聊天主链路相关回归测试通过

### 15.10 Phase 2 结束后的预期状态

Phase 2 完成后，系统会进入一个更清晰的中间态：

- 历史消息视图已正式与 `clean_response` 解耦
- block 级结构化事件成为历史回放的主数据源
- `render_as` 成为系统消息边界的标准化提示
- generation 视图仍未正式拆出，但不会再与历史视图共享同一套构造逻辑

这时再进入 Phase 3，会更自然：

- 新增 `GenerationTranscriptBuilder`
- 将 `state_summary + semantic_traces + assistant_final_text` 正式接入 generation 上下文

---

## 16. Phase 3 实施方案清单

Phase 3 的目标是**正式落地记忆生成视图**，让 generation 链路不再复用历史消息视图，也不再依赖 `LogicalBlock.to_stream_messages()` 这种面向 user/assistant 二元投影的旧接口。

在当前代码状态下，Phase 2 已完成：

- `HistoryTranscriptBuilder` 已接管历史消息主路径
- `PerceptionContextConverter.blocks_to_messages()` 已基于 `turn_events` 构建历史视图
- `LogicalBlock` 已保存：
  - `assistant_final_text`
  - `turn_events`
  - `semantic_traces`
  - `state_summary` 仍在 `ArchivePayload` 层

但 generation 链路当前仍然是：

```text
LibrarianCore._on_generate_memory()
    -> _blocks_to_messages()
    -> block.to_stream_messages()
    -> GenerationRequest(context_messages)
    -> MemoryGenerationEngine._format_transcript(messages)
```

这会继续丢掉：

- `state_summary`
- `semantic_traces`
- `assistant_final_text` 与历史视图的语义区分
- generation 视图与历史视图的边界

因此，Phase 3 的核心目标是：

- 新增 `GenerationTranscriptBuilder`
- 将 generation 输入从 `context_messages` 升级为 generation 专用 context
- 把 `state_summary + semantic_traces + assistant_final_text` 正式接入 generation 主路径
- 保持 Mode A / B / C 的行为兼容

### 16.1 Phase 3 范围

#### In Scope

- 新增 `GenerationTranscriptBuilder`
- 新增 generation 专用 context 模型
- 改造 `LibrarianCore._on_generate_memory()`
- 改造 `GenerationRequest`
- 改造 `MemoryGenerationEngine` 的 transcript 构建入口
- 保持 `write_focus` / `update_focus` 三模式语义不变

#### Out of Scope

- 删除 `LogicalBlock.to_stream_messages()`
- `LogicalBlock.turn = TurnRecord` 的模型收敛
- Phase 4 的兼容字段清理
- 重写 extractor / deduplicator 的业务逻辑

### 16.2 当前 generation 链路的主要问题

根据当前实现：

- [`librarian_core.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/patchouli/kernel/librarian_core.py#L161-L214) 仍通过 `_blocks_to_messages()` 产出 `List[StreamMessage]`
- [`_blocks_to_messages()`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/patchouli/kernel/librarian_core.py#L236-L254) 直接调用 `block.to_stream_messages()`
- [`LogicalBlock.to_stream_messages()`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/perception/models.py#L387-L455) 在 Kernel 模式下仍只输出 `user_query + clean_response`
- [`GenerationRequest`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/generation/models.py#L124-L152) 当前只接收 `context_messages`
- [`MemoryGenerationEngine._format_transcript()`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/generation/engine.py#L339-L365) 也只把 `StreamMessage` 扁平化成 transcript

这意味着当前 generation 仍看不到：

- 话题 `state_summary`
- `semantic_traces`
- `assistant_final_text` 作为 generation 专用响应字段

### 16.3 Phase 3 的设计目标

Phase 3 的设计目标不是把历史视图再包装一层，而是提供**generation 专用的去噪语义上下文**。

generation 视图应满足：

- 保留：
  - `state_summary`
  - `user_query`
  - `semantic_traces`
  - `assistant_final_text`
- 丢弃：
  - `mtp_result` 全量正文
  - `READ` 返回原文
  - `SEARCH` 结果正文
  - XML / IPC 回填原文

也就是说：

- 历史视图强调“模型当时看到了什么”
- generation 视图强调“这一轮发生了什么语义动作，以及最终说了什么”

### 16.4 建议新增的模型

建议在 [`generation/models.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/generation/models.py) 中新增：

```python
class GenerationTurn(BaseModel):
    user_query: str
    assistant_final_text: str = ""
    trace_summaries: list[str] = Field(default_factory=list)
    identity: Identity

class GenerationContext(BaseModel):
    state_summary: str = ""
    turns: list[GenerationTurn] = Field(default_factory=list)
```

字段职责：

- `GenerationTurn`
  - generation 视图中的最小“回合”单位
- `trace_summaries`
  - 从 `semantic_traces` 或 `turn_events` 降维出来的动作摘要
- `GenerationContext`
  - generation 链路的主上下文模型
  - 显式携带 `state_summary`

### 16.5 对 `GenerationRequest` 的建议调整

当前 `GenerationRequest`：

- `context_messages: List[StreamMessage]`
- `write_focus`
- `update_focus`

建议演化为兼容式双入口：

```python
class GenerationRequest(BaseModel):
    context_messages: List[StreamMessage] = Field(default_factory=list)  # 兼容字段
    context: Optional[GenerationContext] = None
    write_focus: Optional[WriteFocus] = None
    update_focus: Optional[UpdateFocus] = None
```

推荐读取优先级：

1. `context`
2. `context_messages`

这样可以避免：

- 一步到位破坏所有现有调用点
- Phase 3 和 Phase 4 被绑死在同一次大改里

### 16.6 建议新增的模块

建议新增：

- `src/hivememory/engines/generation/generation_transcript_builder.py`

职责：

- 从 `LogicalBlock[] + state_summary` 构建 `GenerationContext`
- 或直接渲染 generation transcript

建议接口：

```python
class GenerationTranscriptBuilder:
    def build_context(
        self,
        blocks: list[LogicalBlock],
        state_summary: str = "",
    ) -> GenerationContext:
        ...

    def build_transcript(self, context: GenerationContext) -> str:
        ...
```

### 16.7 generation 视图的渲染规则

#### Block -> GenerationTurn

每个 block 生成一个 `GenerationTurn`：

- `user_query` <- `block.user_query`
- `assistant_final_text` <- `block.assistant_final_text or block.clean_response`
- `trace_summaries` <- 从 `block.semantic_traces` 生成
- `identity` <- `block.identity`

#### `state_summary`

- 不再丢弃
- 直接进入 `GenerationContext.state_summary`

#### `trace_summaries` 建议格式

建议维持简洁、去噪、稳定：

- `SEARCH: "authentication flow"`
- `READ: alias_x`
- `RUN: tool_x (success)`

建议：

- Phase 3 优先使用 `semantic_traces`
- 不直接从 `turn_events` 再次拼 generation 视图
- 避免 generation builder 和 history builder 又重新共享同一渲染逻辑

### 16.8 建议的 transcript 格式

当前 `MemoryGenerationEngine._format_transcript()` 是把 `StreamMessage` 扁平化成：

```text
[User]: ...
[Assistant]: ...
```

Phase 3 建议改成 generation 专用文本格式：

```text
[Topic State]
{state_summary}

[Turn 1]
[User]: ...
[Actions]:
- SEARCH: "..."
- READ: alias_x
[Assistant]: ...

[Turn 2]
[User]: ...
[Actions]:
- RUN: tool_x (success)
[Assistant]: ...
```

这样做的收益：

- `state_summary` 正式进入提取上下文
- action 语义进入 transcript
- 不会把大量系统结果正文重新塞回 generation

### 16.9 需要修改的文件

建议按以下顺序推进：

1. generation 模型层
   - `src/hivememory/engines/generation/models.py`
2. 新增 builder
   - `src/hivememory/engines/generation/generation_transcript_builder.py`
3. perception -> generation 入口
   - `src/hivememory/patchouli/kernel/librarian_core.py`
4. generation engine
   - `src/hivememory/engines/generation/engine.py`
5. 测试
   - `tests/unit/generation/test_generation_transcript_builder.py`
   - `tests/unit/patchouli/kernel/test_librarian_core_generation_context.py`
   - 相关 generation engine 单测

### 16.10 详细实施步骤

#### Step 1：新增 `GenerationContext` / `GenerationTurn`

文件：

- `src/hivememory/engines/generation/models.py`

工作项：

- 新增 `GenerationTurn`
- 新增 `GenerationContext`
- 扩展 `GenerationRequest`，支持 `context`

验收标准：

- 新模型可被单独实例化
- 不破坏现有仅传 `context_messages` 的调用

#### Step 2：实现 `GenerationTranscriptBuilder`

文件：

- `src/hivememory/engines/generation/generation_transcript_builder.py`

最小职责：

- `build_context(blocks, state_summary)`
- `build_transcript(context)`

建议内部拆分：

- `_block_to_turn()`
- `_trace_to_summary()`
- `_format_context()`

最小伪代码：

```python
def build_context(blocks, state_summary=""):
    turns = []
    for block in blocks:
        turns.append(
            GenerationTurn(
                user_query=block.user_query,
                assistant_final_text=block.assistant_final_text or block.clean_response,
                trace_summaries=[self._trace_to_summary(t) for t in block.semantic_traces],
                identity=block.identity,
            )
        )
    return GenerationContext(state_summary=state_summary, turns=turns)
```

#### Step 3：改造 `LibrarianCore`

文件：

- `src/hivememory/patchouli/kernel/librarian_core.py`

目标：

- `_on_generate_memory()` 不再优先调用 `_blocks_to_messages()`
- 改为优先构建 `GenerationContext`

建议：

- 保留 `_blocks_to_messages()` 作为兼容壳
- 新增：
  - `_build_generation_context()`

新的处理路径建议为：

```python
context = self._build_generation_context(blocks, state_summary)
request = GenerationRequest(
    context=context,
    write_focus=...,
    update_focus=...,
)
```

验收标准：

- Mode A / B / C 都能成功构造带 `context` 的 request
- `state_summary` 不再在 LibrarianCore 被丢弃

#### Step 4：改造 `MemoryGenerationEngine`

文件：

- `src/hivememory/engines/generation/engine.py`

目标：

- `process()` 优先消费 `request.context`
- `_format_transcript()` 支持 generation 专用 context

建议实现方式：

- 新增：
  - `_format_generation_context(context: GenerationContext) -> str`
- 保留：
  - `_format_transcript(messages: List[StreamMessage]) -> str`

推荐优先级：

1. 若 `request.context` 存在，则走 `_format_generation_context()`
2. 否则回退到旧 `_format_transcript()`

这样可以平滑兼容：

- 旧调用点
- README 示例
- 可能存在的未迁移测试

#### Step 5：保守处理 `LogicalBlock.to_stream_messages()`

Phase 3 不建议立刻修改 [`LogicalBlock.to_stream_messages()`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/perception/models.py#L387-L455)。

理由：

- 它仍可作为兼容壳存在
- 它不是 Phase 3 的目标接口
- 直接改它容易让历史视图和 generation 视图再次耦合

### 16.11 测试清单

#### 必补单测

1. `GenerationTranscriptBuilder` 基础构建
   - block -> turn
   - state_summary 正常进入 context

2. trace 摘要映射
   - `SEARCH` -> `SEARCH: "..."`
   - `READ` -> `READ: alias_x`
   - `RUN` -> `RUN: tool_x (success)`

3. `LibrarianCore` 新路径
   - `_on_generate_memory()` 构建 `GenerationRequest(context=...)`
   - `state_summary` 未丢失

4. `MemoryGenerationEngine` 新入口
   - `request.context` 存在时走新格式化路径
   - `request.context_messages` 仍可兼容旧路径

5. Mode B / C 兼容
   - WRITE / UPDATE 模式下仍能携带 `write_focus` / `update_focus`
   - transcript 使用 generation context

#### 推荐回归测试

- generation engine 现有单测
- `tests/unit/system/test_chat_logic.py`
- 任何依赖 `LibrarianCore._on_generate_memory()` 的测试

### 16.12 实施顺序建议

建议按下列顺序推进：

1. 先扩展 `GenerationRequest`
2. 先写 `GenerationTranscriptBuilder` 单测
3. 再改 `LibrarianCore`
4. 再改 `MemoryGenerationEngine`
5. 最后跑 Mode A / B / C 回归

这样可以把风险集中在：

- generation 上下文建模
- 而不是先改底层 block / perception 模型

### 16.13 Phase 3 完成标准

满足以下条件，可认为 Phase 3 完成：

1. generation 主路径已不再依赖 `block.to_stream_messages()`
2. `state_summary` 已正式进入 generation transcript
3. `semantic_traces` 已转化为 generation action 摘要
4. `assistant_final_text` 已替代 `clean_response` 成为 generation 视图主响应字段
5. `GenerationRequest` 同时兼容 `context` 与 `context_messages`
6. Mode A / B / C 回归测试通过

### 16.14 Phase 3 结束后的预期状态

Phase 3 完成后，系统会进入一个真正“双视图落地”的状态：

- 历史视图由 `HistoryTranscriptBuilder` 负责
- generation 视图由 `GenerationTranscriptBuilder` 负责
- 历史回放与记忆提取不再共享同一套消息构造逻辑
- `state_summary`、`semantic_traces`、`assistant_final_text` 都进入了 generation 主路径

这时再进入 Phase 4，会更合理：

- 清理 `raw_response`
- 清理 `clean_response`
- 清理 `assistant_message`
- 收敛 `context_messages` 兼容层

---

## 17. 总结

本次重构的关键结论是：

- 不应继续尝试修补 `_reconstruct_raw_assistant_text()` 来适应新的 `role=user` 模式
- 应承认 transcript 已经自然分裂为两套视图：
  - 历史消息视图
  - 记忆生成视图
- 两套视图不需要维护两份真相源
- 它们应共享同一份结构化事件流

最终目标应当是：

- `LoopExecutor` 产出 `assistant_final_text + turn_events`
- 感知层保存 `assistant_final_text + turn_events + semantic_traces`
- `HistoryTranscriptBuilder` 负责历史回放
- `GenerationTranscriptBuilder` 负责生成视图
- `MTPLogParser` 从主路径退为 fallback

这会让：

- 运行时语义更清晰
- 感知层字段职责更稳定
- Generation 真正看到有价值的动作信息
- 后续继续做模块化重构时，文本链路不再反复混乱
