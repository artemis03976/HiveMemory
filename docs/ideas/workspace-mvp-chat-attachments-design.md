---
title: Chat Attachments 初步设计备忘
status: idea
owner: system-patchouli-alice
scope: chat-attachment-runtime-content-and-context-assembly
related_current:
  - ../ROADMAP.md
  - ../architecture/workspace.md
  - ../architecture/boundaries.md
  - ../patchouli/artifacts.md
  - ../patchouli/memory-library.md
related_plans:
  - ../archive/plans/v0.6.2-workspace-mvp.md
last_reviewed: 2026-09-01
---

# Chat Attachments 初步设计备忘

本文只记录尚未形成正式实施计划的 Chat Attachment 设计。附件功能依赖已经完成并稳定的 Workspace MVP 公共契约，但本文不重新定义 Workspace、身份作用域或资源管理模型；Workspace 当前事实以 [Workspace 架构](../architecture/workspace.md) 为准，W0 的实施历史以[归档 Plan](../archive/plans/v0.6.2-workspace-mvp.md)为准。

## 1. 背景与目标

当前 Agent 尚不支持多模态，对图像的记忆化能力也未建立，因此附件 MVP 以文字内容为主。用户可以在 Chat 中上传文件，系统解析出可供 Agent 使用的文字表示；附件本身不立即成为 Memory，也不承诺永久持久化。

目标是让附件在一个话题和相关对话过程中能够被反复引用，而不是上传后只在一次请求中解析并一次性塞入上下文。用户引用的是当前 Store 存活期内稳定的 `AttachmentRef`，Context Compiler 在每次需要时读取已确认可用的解析表示并组装上下文。

## 2. 功能边界

### 2.1 本阶段考虑

- 上传文字附件；
- 保存原始内容的进程内表示和原始元数据；
- 运行解析器生成文字 representation；
- 以稳定 ref 选择附件和具体 representation；
- 在当前 Chat 中选择、重复选择和取消选择；
- 由 Context Compiler 将已选择内容编译进当前请求；
- 只有实际参与 Memory CREATE/UPDATE 的内容才允许在 Materialization 阶段提升为 Artifact 证据。

### 2.2 不在本文制定正式计划

- 底层资源管理、身份隔离或运行时容器方案；
- Memory schema v2 的 ownership 与 visibility 设计；
- v0.7.0 长外源文档解析与完整文档摄取流程；
- OCR、图像、多模态输入和图像记忆化；
- 跨进程、重启后的附件恢复；
- 完整 Durable Artifact、对象存储、保留期和垃圾回收策略；
- 任何尚未冻结的上传 API、权限 grant 或跨话题共享协议。

## 3. 附件生命周期

```text
upload
  -> raw content
  -> parse
  -> READY / FAILED
  -> AttachmentRef
  -> Context Compiler
  -> successful interaction commit
  -> TopicAssetBinding
  -> optional ContextAttachmentUse detail
  -> optional Materialization promotion
```

基本规则：

- 上传先形成原始内容记录，解析成功后才产生可供上下文使用的 representation；
- 只有 `READY` representation 可以被引用和编译；
- 解析失败直接反馈给用户，MVP 不自动重试；用户重新上传时创建新的逻辑附件；
- representation 必须具有固定的 revision 和 content hash；编译与物化不能回查“最新内容”；
- 附件移除会终止对应 ref 的后续 resolve/acquire 能力，但不删除 Topic 中已经成立的真实使用事实；进程结束后系统不保留可在新进程恢复的既有 asset/ref，用户必须重新上传并建立新关系；
- Materialization 使用的 representation 需要冻结其 revision/hash，以便后续 provenance 可重现。

## 4. Attachment 与 Chat、Topic、Run

上传后附件不自动绑定到某个 Topic。零 binding 的 WorkspaceAsset 是关系图中的合法
orphan：它不是失败资产，也不因此立即回收，但在用户真正通过一轮对话使用前不能建立
任何 Topic 关系。上传、最近资产、当前活跃 Topic 或进程级 pending list 都不能替用户
猜测关系。

`TopicAssetBinding` 是 Topic 级“真实使用过”的权威事实。只有 Chat 路由已经确定、
用户在本轮显式选择 READY ref、Context Compiler 已通过该 ref acquire representation
lease 并把内容用于本轮对话、且该轮对话成功完成后，才把 ref 与本轮 block 在同一个
Interaction commit 中绑定到最终 Topic。同一 Topic 再次使用同一资产只幂等命中既有
关系。新上传附件不能仅因上传动作自动挂载；如果产品希望上传后立即使用，前端必须把它
作为本轮显式选择提交，并且仍要经过 acquire、实际使用和成功 Interaction commit。

`ContextAttachmentUse` 可以记录具体 representation revision、content hash、locator、
纳入 token 数和 compile 诊断，但不再承担一套与 `TopicAssetBinding` 竞争“是否使用过”
的判定。若某个 ref 未被本轮有效接纳，当前 Interaction 必须失败或明确拒绝该选择，不能
一边完成对话一边把它留在模糊的“选择过但未使用”中间状态。

附件记录只在确实需要时携带稳定的 `interaction_id`；不为没有实际消费者的裸 `run_id` 预留字段。若未来需要记录某个 Agent 执行实例，应使用明确命名的 provenance 字段，而不是改变附件引用的业务语义。

Topic compact 保留 binding；Topic 删除、结算或真实 evict 会使关系随整个
SemanticBuffer 消失。settlement 必须在 buffer 清理前把全部 binding refs 冻结进任务，
因此后续 Materialization 不再依赖 Topic 存活。附件自身的可用性、移除和过期由其
运行时生命周期负责，也不应改写已经由 task/lease 冻结的消费事实。

asset remove 不调用 Patchouli，也不清理 Topic binding。两个 Store 不组成跨 Store
事务：WorkspaceAssetStore 只决定 ref 当前是否仍可 acquire，ShortTermMemoryStore 只保存
成功 Interaction 的历史使用事实。remove 早于 acquire 时，本轮附件使用失败且不建立
binding；acquire 早于 remove 时，已有 lease 可以完成本轮使用，成功 Interaction 仍可建立
binding，但该 ref 后续不可再次 acquire。这里不增加共同锁、两阶段提交、补偿流程或专用
binding cleanup controller。

## 5. AttachmentRef 与解析 representation

建议的最小模型如下：

```text
Attachment
  attachment_id
  original_metadata
  lifecycle_state
  created_at

AttachmentRepresentation
  representation_id
  attachment_id
  kind: RAW | EXTRACTED_TEXT
  revision
  content
  content_hash
  state: PENDING | PROCESSING | READY | FAILED
  error?
```

用户引用：

```text
AttachmentRef
  attachment_id
  representation_id?
```

系统使用明细（可选，不取代 TopicAssetBinding）：

```text
ContextAttachmentUse
  interaction_id
  attachment_id
  representation_id
  revision
  content_hash
  locators
  included_tokens
  compile_status
```

`AttachmentRef` 只表示用户选择的内容入口，不应携带未经验证的物理路径或任意内容。
系统 acquire ref 时必须再次确认附件仍存在、representation 属于该附件、状态为 READY 且
revision/hash 一致。本轮 Interaction 在实际使用期间持有 lease，成功 commit 后才保存
Topic binding。settlement task 复制当前 Store 存活期内的 opaque ref，Materialization
consumer 再通过 Store acquire 内容并在消费期间持有新的 representation lease。

## 6. Context Compiler

Context Compiler 需要在独立 Attachment Plan 中裁定以下行为：

- 用户选择附件的请求协议，以及当前 run 是否默认自动附加新上传内容；
- 同一附件多个 representation 的选择规则；
- token budget、分片、截断和多附件排序；
- 附件内容在 prompt 中的结构化标记和 source locator；
- representation 不可用、引用无效或编译失败时的用户可见错误；
- 同一附件重复引用时的去重语义；
- `ContextAttachmentUse` 的保留时间和查询用途。

Compiler 的输出应明确记录实际纳入上下文的 representation revision/hash 与定位信息，不能只留下一个附件 ID。

## 7. Materialization 与 Artifact promotion

上传、解析、选择或单纯编译都不自动生成 Artifact。只有某个 representation 实际进入 Agent 上下文，并且该内容参与 Memory CREATE/UPDATE 时，才允许在 Materialization 阶段被提升为对应 Artifact 证据。

promotion 必须使用已经冻结的 representation revision/hash，并对同一次 Materialization retry 保持幂等；不得在 retry 时读取附件的最新表示。若没有发生 Memory 生成或更新，不承诺形成 Artifact。

附件是否应解释为 `DocumentArtifact` 仍是开放问题。它与 v0.7.0 外源文档解析共享“记录外源信息载体并支持溯源”的性质，但附件不一定进入 Memory，也可能包含非文档内容，因此类型映射和多级定位应留给独立的 Attachment/Document Ingestion Plan 裁定。

## 8. 失败矩阵

至少覆盖以下失败路径：

| 阶段 | 失败 | 处理原则 |
|:---|:---|:---|
| upload | 文件接收失败 | 返回明确错误，不创建可用 ref |
| parse | 编码不支持或解析器失败 | 标记 representation 为 FAILED，提示用户重新上传 |
| resolve | 附件已移除、过期或 ref 无效 | 稳定拒绝，不回退到同名内容 |
| interaction | remove 早于本轮 acquire | acquire 失败，本轮不使用附件，也不建立 binding |
| interaction | acquire 后、成功 commit 前发生 remove | 已有 lease 完成本轮使用；成功 commit 保留历史 binding，后续 acquire 拒绝 |
| compile | representation 非 READY | 不进入 prompt，返回可解释错误 |
| compile | token budget 超限或编译失败 | 返回编译诊断，不静默截断关键内容 |
| materialization | promotion 失败 | 保留原始失败信息，按 Materialization 重试语义处理 |
| retry | representation revision/hash 已变化 | 拒绝使用新版本，要求重新选择或重新上传 |

## 9. 尚未形成正式计划的设计问题

- 上传 API、文件大小、MIME 类型和文本编码限制；
- 同步解析还是异步解析，以及 parser 资源预算；
- 解析器注册、版本和失败错误的安全摘要；
- AttachmentRef 的外部协议与防篡改校验；
- Context Compiler 的 token 预算、分片和引用定位；
- 附件在进程内的 retention、主动移除和容量上限；
- `ContextAttachmentUse` 与 Topic/Interaction 历史的保留边界；
- settlement 已冻结历史 binding ref、但 Materialization 开始前资产已 REMOVED 时，是跳过该来源并记录诊断，还是令本次 Materialization 失败；
- source locator 与内容 hash 的组合方式；
- `DocumentArtifact` promotion 是否适用，以及与 v0.7.0 文档摄取的复用边界。

## 10. 与正式计划的关系

本文只作为 Chat Attachment idea，不是实现承诺。附件功能将在前置公共契约稳定后，另行制定上传、解析、引用、上下文编译和 Artifact promotion 的正式实施计划。
