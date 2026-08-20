---
title: Chat Attachments 初步设计备忘
status: idea
owner: system-patchouli-alice
scope: chat-attachment-runtime-content-and-context-assembly
related_current:
  - ../ROADMAP.md
  - ../architecture/boundaries.md
  - ../patchouli/artifacts.md
  - ../patchouli/memory-library.md
related_plans:
  - ../plans/v0.6.2-workspace-mvp.md
last_reviewed: 2026-08-20
---

# Chat Attachments 初步设计备忘

本文只记录尚未形成正式实施计划的 Chat Attachment 设计。附件功能依赖已经完成并稳定的 Workspace MVP 公共契约，但本文不重新定义 Workspace、身份作用域或资源管理模型；Workspace 的设计真相源是 [v0.6.2 W0 Workspace MVP 正式 Plan](../plans/v0.6.2-workspace-mvp.md)。

## 1. 背景与目标

当前 Agent 尚不支持多模态，对图像的记忆化能力也未建立，因此附件 MVP 以文字内容为主。用户可以在 Chat 中上传文件，系统解析出可供 Agent 使用的文字表示；附件本身不立即成为 Memory，也不承诺永久持久化。

目标是让附件在一个话题和相关对话过程中能够被反复引用，而不是上传后只在一次请求中解析并一次性塞入上下文。用户引用的是稳定的 `AttachmentRef`，Context Compiler 在每次需要时读取已确认可用的解析表示并组装上下文。

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
  -> ContextAttachmentUse
  -> optional Materialization promotion
```

基本规则：

- 上传先形成原始内容记录，解析成功后才产生可供上下文使用的 representation；
- 只有 `READY` representation 可以被引用和编译；
- 解析失败直接反馈给用户，MVP 不自动重试；用户重新上传时创建新的逻辑附件；
- representation 必须具有固定的 revision 和 content hash；编译与物化不能回查“最新内容”；
- 附件移除或进程结束后，旧 ref 可以明确失效，不得解析到同名或复用 ID 的其他内容；
- Materialization 使用的 representation 需要冻结其 revision/hash，以便后续 provenance 可重现。

## 4. Attachment 与 Chat、Topic、Run

上传后附件不自动绑定到某个 Topic。只有在 Chat 路由已经确定、并且某次 Chat run 首次选择或实际使用该 ref 时，才创建该附件与当前 Topic/Interaction 的关系。因此 Topic 是否在上传时已经存在，不影响附件先被接纳为可引用内容。

当前 run 中新上传的附件可以语义上自动挂载一次，方便用户立即使用；之后用户仍可自由选择是否再次挂载。选择记录与实际编译使用记录应分离：前者表示用户意图，后者表示某一版本 representation 真的进入了上下文。

附件记录只在确实需要时携带稳定的 `interaction_id`；不为没有实际消费者的裸 `run_id` 预留字段。若未来需要记录某个 Agent 执行实例，应使用明确命名的 provenance 字段，而不是改变附件引用的业务语义。

Topic 删除、结算或短期内容清理可以使 Topic 关系消失，但不应改写已经在 Materialization 中冻结的附件内容快照。附件自身的可用性、移除和过期由其运行时生命周期负责。

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

系统使用记录：

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

`AttachmentRef` 只表示用户选择的内容入口，不应携带未经验证的物理路径或任意内容。系统解析 ref 后必须再次确认附件仍存在、representation 属于该附件、状态为 READY 且 revision/hash 一致。

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
- source locator 与内容 hash 的组合方式；
- `DocumentArtifact` promotion 是否适用，以及与 v0.7.0 文档摄取的复用边界。

## 10. 与正式计划的关系

本文只作为 Chat Attachment idea，不是实现承诺。附件功能将在前置公共契约稳定后，另行制定上传、解析、引用、上下文编译和 Artifact promotion 的正式实施计划。
