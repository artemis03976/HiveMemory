---
title: Chat 附件链路
status: current
owner: system
scope: chat-attachment-upload-parse-select-compile-bind-and-promotion
code_paths:
  - src/hivememory/server/routers/workspace_assets.py
  - src/hivememory/server/models/workspace_asset.py
  - src/hivememory/server/models/chat.py
  - src/hivememory/server/routers/chat.py
  - src/hivememory/system/application/workspace_asset_service.py
  - src/hivememory/system/services/attachments/upload.py
  - src/hivememory/system/services/attachments/parse_service.py
  - src/hivememory/system/config/attachments.py
  - src/hivememory/system/services/attachments/
  - src/hivememory/system/runtime/workspace/store.py
  - src/hivememory/system/runtime/serial_gate.py
  - src/hivememory/engines/attachment_compiler/
  - src/hivememory/patchouli/service.py
  - src/hivememory/patchouli/control/interaction_submission.py
  - src/hivememory/patchouli/services/memory_generation.py
  - src/hivememory/core/models/workspace_asset.py
  - src/hivememory/core/protocol/models.py
  - frontend/src/services/attachmentApi.ts
  - frontend/src/stores/attachment/
  - frontend/src/components/chat/AttachmentQueue.tsx
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
  - docs/contracts/error-model.md
related_docs:
  - docs/architecture/workspace.md
  - docs/system/application-services.md
  - docs/frontend/chat-workspace.md
  - docs/patchouli/artifacts.md
related_plans:
  - docs/archive/plans/v0.6.2-w1-chat-attachments.md
last_reviewed: 2026-09-11
---

# Chat 附件链路

本文是 Chat 附件从上传、确定性解析、Chat 选择、上下文编译到 Topic binding 与 Artifact promotion 的当前事实入口。它描述一条横跨 System（上传应用服务与 WorkspaceAssetStore）、Patchouli（prepare/finalize 与 binding）和前端（上传队列与选择状态）的完整链路；WorkspaceAssetStore 本身的两级状态机、READY-only 使用、删除与 lease 底层语义以[Workspace 架构](../architecture/workspace.md)为准，跨子系统身份与 interaction 时序以[子系统公共契约](../contracts/subsystem-contracts.md)为准。

链路的核心不变量只有一条：**Interaction 载荷只携带一份附件事实——AttachmentCompiler 确认实际进入上下文的使用引用**。用户选择只作为 prepare/compiler 的短期输入，上传、选择或编译跳过本身不产生任何长期关系。

## 1. 阶段链路总览

```text
上传 POST /api/v1/workspace/assets（multipart，Idempotency-Key）
  -> 上传应用服务持有 operation 串行门，委托 receive_upload 校验/受限读取/SHA-256
  -> Store.register_uploaded_asset：asset + bound ref + RAW READY（原子）
  -> AttachmentParseService 在同一请求内交接确定性解析（AttachmentParserConfig 限制）
       TXT/Markdown 解码 | DOCX 正文提取
  -> complete/fail representation（token guard，同一临界区提交聚合状态）
  -> HTTP 响应即终态：READY 或 FAILED + 安全摘要

Chat 请求 attachments（bound ref + 可选版本摘要）
  -> Patchouli prepare：逐项 acquire READY representation + 版本核对
  -> AttachmentCompiler：确定性 section + used refs + 诊断
  -> AgentRunContext.attachment_compile_result

finalize
  -> InteractionPayload.used_attachments（不可变 transport snapshot）
  -> submission handler 以 asset_refs 形参传入 Perception apply
  -> Perception apply：TopicAssetBinding（按 asset_id 幂等）

Topic settlement
  -> Materialization task 携带 bindings
  -> Memory CREATE/UPDATE 时：按 binding.asset_ref acquire -> DocumentArtifact -> release
```

## 2. 上传与资产注册

上传入口是 `POST /api/v1/workspace/assets`，单文件 `multipart/form-data`，文件字段名 `file`；重复 file part、多文件 part 或额外业务字段按非法请求拒绝。请求头 `Idempotency-Key` 是本次上传的稳定 operation identity，服务层将其映射为 Store 的 `client_operation_id`；同一文件重试必须沿用同一取值。身份仍由 `x-user-id` / `x-workspace-id` 统一承载。

上传应用服务（`WorkspaceAssetApplicationService`）只负责用例顺序：持有 operation 串行门（`system/runtime/serial_gate.py` 的 `KeyedSerialGate`）、接收文件、调用 Store 原子注册、交给解析服务并返回最终回执。文件接收由 `system/services/attachments/upload.py` 的 `receive_upload` 完成，解析接纳由同目录 `parse_service.py` 的 `AttachmentParseService` 承担；两者都不额外引入上传数据模型。

接收阶段在创建资产前完成全部无副作用校验：拒绝空文件；文件名做 NFKC 规范化、去除路径分隔符与控制字符并限制为 200 字符（只作展示用途的 `display_name`）；按批准格式集合分派规范媒体类型（`.txt`/`.md`/`.markdown`/`.docx`，客户端 MIME 只作提示、明确冲突即拒绝）。读取按 64 KiB 分块进行，`size_bytes` 与 SHA-256 均以实际读取字节为准，超出 `max_raw_bytes` 立即中止。上传流与框架临时文件由 router 在 transport 边界关闭；接收函数不访问 Store，也不持有资产生命周期。

通过校验后，服务调用 Store 的 `register_uploaded_asset` 命令：在 Store 同一临界区内按 `(workspace_identity, client_operation_id)` 幂等定位，fingerprint 由 metadata 与原始内容哈希共同组成——相同 operation 重放返回同一资产（HTTP 200），携带另一份文件或不同 metadata 报告操作冲突（HTTP 409）；新建时一次完成资产创建、bound ref 签发与 RAW representation（revision 1、READY）注册。文档资产的 `required representation` 是 `EXTRACTED_TEXT`，因此 RAW 注册后聚合保持 `PROCESSING`。

响应只包含安全摘要：bound ref、`asset_id`、规范化 `display_name`、媒体类型、大小、资产状态，以及 RAW 与 required representation 的版本摘要（representation_id、revision、state、content_hash、producer/version）。可用性只通过 `state` 与 required representation 摘要表达，不设可漂移的 `is_ready` 标志；响应不包含原始字节、物理路径或内部对象。

## 3. 确定性解析

解析在同一次上传请求内完成（同步、受控），不建队列、任务 ID 或状态查询路由。解析器只消费 Store 命令快照冻结的 RAW 字节，不依赖 HTTP UploadFile、临时路径、Store 对象、Agent 或 LLM。

`AttachmentParseService` 负责 register/start/parse/complete-or-fail 交接，只接收 `IdentityScope` 和注册后的 `WorkspaceAssetHandle`，返回最终 `WorkspaceAsset` 快照。媒体类型和 RAW 均来自 Store 快照；RAW 的字节类型、实际大小和实际 SHA-256 在解析前校验，不另传接收阶段的元数据副本。解析算法仍由各 parser 执行，幂等、revision、token 和状态迁移仍由 Store 决定。assembler 为上传应用服务和解析服务注入同一 Store 与 `AttachmentParserConfig`。

### 3.1 格式与限制

批准格式为 `.txt`（`text/plain`）、`.md`/`.markdown`（`text/markdown`）与 `.docx`（WordprocessingML 包）；三者统一登记为 `kind="document"`，required representation 均为 `EXTRACTED_TEXT`。扩展名大小写不敏感；旧版 `.doc` 提示另存为 `.docx`，PDF/RTF/宏文档等不进入本轮，解析失败也不回退为 TXT。

资源限制来自 `AttachmentParserConfig`：RAW 输入 10 MiB（与上传接收共用同一值）、提取正文 8 MiB、完整 canonical 内容 16 MiB、locator 数量 5 万、DOCX 包成员数/解压大小/压缩比、XML 深度与节点数，以及可注入单调时钟的协作式解析预算（默认 5 秒）。超限整体失败，不静默截断。

### 3.2 解码与提取

TXT 与 Markdown 共用固定顺序解码：UTF-8 BOM → UTF-16 LE/BE BOM → 无 BOM 时仅严格 UTF-8；UTF-32 显式拒绝，不尝试 GBK/GB18030 或概率探测。CRLF/CR 统一为 LF，保留其他空白、缩进、空行与末尾换行；除 TAB/LF/FF 外的控制字符与 C1 控制符拒绝。Markdown 保留原始语法，不构建 AST、不渲染、不执行。

DOCX 使用标准库 `zipfile` 与 `defusedxml` 流式解析：包校验（成员数、路径寻址、加密标志、声明解压大小与压缩比、重复成员）后定位主文档 part 并验证为非宏 Word 文档；正文按 XML 文档顺序提取，段落/run 间空白保真，表格降级为文字行（行间 LF、单元格间 TAB），超链接只保留显示文字，字段只取已保存的显示结果。未接受修订、嵌套表格、`altChunk`、正文内容控件与 `AlternateContent` 整体归入受控失败，不静默跳过；页眉页脚、脚注、批注与图片输出稳定 warning record。

### 3.3 内容对象、定位与哈希

成功产物是可递归冻结的 JSON 风格 `content_object`：`schema_version`、`format`（`plain_text`/`markdown`）、`text`、`source_raw`（唯一 RAW 的 revision 与 content hash）、`locators`（正文码点区间与源位置映射，TXT/Markdown 按非空行、DOCX 按正文段落与表格行，序号计入空元素）和 `warnings`（去重且排序稳定）。正文与 locator 校验、大小累计和 canonical 哈希只在共享结果模型中实现一次；哈希覆盖正文、format、locator、warning 与 RAW 来源版本，键排序、紧凑分隔符、非 ASCII 直接编码。

可预期失败只返回内部类别（`content_unreadable` / `resource_limit` / `execution_failure`）与安全文案；解析失败统一以公共终态 `workspace.asset.failed` 提交，不生成半份成功内容，也不新增 `workspace.asset.parse.*` 公共错误码。

### 3.4 终态提交与响应

解析服务在 Store 锁外经标准线程转交执行 parser。成功结果经来源核对（RAW revision/hash 与 producer/version 匹配）后以原 parse token 调用 `complete_representation`；预期失败调用 `fail_representation` 并附 `AssetSafeError(code="workspace.asset.failed")`。required representation 与资产聚合状态在同一临界区原子进入 READY/FAILED，上传响应因此只会是终态快照或稳定错误。RAW 注册成功后的解析失败仍以 201/200 返回 `state=failed` 与安全摘要，不改写为上传失败；失败保留 RAW，但普通 reader 拒绝 FAILED 资产。应用层用最终快照重建回执并保留 Store 原始 `created` 标记，解析服务不参与 HTTP 201/200 判定。

同一 `(workspace_identity, client_operation_id)` 的并发上传由应用服务持有的独立 `KeyedSerialGate` 实例在单 event loop 内串行化，持门范围覆盖接收、注册与解析收尾全过程。等待方在持有方到达终态或错误收尾后继续，随后命中既有重放或冲突路径。公共门的取消与回收机制见[System 运行时](./runtime-and-bus.md#5-keyedserialgate)；上传服务不保存另一份幂等结果或资产状态。重复请求不重复解析；解析失败后的重新上传使用新的 operation，形成新资产。

complete/fail 被 Store 以 stale、removed 或 closed 拒绝时直接传播既有错误，HTTP 分别映射 409、410、503；不查询列表后回退到旧上传快照。请求取消时，解析服务以原 token 尽力提交安全失败，Store 拒绝只记录日志，继续传播 `CancelledError`。这些行为保证晚到结果不能覆盖已有终态，也不会因收尾回退而返回 PROCESSING。

## 4. Chat 选择与 lease

前端上传队列与 Chat 回合选择是两种状态：队列项保存文件与服务端回执，选择集合只保存 `assetState=ready`（即 required representation READY）的项。用户可逐项选择与取消；取消只移除本地选择，不调用 Store、不创建 binding、不删除资产。opaque ref 与选择集合只存在于当前运行时内存，不写入 localStorage；页面刷新后需重新上传。

`POST /api/v1/chat` 的 `attachments` 字段是发送时冻结的有序选择数组，每项携带 bound ref 与可选的预期版本摘要（representation_id/revision/content_hash）；同一 ref 只能出现一次，空数组与缺省等价。HTTP 层只校验字段类型与数组结构；ref 归属、READY 状态与版本核对发生在 Patchouli prepare 边界。

prepare 按用户顺序逐项调用 reader 的 `acquire_ready_representation()`——该端口已在 Store 同一临界区完成归属、asset READY 与 representation READY 校验并建立 lease，不做前置 `resolve_asset`。返回 lease 的 representation ID/revision/hash 与请求摘要核对，任一失败释放已取得的 lease 并拒绝整个 run；请求字段结构错误在 HTTP body 校验拒绝（422），ref/READY/版本失败在 prepare 拒绝（Alice 未启动、Interaction 未提交），沿 Chat/Workspace 错误边界以安全文案返回。remove 早于 acquire 按既有 Store 语义拒绝本轮；acquire 早于 remove 时已有 lease 保存冻结内容，本轮继续可用。

lease 生命周期覆盖 prepare acquire、附件编译到 Interaction finalize：prepare 失败立即逐项释放；prepare 成功未进入 finalize 时由 `cleanup_prepared_agent_run` 释放；进入 finalize 后由 continuation 在 Interaction 与后置工作完成后释放。释放容忍 Store 已关闭并记录摘要；重复释放沿 Store 幂等语义处理。

## 5. AttachmentCompiler

`AttachmentCompiler` 是独立于 `MemoryCompiler` 的附件上下文编译组件：不接收 `MemoryAtom`、不使用 Memory target 枚举、不产生 Memory artifact、不做资产状态迁移。它在 prepare 阶段被调用，输入是校验后的 lease 集合与 `AttachmentCompilerConfig` 预算，输出不可变的 `AttachmentCompileResult`：

- `attachment_context`：确定性附件 section（边界标记声明正文精确长度，正文逐字保留，不构成系统指令）；
- `used_attachments`：实际进入 section 的 bound ref 集合（按用户顺序）；
- `diagnostics`：截断、跳过与预算摘要等结构化诊断。

预算规则固定为：多附件严格按用户顺序；单附件超预算时在 locator 边界保留前部完整内容并声明 truncated；合计超总预算时跳过剩余附件；首个完整单元无法保留且此前无保留内容时整体编译失败；全部选中项被跳过同样整体失败，不生成空 section。正文中的指令样式文本只保留字面内容。

用户选择只存在于 prepare 的短期输入（有序 lease）；compiler 的输入就是这组 lease，输出 `used_attachments`（bound ref 集合）是 binding 投影的唯一输入，`AgentRunContext` 与 `InteractionPayload` 中不存在独立的选择字段。

## 6. Interaction binding

finalize 从 `AttachmentCompileResult` 生成一份有序的实际使用引用快照，写入 `InteractionPayload.used_attachments`。进入 submission 后它是该输出的不可变 transport snapshot：retry 重放同一份引用，handler 把这份快照一次性作为 `asset_refs` 形参传给 `apply_interaction`，不回查原始选择、asset 列表或当前 UI 状态。

Perception 在 Interaction 成功 apply 的同一 Topic 快照更新中，把去重后的 `(asset_id, asset_ref)` 写入 `TopicAssetBinding`（按 `asset_id` 幂等，保留首次绑定时间语义）。编译跳过、预算未保留、admission/apply 失败与取消均不建立 binding；上传和解析本身同样不产生 binding。快照中的引用坐标可参与 interaction digest 以防 retry 替换 ref，但正文与 lease 不进入长期载荷。

## 7. Settlement 与 Artifact promotion

Topic settle 在删除 Topic 前把 `TopicData.bindings` 冻结进 `TopicMaterializeTask`，经既有 queue codec 传入 `InteractionArtifactInput.asset_bindings`；不新增第二份使用明细，也不把正文、representation snapshot、lease_id 或文件字节写入 Topic、task、queue 或 Memory 数据。

Memory generation 只有在确实产生 CREATE/UPDATE 时才对 task 中的 bindings 做 promotion（TOUCH/DISCARD 不提升）：按 `binding.asset_ref` 在当前 Workspace scope 下 acquire READY representation、构建独立 `DocumentArtifact`、释放 lease。提升产物的 `source_uri` 钉住源 asset/representation 标识、revision 与 parser producer/version，`content_hash` 单列，使产物自身锁定来源版本；这是创建独立证据快照，不是把 WorkspaceAsset 原地转换。

ref 已 remove、Store 已关闭或写入失败时跳过该 binding 的 promotion 并记录结构化 warning；已提交的 binding 保持不变，不回滚 Interaction 或 Memory 结果。promotion retry 复用现有 generation operation identity 与同一 binding payload。

## 8. 失败语义与限制汇总

- 上传前可确定的错误：缺少/非法文件元数据 400、大小超限 413、不支持的媒体类型 415、operation 输入冲突 409、原资产已移除 410、Store 不可用 503；
- Chat 选择错误：字段结构 422；ref/READY/版本失败在 prepare 边界以 `workspace.asset.not_found` / `not_ready` / `failed` / `removed` / `operation_conflict(store_closed)` 语义沿 SSE `error` 事件返回安全文案与 code；
- 解析失败：资产终态 `workspace.asset.failed` + 安全文案，RAW 保留；
- ref 失效或 Store 关闭：已提交 binding 不变，promotion 跳过并记录结构化 warning。

## 9. 代码与测试入口

后端：

- 上传路由与应用服务：[`server/routers/workspace_assets.py`](../../src/hivememory/server/routers/workspace_assets.py)、[`system/application/workspace_asset_service.py`](../../src/hivememory/system/application/workspace_asset_service.py)、[`server/models/workspace_asset.py`](../../src/hivememory/server/models/workspace_asset.py)；
- 接收、解析交接与公共串行门：[`upload.py`](../../src/hivememory/system/services/attachments/upload.py)、[`parse_service.py`](../../src/hivememory/system/services/attachments/parse_service.py)、[`runtime/serial_gate.py`](../../src/hivememory/system/runtime/serial_gate.py)；确定性 parser、结果模型与受控错误同属 [`system/services/attachments/`](../../src/hivememory/system/services/attachments/)；
- Chat 选择与编译交接：[`patchouli/service.py`](../../src/hivememory/patchouli/service.py)、[`engines/attachment_compiler/`](../../src/hivememory/engines/attachment_compiler/)；
- binding 投影与 promotion：[`patchouli/control/interaction_submission.py`](../../src/hivememory/patchouli/control/interaction_submission.py)、[`patchouli/services/memory_generation.py`](../../src/hivememory/patchouli/services/memory_generation.py)；
- 配置：[`system/config/attachments.py`](../../src/hivememory/system/config/attachments.py)（`AttachmentParserConfig` / `AttachmentCompilerConfig`）。

前端：

- 上传队列与选择状态：[`stores/attachment/attachmentStore.ts`](../../frontend/src/stores/attachment/attachmentStore.ts)、[`services/attachmentApi.ts`](../../frontend/src/services/attachmentApi.ts)、[`components/chat/AttachmentQueue.tsx`](../../frontend/src/components/chat/AttachmentQueue.tsx)、[`hooks/useAttachmentUpload.ts`](../../frontend/src/hooks/useAttachmentUpload.ts)。

代表性行为测试：

- Store 原子注册与幂等：[`tests/unit/system/runtime/workspace/test_store.py`](../../tests/unit/system/runtime/workspace/test_store.py)；
- 上传服务与请求内解析：[`tests/integration/system/application/test_workspace_asset_service.py`](../../tests/integration/system/application/test_workspace_asset_service.py)、[`test_workspace_asset_parsing.py`](../../tests/integration/system/application/test_workspace_asset_parsing.py)；
- 公开入口：[`tests/integration/system/test_workspace_asset_upload_api.py`](../../tests/integration/system/test_workspace_asset_upload_api.py)、[`test_workspace_asset_chat_selection.py`](../../tests/integration/system/test_workspace_asset_chat_selection.py)、[`test_workspace_asset_parse_acceptance.py`](../../tests/integration/system/test_workspace_asset_parse_acceptance.py)；
- 解析器与编译器：[`tests/unit/system/services/attachments/`](../../tests/unit/system/services/attachments/)、[`tests/integration/system/services/attachments/`](../../tests/integration/system/services/attachments/)、[`tests/unit/engines/attachment_compiler/`](../../tests/unit/engines/attachment_compiler/)；
- codec 与 binding/promotion：[`tests/unit/patchouli/control/test_interaction_submission_v2.py`](../../tests/unit/patchouli/control/test_interaction_submission_v2.py)、[`tests/unit/patchouli/test_prepare_attachments.py`](../../tests/unit/patchouli/test_prepare_attachments.py)、[`tests/unit/patchouli/services/test_memory_generation_promotion.py`](../../tests/unit/patchouli/services/test_memory_generation_promotion.py)。

## 10. 当前边界与限制

- WorkspaceAsset、ref 与 lease 只在当前进程 Store 存活期内有效；服务重启或页面刷新后需重新上传；
- 解析为同步受控执行，只适用于 10 MiB 以内的受控输入；异步队列、硬超时和进程隔离在出现真实负载后另行设计；
- DOCX 不还原版式、自动编号与页码；表格只保证行级文字顺序；PDF/RTF/宏文档与图片类附件不在本轮；
- promotion 只在 Memory CREATE/UPDATE 时发生且 best-effort；ref 失效或 Store 关闭时跳过并记录 warning，不回滚已提交结果；
- Agent 主动解析、MTP RUN 执行附件和强沙箱不属于本链路，按独立执行能力计划推进。
