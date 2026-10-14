---
title: kernel 遗留命名清理
status: archived
archived_at: 2026-10-12
implemented_by: refactor/kernel-legacy-naming（本归档提交）
superseded_by: docs/alice/mtp-runtime.md
owner: project
scope: legacy-kernel-naming-cleanup
related_docs:
  - docs/ideas/fifth-generation-architecture.md
  - docs/contracts/mtp.md
  - docs/alice/mtp-runtime.md
  - docs/frontend/chat-view.md
  - docs/frontend/management-views.md
  - docs/help/troubleshooting.md
  - docs/ideas/execution-unit-thread-and-environment.md
last_reviewed: 2026-10-12
---

# kernel 遗留命名清理

> **文档状态**：Archived（2026-10-12）。已在分支 `refactor/kernel-legacy-naming` 实施并验收；当前事实见 [MTP 契约](../../contracts/mtp.md)第 3.3 节、[MTP Runtime](../../alice/mtp-runtime.md)第 4、5 节、[Chat 视图](../../frontend/chat-view.md)与[状态与传输](../../frontend/state-and-transports.md)。下文“问题与证据”记录的是实施前的状态。

## 问题与证据

“kernel”在大约第三代项目架构时引入，随子系统分层逐渐取消。owner 于 2026-10-06 确认它不再作为概念使用，CPU 内部的 backend 称为 harness 实例（[执行单元 Idea](../../ideas/execution-unit-thread-and-environment.md#1-前提owner-提出)前提 5、8）。剩余使用点是当时未改动的命名，与它配对的“用户态工具”和同期的“HiveOS”一并清理。按含义分为四组（2026-10-12 重新核对，不含 `docs/archive/` 与冻结的 `docs/governance/baselines/`）：

| 组 | 使用点 | 当前含义 |
|:---|:---|:---|
| MTP 系统工具 | 代码：`agent_runtime/mtp/syscalls/` 的 `KernelSyscall`、`build_kernel_registry` 与“内核级工具”注释；`agent_runtime/mtp/runtime.py` 的 `_kernel_registry` 与 RUN 分发注释（`KERNEL_REGISTRY`、“内核工具快速路径”“用户态工具路径”）；`i18n/mtp_runtime.py` 的错误键 `mtp.run.kernel_tool_not_found` 及中英文案。文档：[MTP 契约](../../contracts/mtp.md)第 3.3 节的“Kernel Registry”；[MTP Runtime](../../alice/mtp-runtime.md)第 4、5、8、11 节的“Kernel syscall”“Kernel Registry”“内核注册表”与第 5.2 节标题“用户态工具”；[子系统契约](../../contracts/subsystem-contracts.md)第 4.6 节、[Workspace 架构](../../architecture/workspace.md)第 4.5 节、[System 配置](../../system/configuration.md)第 1.1 节的“用户态工具”；根 `README.md` 与 `README_EN.md` 的 RUN 说明；[前端管理视图](../../frontend/management-views.md)第 2.1 节的“kernel syscall” | `sys_*` 系统工具及其注册表，与由 `CODE_SNIPPET` 记忆形成的用户工具相对 |
| 提示词 | `i18n/prompts.py` 中 RUN 的说明“执行内核工具 / Execute a kernel tool”、MTP 协议规格的标题 `HIVE MEMORY KERNEL CONTEXT`（中文“HIVE MEMORY 内核上下文”）与身份句“运行在 HiveOS 上……持久化记忆内核”，以及 State Manager 提示词中的“AI 操作系统 (HiveOS)” | 模型可见的措辞 |
| 后端注释与示例 | `agent_runtime/execution/loop.py` 的“Kernel Loop Executor”；`core/protocol/models.py` 的“Eye 与 Kernel”（整段说明已过时）；`core/mtp/models.py` 的“Kernel 执行链路”“内核执行结果”；`prompts/mtp.py` 的“内核上下文”“内核工具列表”；`agent_runtime/mtp/runtime.py` 的“供 Kernel 注入”“进入内核态执行”；`server/routers/logs.py` 的示例模块名 `kernel.py` | 早期架构中的组件名 |
| 前端观测面板 | `frontend/src/stores/kernel/`（持久化 key `kernel-store`、BroadcastChannel `hivememory_kernel_logs`）、`types/kernel.ts`、`components/chat/KernelVision.tsx` 与 `KernelTerminalTab.tsx`、`stores/ui/appUiStore.ts` 的面板状态（随 `chat-ui-store` 持久化）、界面标题“内核视界”；`types/config.ts` 与 `constants/settings.ts` 的 `include_kernel_tools`；[Chat 工作区](../../frontend/chat-view.md)、[状态与传输](../../frontend/state-and-transports.md)、[应用外壳](../../frontend/application-shell.md)、[前端索引](../../frontend/README.md)、[排障](../../help/troubleshooting.md)第 9 节与[任务进程 Idea](../../ideas/task-process-table-and-registration-entry.md)中的“Kernel Vision”“Kernel Terminal”“内核终端” | 日志、trace 与 RuntimeEvent 的观测面板 |

面板改称 Workspace Vision 后，前端的 Workspace 一词留给后端 Workspace 概念的映射，原先以“工作区”指称 Chat 页面与中间栏的命名随前端一组一并更换：`components/chat/ChatWorkspace.tsx`（及其注释、`ChatLayout.tsx` 的导入、`hooks/useAttachmentUpload.ts` 的注释）；[Chat 工作区](../../frontend/chat-view.md)的文件名、标题与正文，[应用外壳](../../frontend/application-shell.md)、[前端索引](../../frontend/README.md)与[状态与传输](../../frontend/state-and-transports.md)中的“Chat 工作区”“Chat Workspace”“工作区偏好”；引用 `docs/frontend/chat-workspace.md` 路径的 [文档治理](../../DOCUMENTATION.md)目录树、[附件链路](../../system/attachments.md)元数据与 `docs/archive/` 中 7 份文档的链接。

测试中有 12 个文件引用上述名称：`tests/e2e/pipeline/test_agent_loop_e2e.py` 的模块说明，`tests/integration/mtp/` 的 `conftest.py`、`test_mtp_integration.py`、`test_run_chain.py`、`test_agent_permission_chain.py`、`test_process_operations.py`，`tests/unit/agent_runtime/` 的 `execution/test_loop_turn_events.py`、`mtp/syscalls/conftest.py`、`mtp/syscalls/test_registry.py`，`tests/unit/prompts/` 的 `test_mtp.py`、`test_mtp_permissions.py`（断言标题 `HIVE MEMORY KERNEL CONTEXT`）、`test_message_assembler.py`（断言 HiveOS 身份句）。

## 影响

- 与当前分层的用语冲突：读者容易以为存在一个“kernel”层，而执行单元 Idea的分层是 CPU 端口、CPU 驱动、harness 实例与操作适配器。
- 四组的含义各不相同，不能整体替换为同一个词。

## 已决定的命名（owner，2026-10-12）

MTP 系统工具一组保留 syscall 相关命名（`syscalls/` 包、`SyscallResult`、`syscall.*` 错误键前缀、`sys_` 前缀），只去掉 Kernel：

| 现名 | 新名 |
|:---|:---|
| `KernelSyscall` | `Syscall` |
| `build_kernel_registry` | `build_syscall_registry` |
| `KoakumaRuntime._kernel_registry` | `_syscall_registry` |
| 错误键 `mtp.run.kernel_tool_not_found`，文案“内核工具 / Kernel tool” | `mtp.run.tool_not_found`，文案“工具 / Tool” |
| 文档与注释中的“Kernel syscall”“Kernel Registry”“内核工具” | “syscall”“syscall 注册表”“系统工具” |
| “用户态工具” | “用户工具” |

提示词：RUN 的说明改为“执行工具 / Execute a tool”；移除全部 HiveOS 命名；协议规格标题与“持久化记忆内核”去掉 kernel，具体措辞随实施拟定。

前端观测面板改称 Workspace Vision，其余名称同理：

| 现名 | 新名 |
|:---|:---|
| `isKernelVisionCollapsed`、`setKernelVisionCollapsed`、`toggleKernelVision` | `isWorkspaceVisionCollapsed`、`setWorkspaceVisionCollapsed`、`toggleWorkspaceVision` |
| `kernelVisionTab`、`setKernelVisionTab`、`KernelVisionTab` | `workspaceVisionTab`、`setWorkspaceVisionTab`、`WorkspaceVisionTab` |
| `KernelVision`、`KernelTerminalTab`，界面“内核视界” | `WorkspaceVision`、`WorkspaceTerminalTab`，界面“工作区视界” |
| `stores/kernel/`、`useKernelStore`、`KernelStore`、`KernelSet` / `KernelGet`、日志前缀 `[KernelStore]` | `stores/workspace/`、`useWorkspaceStore`、`WorkspaceStore`、`WorkspaceSet` / `WorkspaceGet`、`[WorkspaceStore]` |
| `types/kernel.ts`、`KernelConnectionStatus` | `types/workspace.ts`、`WorkspaceConnectionStatus` |
| 持久化 key `kernel-store`、BroadcastChannel `hivememory_kernel_logs` | `workspace-store`、`hivememory_workspace_logs` |
| 文档中的“Kernel Vision”“Kernel Terminal”“Kernel store”“内核终端” | “Workspace Vision”“Workspace Terminal”“Workspace store”“工作区终端” |

只指日志流而不指面板的说法（如“Kernel 日志 WebSocket”“Kernel stream URL”）直接去掉 Kernel。

Chat 页面与中间栏让出 Workspace 一词（2026-10-12）：

| 现名 | 新名 |
|:---|:---|
| 中间栏组件 `ChatWorkspace`、`ChatWorkspaceProps`，文档中的“Chat Workspace” | `ChatPane`、`ChatPaneProps`，“Chat Pane / 对话区” |
| 整个 Chat 页面“Chat 工作区”，文档 `docs/frontend/chat-workspace.md` | “Chat View / Chat 视图”，文档改为 `docs/frontend/chat-view.md` |

页面组件 `ChatLayout` 不含 Workspace，保持不变。

## 实施注意

- 前端管理视图第 2.1 节“编辑 MTP verb 和五个 kernel syscall 白名单”与第 2.2 节前两条，自 #114 移除 Agent 图纸权限编辑后已不成立，应删除，不做改名。
- 前端 `include_kernel_tools` 是死字段：后端 `MTPPromptConfig` 自 #26 起已无此字段，界面也不读取它，应删除，不做改名。
- `chat-ui-store` 持久化了面板状态，字段改名需要升到 version 3 并迁移旧字段；`kernel-store` 改 key 后旧 filters 不会自动带入，迁移旧 key 或接受一次性重置，在 PR 中说明。
- 改写 `tool_not_found` 文案时一并评估其 Suggestion：“用 SEARCH 查找可用工具”与系统工具来自提示词菜单不符，且 Agent 可能没有 `memory.search`。

## 完成条件

- 四组按上述命名完成；错误键的全部消费者与测试一并更新，测试辅助与测试类名（如 `simulate_kernel_loop_single`、`TestKernelRegistry`、`TestRunKernelFastPath`）同步改名。
- 提示词的改写按模型可见行为的变更评估，不只按重命名处理。
- 后端注释与示例按当前组件名改写；已过时的整段说明（如 `core/protocol/models.py`）按现状重写。
- 完成后，除 `docs/archive/` 与 `docs/governance/baselines/` 外，代码与文档不再以 kernel、内核、用户态或 HiveOS 指称现有组件，前端不再以 Workspace 或“工作区”指称 Chat 页面与中间栏；`chat-view.md` 的全部入链（含归档文档，只修链接）已更新。保留项：[AE2 类比 Idea](../../ideas/ae2-hivememory-architecture-analogy.md) 中 AIOS 角色列的“Kernel Composition”，以及执行单元 Idea 记录本决定的条目。

## 实施记录（2026-10-12）

- 系统工具：按命名表实施；`mtp.run.tool_not_found` 的 Suggestion 改为指向提示词中的 `[RUNTIME TOOLS]` 列表，不再建议用 SEARCH 查找（`sys_*` 工具不是记忆）。RUN 分发注释去掉 Level 0/1 与旧规格章节号，用户工具路径按 `ResolveRunTargetRequest` 链路描述。
- 提示词：协议规格标题改为“HIVE MEMORY 上下文 / HIVE MEMORY CONTEXT”，身份句改为“运行在 HiveMemory 上……访问持久化记忆”，State Manager 提示词改为“HiveMemory 的状态管理器”。
- 后端注释：按完成条件改写；`core/protocol/models.py` 的模块说明按现有模型重写，`MessageType` 注释中的 Eye、RetrievalFamiliar、LibrarianCore 等更早的组件名不属于本 Todo，未改动。
- 测试：辅助与测试类名同步改名；syscalls 单元 conftest 中未被引用、签名已过时的循环模拟辅助函数直接删除；`test_loop_turn_events` 中从未被读取的 kernel mock 改为真实 `AgentRuntimeConfig`。
- 前端：按命名表实施；终端内 logs/events 切换的局部类型命名为 `TerminalView`；只指日志流的界面文案（“Connecting to kernel...”等）改为 log stream。`chat-ui-store` 升至 version 3 迁移旧面板字段，持久化 key 首次加载时从 `kernel-store` 迁到 `workspace-store`；迁移代码中保留旧字段名与旧 key 作为迁移来源。`include_kernel_tools` 已删除。
- 文档：MTP 契约第 3.3 节补充未注册 `sys_*` alias 直接返回 `mtp.alias.not_found`；MTP Runtime 第 4 节 RUN 一行的交接边界与第 8 节的 kill switch 说明改为访问登记白名单语义（原文仍写 Profile 白名单，是 #114 的遗留）；前端管理视图第 2 节删除权限编辑的过时说明；`chat-workspace.md` 改为 `chat-view.md`，入链与 7 份归档文档的链接同步。
- 验证：`pytest tests/unit tests/integration -m "not live_llm and not e2e and not slow" -n auto --cov=hivememory --cov-fail-under=85` 3038 通过、2 跳过，覆盖率 93.18%；`uv run --no-sync mypy --warn-unused-ignores src scripts/typecheck_bus_routes.py` 无问题；`ruff check .`、`black --check .` 通过；前端 `npm run lint`、`npm run build` 通过。e2e 与 live_llm 未运行；前端持久化迁移未在浏览器中实测。
