---
title: kernel 遗留命名清理
status: todo
owner: project
scope: legacy-kernel-naming-cleanup
related_docs:
  - docs/ideas/workspace-network-task-process-architecture.md
  - docs/contracts/mtp.md
  - docs/alice/mtp-runtime.md
  - docs/frontend/chat-workspace.md
  - docs/help/troubleshooting.md
last_reviewed: 2026-10-06
---

# kernel 遗留命名清理

## 问题与证据

“kernel”在大约第三代项目架构时引入，随子系统分层逐渐取消。owner 于 2026-10-06 确认它不再作为概念使用，CPU 内部的 backend 称为 harness 实例（[总 Idea](../ideas/workspace-network-task-process-architecture.md#17-第四部分前提owner-提出) 第四部分前提 5、8）。剩余使用点是当时未改动的命名，按含义分为四组（2026-10-06 核对，不含 `docs/archive/`）：

| 组 | 使用点 | 当前含义 |
|:---|:---|:---|
| MTP 系统工具 | `agent_runtime/mtp/syscalls/` 的 `KernelSyscall`、`build_kernel_registry`；`agent_runtime/mtp/runtime.py` 的 `_kernel_registry` 与注释中的 `KERNEL_REGISTRY`；`i18n/mtp_runtime.py` 的错误键 `mtp.run.kernel_tool_not_found` 及其文案；[MTP 契约](../contracts/mtp.md)第 3.3、4 节的“Kernel Registry”“Kernel syscall”；[MTP Runtime](../alice/mtp-runtime.md)第 5.1 节等；[前端管理视图](../frontend/management-views.md)与 [Meal Assistant 规格](../applications/MealAssistantProductSpec.md)中的“kernel syscall”“内核工具” | `sys_*` 系统工具及其注册表，与由 `CODE_SNIPPET` 记忆形成的用户工具相对 |
| 提示词 | `i18n/prompts.py` 中 RUN 的说明“Execute a kernel tool”、上下文标题 `HIVE MEMORY KERNEL CONTEXT` 与“persistent memory kernel” | 模型可见的措辞 |
| 后端注释与示例 | `core/protocol/models.py`、`core/mtp/models.py`、`agent_runtime/execution/loop.py` 的模块说明；`server/routers/logs.py` 的示例模块名 `kernel.py` | 早期架构中的组件名 |
| 前端观测面板 | `frontend/src/stores/kernel/`（`kernel-store`）与界面中的“Kernel Vision”“Kernel Terminal”；[Chat 工作区](../frontend/chat-workspace.md)、[状态与传输](../frontend/state-and-transports.md)、[应用外壳](../frontend/application-shell.md)、[前端索引](../frontend/README.md)与[排障](../help/troubleshooting.md)第 9 节 | 日志、trace 与 RuntimeEvent 的观测面板 |

测试中有 11 个文件引用上述名称，集中在 `tests/integration/mtp/`、`tests/unit/agent_runtime/` 与 `tests/unit/prompts/`。

## 影响

- 与当前分层的用语冲突：读者容易以为存在一个“kernel”层，而总 Idea 第四部分的分层是 CPU 端口、CPU 驱动、harness 实例与操作适配器。
- 四组的含义各不相同，不能整体替换为同一个词。

## 完成条件

- MTP 系统工具一组改用与 `allowed_sys_tools`、`sys_*` 一致的名称（例如“系统工具 / 用户工具”）；类名、注册表、错误键与文案、MTP 契约、MTP Runtime 与引用它的前端和应用文档同步，错误键的全部消费者与测试一并更新。
- 提示词的改写按模型可见行为的变更评估，不只按重命名处理。
- 后端注释与示例按当前组件名改写。
- 前端观测面板的名称是用户可见的界面命名，先单独决定新名称，再同步代码、前端文档与排障文档。
- 完成后，除 `docs/archive/` 外，代码与文档不再以 kernel 指称现有组件。
