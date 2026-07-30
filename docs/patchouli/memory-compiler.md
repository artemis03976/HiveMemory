---
title: MemoryCompiler
status: current
owner: patchouli
scope: memory-ir-and-task-specific-compilation
code_paths:
  - src/hivememory/engines/memory_compiler/
  - src/hivememory/system/config/memory_compiler.py
related_contracts:
  - docs/contracts/mtp.md
  - docs/system/i18n.md
  - docs/contracts/subsystem-contracts.md
last_reviewed: 2026-07-29
---

# MemoryCompiler

MemoryCompiler 把持久化记忆编译成一次任务能够使用的工作视图。它存在的根本原因是：同一条 MemoryAtom 在向量化、检索注入、MTP READ、Agent Profile 菜单和子 Agent 共享上下文中，不应复制成五套逐渐漂移的字符串模板；但它也不能变成决定检索、权限、运行时策略和工具执行的另一个 God Object。

因此，Compiler 只做两件事：把不同记忆类源归一为中间表示，再按明确 target 生成文本与元数据。谁选择源、谁拥有权限、何时运行 Agent，仍由调用方负责。

单条 `MemoryAtom` 的表达，与一组记忆在运行时语境中的包装，并不是同一责任。`PROMPT_FULL`、`PROMPT_INDEX`、`MTP_READ` 等 unit target 只负责单项语义；retrieval context、READ response 与 shared context 的 envelope 则负责 section、header/footer、空结果提示和行为说明。把两者重新塞回单一 renderer 或 God Object，会同时污染检索预算、权限策略和 Agent-facing 文案，也会让同一记忆无法按不同调用场景稳定复用。

## 1. 编译管线

```text
MemoryAtom | PendingAtom | ResolveResult | PendingAtomSettlement
  -> source builder
  -> MemoryUnitIR
  -> unit target handler
     or section/bundle IR -> envelope strategy
  -> CompiledMemory(text + metadata + sections)
```

公开 `compile(source, target, options)` 拒绝 IR、CompiledMemory 等内部中间结果作为 source，防止调用方绕过 builder 不变量。列表源会被逐项转换；语言未指定时使用系统默认 language。

## 2. Memory IR

`MemoryUnitIR` 分为三组信息：

- `MemoryIdentityIR`：source kind、alias、redirected-from、memory id；
- `MemoryContentIR`：title、summary、content、instruction、tags、memory type；
- `MemoryStatusIR`：pending 状态、source verb、terminal/redirect/discard、message/reason/error。

当前 builders 支持：

- MemoryAtom：正式记忆当前 head；
- PendingAtom：仍在物化中的 WRITE/UPDATE 候选；
- ResolveResult：atom、pending、redirect、discard、failed/expired 等解析结果；
- PendingAtomSettlement：先投影为 redirect/discard ResolveResult。

IR 的意义是让 target handler 面对稳定语义，而不是知道 MemoryAtom、PendingAtom 和 resolver 的全部字段结构。IR 不对外承诺为公共协议，也不应被持久化成另一份记忆真相。

## 3. Unit targets

| Target | 当前用途 |
|:---|:---|
| `PROMPT_FULL` | title/summary/content 等完整 prompt 记忆块 |
| `PROMPT_INDEX` | 紧凑标题、摘要、alias 索引视图 |
| `MTP_READ` | 正式、pending、redirect、discard、failed 等 READ 文本 |
| `SHARED_CONTEXT` | 子 Agent 共享的单元视图 |
| `DENSE_EMBEDDING` | Qdrant dense vector 与 cross-encoder 输入 |
| `SPARSE_EMBEDDING` | Qdrant sparse vector 输入 |
| `AGENT_PROFILE_MENU` | Agent Profile 候选菜单 |
| `RUNNABLE_TOOL` | 仅保留枚举，当前显式抛出 Phase 3 reserved 错误 |

Unit 编译单项返回 CompiledMemory，列表返回同长度列表。CompiledMemory 除 text 外还保留 source kind、alias、memory id、status 与 metadata，使运行时不必从文本反向猜测 redirect 或 pending 状态。

## 4. Envelope targets

Envelope 把多个 unit 分 section 组织成完整上下文：

- `RETRIEVAL_CONTEXT`：区分普通 memories 与 agent profiles，添加本地化 header/footer；
- `MTP_READ_RESPONSE`：组合 READ 结果标题与多个编译单元；
- `SHARED_CONTEXT_INJECTION`：为子 Agent 生成带说明的共享上下文。

Retrieval context 中，Agent Profile 使用 `AGENT_PROFILE_MENU`，普通记忆按 retrieval strategy 选择 full/index。其他默认 section 当前使用 `SHARED_CONTEXT` 编译。

## 5. Retrieval context 策略

当前三种策略由 `MemoryCompilerConfig.retrieval_context.strategy` 选择：

### Full

按顺序把每条记忆编译为 `PROMPT_FULL`，超过预算停止。它适合候选少、需要正文细节的用例。

### Cascade

前 `full_payload_count` 条优先尝试 full；剩余或放不下 full 的单元降为 index，直到 token budget 用尽。它保留头部相关结果的细节，同时让尾部候选仍有 alias/摘要可供 Agent 主动 READ。

### Compact

所有结果都使用 `PROMPT_INDEX`，在预算内尽可能容纳更多记忆。当前默认策略为 Compact，反映了“预检索只提供导航，详细内容可主动读取”的 memory-as-a-tool 取向。

Compiler 维持输入顺序，不重新排序。top-k、分数和权限必须在 Retrieval 侧先完成。

## 6. 当前调用点

- Patchouli prepare：把 Gateway 计划召回的 atoms 编译为 AgentRunContext.memory_context；
- System Passive Ingress：构造被动分析所需记忆上下文；
- Koakuma MTP：编译 SEARCH envelope 与 READ 的 atom/pending/redirect/terminal 结果；
- Alice orchestration：编译子 Agent shared context injection；
- Qdrant storage：为 MemoryAtom 生成 dense/sparse embedding 文本；
- Retrieval reranker：为 cross-encoder 生成候选文本。

这些调用点共享 compiler，却各自拥有 target 和用例时序。MemoryCompiler 不持有 bus、store、Agent frame 或 RetrievalEngine。

## 7. i18n 与格式

Handler 和 envelope 的标签、空值、pending/redirect/error 文本通过 i18n memory compiler catalog 解析。Language 由单次 options 或系统默认值确定，避免 MemoryAtom 本身携带展示语言。

`MemoryCompileOptions.format` 可以进入 envelope metadata，但当前主要终态仍由各 handler 的文本模板决定，并非所有 target 都已经支持 xml/markdown/plain 三种完全不同的 renderer。格式字段不能被当作完整多格式能力承诺。

## 8. 当前限制与设计张力

- `RUNNABLE_TOOL` 仍是 reserved target，没有可执行工具编译；
- Full strategy 的 `max_tokens` 当前用 `len(text)` 比较字符数，而 Cascade/Compact 使用 token estimator，预算口径不一致；
- IR 中部分 list/dict 字段使用直接默认值，依赖 Pydantic 复制语义，后续应统一 default factory；
- Retrieval score/rank 尚未稳定注入 `MemoryUnitIR.metadata`，envelope 主要依赖调用方排序；
- `format` 只部分进入 metadata，尚无全 target 的多格式保证；
- citations 字段已经存在于 CompiledMemory，但当前 handlers 没有形成完整引用链；
- `MTP_READ_RESPONSE` envelope 不是所有 READ 路径的唯一出口，Koakuma 仍逐项编译并负责局部 warning 组合；
- compiler config 当前主要覆盖 retrieval context，其他 target 的长度与语言策略仍由单次 options/handler 管理。

MemoryCompiler 的演进方向应是更稳定的语义 IR、更一致的预算和引用输出，而不是把检索、权限、工具执行或持久化并入 compiler。Context 是被编译出来的临时视图；Compiler 重要，正因为它不能成为新的长期真相源。
