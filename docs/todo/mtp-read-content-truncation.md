---
title: MTP READ 截断正式记忆的正文
status: todo
owner: alice
scope: mtp-read-formal-atom-content-truncation
code_paths:
  - src/hivememory/agent_runtime/mtp/verbs/read.py
  - src/hivememory/engines/memory_compiler/models.py
  - src/hivememory/engines/memory_compiler/handlers/mtp.py
  - src/hivememory/engines/memory_compiler/handlers/prompt.py
related_docs:
  - docs/contracts/mtp.md
  - docs/patchouli/memory-compiler.md
  - docs/ROADMAP.md
last_reviewed: 2026-10-30
---

# MTP READ 截断正式记忆的正文

## 问题与证据

READ 是 Agent 检查记忆全文的指令，但 Alice 经 MTP READ 读到的正式记忆正文最多约 500 字符：

- [`handle_read`](../../src/hivememory/agent_runtime/mtp/verbs/read.py) 以 `MemoryCompileOptions(requested_alias=alias)` 编译 `MTP_READ` 目标，没有设置 `max_content_length`，取 `MemoryCompileOptions` 的默认值 500；
- 正式原子经 `_render_full_from_ir` 渲染，正文超过上限时在上限以内、后半段的句子边界处截断，并追加截断提示“如需阅读完整内容请使用 READ 指令读取”（英文为 “Use READ to inspect the full memory content.”）；
- Agent 照提示再次 READ，得到的仍是同样的截断结果，超出部分经 MTP 无法读到。redirect 与 pending 的 READ 文本不截断，只有正式原子受影响。

2026-10-29 实施 MCP 读取工具时发现。MCP 的读取工具与 MTP READ 共用交付规则与 `MTP_READ` 编译目标，但传入不截断的上限，不受此问题影响。

## 影响范围

- Alice 读取较长的正式记忆（代码片段、长篇说明）时只看到前段，可能依据不完整的内容作答；
- 截断提示指引的动作无效，可能诱发重复 READ，浪费执行轮次；
- [MTP 契约](../contracts/mtp.md)第 3.2 节没有说明 READ 会截断，Agent 与读者都无从得知结果不完整。

可能的处理：MTP READ 与 MCP 读取一样交付全文；或按 token 预算截断，并给出能读到其余内容的方式（片段或偏移），截断提示不再指向会得到同样结果的 READ。后者属于 [ROADMAP](../ROADMAP.md#45-mtp-read-专项编译与来源表达候选)第 4.5 节 MTP READ 专项编译的片段定位、token 预算与截断表达。

## 完成条件

- MTP READ 交付正式原子时，要么交付全文，要么截断后给出读到其余内容的方式；截断提示不再指向会得到同样结果的 READ；
- 回归覆盖正文超过截断上限的正式原子经 MTP READ 的输出；
- MTP 契约第 3.2 节与 [MemoryCompiler](../patchouli/memory-compiler.md) 按最终行为更新。

## 追踪

- 2026-10-29：记忆访问（MCP）计划阶段三审查时发现。owner 意见：大概随 v0.7.3 MTP READ 专项编译（Candidate，尚未建立 Plan）一起修复。未绑定 Issue。
- 2026-10-30：v0.7.0 之后的计划不预先排定顺序（owner），MTP READ 专项编译撤销 v0.7.3 的版本号，改为不绑定版本的候选方向；本 Todo 仍随它修复，未排期。
