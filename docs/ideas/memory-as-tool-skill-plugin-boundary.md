---
title: MaaT 与 Skill/Plugin 边界的资产消费模型
status: idea
owner: alice-patchouli
scope: memory-as-a-tool-asset-consumption-and-executable-compilation
related_contracts:
  - docs/contracts/mtp.md
  - docs/patchouli/memory-compiler.md
related_docs:
  - docs/patchouli/artifacts.md
  - docs/ideas/workspace-mvp-chat-attachments-design.md
  - docs/architecture/decisions/0003-memory-as-a-tool-and-mtp-run-boundary.md
last_reviewed: 2026-09-07
---

# MaaT 与 Skill/Plugin 边界的资产消费模型

**文档状态**: Idea
**目标阶段**: Unscheduled
**适用范围**: `MemoryCompiler`、MTP `RUN`、可执行记忆资产（Patchouli 资产层 + Alice 执行层）
**前置条件**: 本 Idea 依赖已裁定边界 [ADR-0003](../architecture/decisions/0003-memory-as-a-tool-and-mtp-run-boundary.md)；不在此重新定义 Workspace、身份或 Artifact 所有权

---

## 1. 定位

本文保存关于 `Memory as a Tool`（MaaT）与"万物皆记忆"的资产消费模型探索。它不承诺任何实现排期，也不把下文草案写成当前能力。

核心问题是：业界 harness 用 skill（文本手册）和 plugin（外部可插拔能力）两种方式让 Agent 按需获得能力，HiveMemory 是否需要用记忆原子把两者"大一统"，以及 `MTP RUN` 相对"skill 文档 + 终端命令"的简单做法究竟能带来什么。

边界结论已由 ADR-0003 裁定，本文只承接这些边界并展开尚未收敛的草案与验证问题：

- skill/plugin 是消费模式轴，不是资产类型轴；
- 记忆原子采用 index / content / 引用三层结构，可执行内容以 Artifact 引用挂载；
- 可执行性是权限化属性，不是记忆的默认形态；
- MTP RUN 的编译产物是结构化执行意图，不是命令字符串。

---

## 2. 背景：两条业界路线

| 路线 | 代表 | 形态 | 能力获取方式 |
|:---|:---|:---|:---|
| skill | Claude Code / Codex | 文本手册 + 配套工具代码 | Agent 照着手册走，框架渐进披露 |
| plugin | Pi / DeepSeek harness | 框架不内置，靠外部插件组合 | Agent 通过插件协议按需插拔能力 |

两者并非冲突：最终都趋向"让 Agent 自己决定解析哪一部分，而不是人在外面完成解析再投送"。Skill 和 Plugin 都可能同时包含说明和执行资料；在本 Idea 中，它们只是两种外部能力消费形态，不作为 HiveMemory 的 Memory 类型。

这正好落在 HiveMemory 的 `MemoryCompiler` target 体系上：`PROMPT_FULL`/`PROMPT_INDEX`/`MTP_READ` 提供 Agent 可阅读的资产视图，`RUNNABLE_TOOL` 提供 MTP RUN 所需的执行视图。因此 HiveMemory 不必在 Skill 与 Plugin 之间二选一，也不必为它们各建一类记忆。

---

## 3. 三层语义模型

一条可执行记忆原子可以从概念上理解为三层，而非一条扁平的正文。三层是资产语义的分区，不是对当前 `MemoryAtom` 物理字段的确定修改：

```text
MemoryAtom
├─ index 层   —— 可寻址身份、alias、类型、标签（具有 frontmatter 的导航作用）
├─ content 层 —— 声明式使用指引：何时用、参数、安全边界、预期结果（具有 Skill 正文的语义作用）
└─ 引用层     —— Memory 与外部代码、工具或其他资料的关联
```

约束：

- 引用层不是调用入口，也不自动授予执行权；content 层负责表达使用指引；
- 可执行资料应具备版本和完整性信息，但是否由 ArtifactRef、独立引用结构或其他方式承载尚未裁定；
- 同一资产可按上下文被编译到 `PROMPT_*`（读）或 `RUNNABLE_TOOL`（跑）。

这个模型直接回答"为什么不合一成一个记忆指向工具代码"：可以由同一个 Memory 资产同时表达指引和引用关系，但不需要现在就决定引用层的物理实现，也不应把代码必然塞进 `payload.content`。

---

## 4. MTP RUN 编译语义草案

以下模型是草案，不代表当前实现，也不代表已经排期。

### 4.1 语义重释

旧语义下，`CODE_SNIPPET` 的 content 本身就是可执行代码。新语义下，MTP RUN 应表达"根据这条记忆的指引，使用它与引用层关联的工具或外部资料"，即：

```text
compile(source = 可执行记忆, target = RUNNABLE_TOOL)
  -> ToolInvocationIR（结构化执行意图）
  -> 由某后端执行（本地 subprocess / 未来沙箱 / MCP）
```

### 4.2 RUN 执行依据草案

```text
ToolInvocationIR
  memory_id / alias          # 可寻址
  pinned_reference           # 引用及其内容 hash、工具/解析器版本
  invocation                 # entrypoint + 参数约束
  sandbox_profile            # 隔离级别、资源上限、可信来源审查
  permission_decision        # 由 MTP Runtime 结合 AgentProfile + IdentityScope 得出
  citation_hook              # 触发 mtp.run 观测
  failure_contract           # 失败 -> 结构化 MTPErrorInfo，不回灌裸 stderr
```

要点：这里描述的是未来 RUN 的结构化执行依据。MemoryCompiler 根据 target 编译资产信息；当前调用方的身份、权限、参数、deadline、取消和资源限制由 MTP Runtime 在执行前绑定。Shell 只是其中一个后端，不是协议本体。

### 4.3 执行证据回流草案

`ToolInvocationIR` 的 `citation_hook` 与 `failure_contract` 只是出口；回流的实质是 run 记录驱动哪些生命周期状态迁移：

```text
run 记录（原子、revision/hash、参数、IdentityScope、结果、耗时）
  -> 结构化事件回流 Patchouli
  -> 驱动资产生命周期状态迁移
```

候选信号（草案）：

- run 成功 -> 记录一次执行结果；不能仅凭进程成功自动推进验证状态；
- 失败率与异常模式 -> 先区分输入错误、Provider 故障、策略拒绝和能力失效，再决定是否产生 vitality 或冲突信号；
- 重产物 -> 作为证据 artifact 挂回 run 记录，可检索、可审计；
- 信任等级是否能够晋升，需要独立验证、来源审查和明确策略，不能仅由连续成功次数决定。

### 4.4 分期落地草案

强沙箱（Executable Asset Sandbox）处于 deferred，但编译链路的大部分差异化可以在沙箱之前分期兑现，每期独立有价值：

1. payload 结构化契约 + RUN 参数校验 + 结构化错误（复用 `MTPErrorInfo` 的 agent_fault 通道）；
2. artifact revision/hash 冻结进 run 记录与 `mtp.run` citation（补齐可复现性）；
3. run 证据回写验证状态与 vitality（打通使用到演化的闭环）；
4. 信任分级与强沙箱（对应 ROADMAP 中 deferred 的 Executable Asset Sandbox 范围）。

前三期完成后，即使执行仍是本地执行器，MTP RUN 也可以统一提供版本、权限、结果和来源语义；这些语义也可以由终端 wrapper 自行实现，因此仍需通过真实场景验证其集中治理和复用价值。沙箱是第四期加固，不是整条链路的前置。

---

## 5. MTP RUN 相对终端命令的差异化

终端命令把"执行"留在环境与一次性 prompt 里；MTP RUN 把"执行"变成一条可治理的资产引用。差异集中在四类需求：

| 需求 | skill 文档 + 终端命令 | MTP RUN 编译后的记忆原子 |
|:---|:---|:---|
| 运行时授权 | 不会自动继承 HiveMemory 的 `IdentityScope` 和 AgentProfile 约束，需要额外 wrapper | 从 frame 冻结的 `IdentityScope` 注入 + AgentProfile 白名单 + 未来沙箱，运行时硬约束 |
| 可复现 | 依赖 PATH 上的临时版本，除非额外建立版本固定机制 | 冻结引用及其 hash、版本和运行来源 |
| 溯源 | 不会自动形成 HiveMemory 的 `mtp.run` 记录 | `mtp.run` citation + 来源和结果记录 |
| 复用/回流 | 每会话重写，结果易逝 | 可 SEARCH/CALL/跨 agent 复用，结果回流为记忆演化输入 |

因此 MTP RUN 的价值在 trust / reproducibility / provenance / reuse 四轴，而非 happy path 的便利性。对一次性、无信任风险、无复用诉求的命令，终端路径更简单，MTP RUN 不应覆盖所有命令。

终端可以承载探索与一次性粘合；当某个实践反复出现并稳定下来，它是否经 `WRITE`、验证和编译晋升为可执行资产，需要由后续流程和真实场景决定。这可能成为 HiveMemory 相对于标准 Skill 或外部 Plugin 路径的生命周期优势，但目前仍是待验证假设。

---

## 6. 与现有实现的关系

当前已经成立的事实（以代码和契约为准）：

- `MemoryCompiler` 已有 `PROMPT_FULL`/`PROMPT_INDEX`/`MTP_READ` 等 target，`RUNNABLE_TOOL` 为保留 target，显式抛 "Phase 3 reserved" 错误；
- MTP `RUN` 已实现两层分发：`sys_tool`（框架注册工具）与 `CODE_SNIPPET` MemoryAtom；
- `MemoryAtom` 与 `Artifact` 已分离，`ArtifactRef`/source artifact 当前承载不可变证据与版本；它们尚未构成通用的 Plugin 执行包模型；
- MTP RUN 当前用户代码经本地执行器运行，尚无强隔离沙箱、进程级资源限制与真取消。

尚未成立、仍属探索的内容：

- 三层分区的正式落地（content 指引 + Artifact 引用替代"content 即代码"）；
- `RUNNABLE_TOOL` 的 `ToolInvocationIR` 编译产物与公共契约；
- 可执行资产的来源/信任边界与强沙箱；
- MTP RUN 相对外部 harness + Patchouli 的结构性优势（对应 [VISION](../VISION.md) 的 H4/H5 假设）。

---

## 7. 开放问题

1. `CODE_SNIPPET` 的"content 即代码"语义如何迁移到三层语义模型？是否继续复用现有 `CODE_SNIPPET`，还是由后续设计引入新的表示方式？
2. 引用层是否继续使用现有 `ArtifactRef`，还是需要独立的引用结构？如果引用可执行资料，如何保证版本、hash、entrypoint 和来源完整？
3. content 中哪些内容属于 Agent 使用指引，哪些内容必须由 Runtime 结构化校验？如何避免从自由文本推导权限和执行入口？
4. `RUNNABLE_TOOL` 的执行依据如何与当前 `MemoryCompiler` target 对接？哪些字段由 Compiler 提供，哪些字段由 MTP Runtime 在本次调用中绑定？
5. `RUNNABLE_TOOL` 一旦进入公共契约，与 `mtp.md`、`memory-compiler.md` 的哪些字段需要同步冻结？
6. 强沙箱、可信来源审查与资源限制应在哪个治理主题下建立，是否与 [Executable Asset Sandbox](../ROADMAP.md) 合并立项？
7. 执行结果如何回流为记忆演化输入：`mtp.run` citation 扩展、独立 outcome 记录还是事件回流？哪些结果只能记录，哪些结果才允许影响验证状态或 vitality？
8. 一次 RUN 的重试、取消、过期、Provider 撤销和结果大小限制如何表达？
9. 从终端实践到可执行资产的晋升是否需要系统内捕获机制，还是完全依赖 Agent 或用户主动 `WRITE` 沉淀？
10. 如何证明 MTP RUN 相比"外部 harness + Patchouli"的结构性优势（对照基线、指标、失败样本）？

---

## 8. 升级为 Plan 的条件

本 Idea 进入 `docs/plans/` 前至少需要：

1. 有真实场景证明可执行记忆资产的 trust/reproducibility/provenance/reuse 需求存在；W1 Chat Attachments 与 `v0.7.0` Document Ingestion 是最现成的候选证据来源：默认解析器走 sys 侧、Agent 主导的选择性解析走能力包，RAW representation 是锚点、解析产物是同一资产的额外 representation，见 [Chat Attachments Idea](./workspace-mvp-chat-attachments-design.md)；
2. 明确目标、非目标、受影响的所有权（Alice 执行层 / Patchouli 资产层 / MemoryCompiler）；
3. 明确三层语义模型与当前 `MemoryAtom`、WorkspaceAsset、Artifact 之间的映射边界；
4. 冻结 `RUNNABLE_TOOL` 的执行依据以及 Compiler 与 MTP Runtime 的字段分工；
5. 明确 `CODE_SNIPPET` 的迁移与兼容策略；
6. 建立独立 Plan，并列出完成后必须更新的当前文档（`mtp.md`、`memory-compiler.md`、`artifacts.md`、`ROADMAP.md`）。

在这些条件满足前，本 Idea 不进入 Roadmap，不拆成没有独立完成语义的 Todo。
