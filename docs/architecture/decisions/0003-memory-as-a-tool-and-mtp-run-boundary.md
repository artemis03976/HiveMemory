---
title: ADR-0003 Memory-as-a-Tool 与 MTP RUN 的边界语义
status: accepted
owner: project
scope: memory-as-a-tool-asset-consumption-and-executable-boundary
decided_at: 2026-09-07
last_reviewed: 2026-09-07
related_docs:
  - docs/VISION.md
  - docs/contracts/mtp.md
  - docs/patchouli/memory-compiler.md
  - docs/patchouli/artifacts.md
  - docs/ideas/memory-as-tool-skill-plugin-boundary.md
---

# ADR-0003：Memory-as-a-Tool 与 MTP RUN 的边界语义

## Context

HiveMemory 的长期命题是让记忆资产不仅作为上下文被检索，还作为 Agent 可主动使用的能力参与执行。这一命题在当前实现中体现为两条已经落地的线索：`MemoryCompiler` 通过 target 把同一记忆编译成不同消费形态（`PROMPT_FULL`/`PROMPT_INDEX` 用于阅读，`RUNNABLE_TOOL` 作为保留的可执行 target）；MTP 的 `RUN` 动词通过两层分发执行能力（框架注册的 `sys_tool` 与 `CODE_SNIPPET` MemoryAtom）。

与此同时，业界 Agent harness 对"附件解析/能力获取"形成了两种看似对立的做法：Claude Code/Codex 的 skill（文本手册 + 配套工具代码，Agent 照着走），以及 Pi/DeepSeek 的 plugin（框架不内置，靠外部插件可插拔组合）。这两种做法容易让人试图在 HiveMemory 里为 skill 和 plugin 各建立一类记忆资产，或用 `memory_type` 去区分它们。

本决策需要澄清项目 `Memory as a Tool`（MaaT）与 `everything is memory` 理念的准确含义，并确定 `MTP RUN` 的编译语义边界，避免"统一"退化成"类别坍缩"：既不能把 skill 和 plugin 割裂成两套资产，也不能让所有记忆都获得可执行语义。

## Decision

### 1. Skill 与 Plugin 是外部能力消费形态，不是 Memory 资产类型

Skill 和 Plugin 都可能同时包含说明、代码和运行时能力，不能简单等同为 Agent **阅读**和**调用**两种严格不同的类型。它们是外部 Agent harness 组织和消费能力的不同形态，不应被直接映射为 HiveMemory 的两种 Memory。

- 在 HiveMemory 内部，`MemoryCompiler` 的 target 表达同一资产面向不同消费者的表示方式：
  - `PROMPT_FULL` / `PROMPT_INDEX` / `MTP_READ` 等目标生成 Agent 可阅读的资产视图，具有 Skill-like 的消费特征；
  - `RUNNABLE_TOOL` 目标生成 MTP RUN 所需的执行依据，具有 Plugin-like 的消费特征。

这些 target 只描述 Memory 的消费视图，不定义 Plugin 的注册、部署或运行时生命周期。因此不为 Skill 和 Plugin 各开一类记忆类型，也不引入 `skill_type`/`plugin_type` 这类把消费方式误写成资产种类的字段。同一资产可以按运行时上下文被编译到不同 target。

### 2. 记忆原子采用三层语义模型，可执行内容以引用关联

一条可执行记忆原子可以从语义上分为三层。这是资产模型，不是对当前 `MemoryAtom` 物理字段布局的立即修改承诺：

1. **index 层**：可寻址身份、别名、类型、标签等索引元数据，具有 Skill frontmatter 的导航作用；
2. **content 层**：声明式本体，即"何时使用、参数、安全边界、预期结果"这类使用指引，具有 Skill 正文的语义作用；
3. **引用层**：Memory 与外部代码、工具或其他资料之间的关联，提供执行所需的外部依据。

引用层本身不是调用入口，也不自动授予执行权。模型通过 content 获得使用指引，MTP Runtime 通过编译后的执行依据解析引用并执行。当前实现中，引用关系仍主要由 `payload.artifacts.refs` 表达；它是否演变为独立引用层，以及可执行资料是否属于 Artifact，留待后续 Plan 裁定。

这个结构的关键性质是同一资产具有**两种消费路径**：Agent 经 `READ` 获得使用指引，Runtime 经 `RUNNABLE_TOOL` 获得执行依据。"Skill 文档 + 终端命令"路径主要依赖模型和环境自行遵守这些指引；MTP RUN 的价值在于让 Runtime 也能依据结构化执行语义进行校验和治理。

### 3. 可执行性是权限化属性，不是记忆的默认形态

`everything is memory` 统一的是"任何需要跨执行周期存续、引用、修正、验证、调度或复用的资产都应具有明确身份和生命周期"，它统一的是**身份 + 生命周期**，不是**执行能力**。

`Memory as a Tool` 的统一之处是统一的工具协议入口（`SEARCH`/`READ`/`RUN`/`WRITE`/`UPDATE`），不是"每条记忆都可执行"。`RUN` 是六个动词中唯一执行能力的入口，只应服务于少数确实需要执行语义的资产；可读、可改、可执行是三种不同的权限语义。不得把"把所有记忆类型都变成可执行工具"作为目标。

### 4. MTP RUN 的编译产物是结构化执行依据，不是命令字符串

`RUNNABLE_TOOL` 的编译结果不得退化成一段自由文本命令。它应表达一次受约束的执行依据，至少能够关联：

- 可寻址身份（`memory_id`/alias）；
- 冻结的引用及其内容 hash、工具/解析器版本，保证可复现；
- 当前调用方的 `AgentProfile`、`IdentityScope` 和最终权限判断；
- 沙箱/隔离配置与可信来源审查（当强沙箱落地后）；
- 来源与观测 hook（`mtp.run` citation，执行结果回流为记忆演化输入）；
- 失败契约（结构化 `MTPErrorInfo`，不回灌裸 stderr）。

其中调用方身份、当前权限、参数、deadline、取消和本次资源限制由 MTP Runtime 在执行前绑定，不能被缓存为脱离调用方的静态 Memory 视图。具体执行后端（当前本地 subprocess，未来沙箱容器或 MCP）是 RUN 的可替换落点，不是资产语义的一部分。

附件的 `AssetRepresentation` 当前已经携带 `revision` 与 `content_hash`，但它属于 WorkspaceAsset 的进程内表示，不是通用的可执行资产模型。未来 RUN 若消费附件或其他外部资料，应建立等价的版本冻结约束；本 ADR 不把它提前提升为已经成立的全系统不变量。

### 5. 可执行内容的落地原则

长期方向是让 `payload.content` 主要表达使用指引，让执行所需代码或工具通过引用关联。现有 `CODE_SNIPPET` 的"content 即代码"语义是否以及如何迁移，引用关系由何种结构承载，均由独立 Plan 裁定；本 ADR 不冻结具体字段、Artifact 类型或存储方式。

## Consequences

正面结果：

- 记忆资产在"身份 + 生命周期"层面保持统一，同时避免 skill/plugin 割裂或类别坍缩；
- `MemoryCompiler` 的 target 为同一资产提供不同消费视图，不必为 Skill 和 Plugin 新增平行 Memory 类型；
- MTP RUN 获得可寻址、可复现、可授权、可溯源、可回流的执行语义，而不是"套了壳的终端命令"；
- 执行所需的外部资料可以与既有 provenance/版本模型关联，但具体承载方式仍需后续设计；
- 终端可以成为候选实践的来源，稳定实践是否经 `WRITE`、验证和编译晋升为可执行资产，仍需真实场景验证和独立 Plan。

代价与限制：

- 三层结构要求实现方在"引用 vs 内联"上保持纪律，避免重新把代码塞回 `content`；
- `RUNNABLE_TOOL` 的编译产物一旦进入公共契约，即构成跨子系统契约变更，需要与 `mtp.md`、`memory-compiler.md` 同步更新；
- MTP RUN 的价值依赖于 trust/reproducibility/provenance/reuse 四类需求真实出现；对一次性、无信任风险、无复用诉求的命令，终端路径仍更简单，本决策不主张用 MTP RUN 覆盖所有命令。

## Alternatives

### 为 skill 和 plugin 各建立一类记忆资产

拒绝。它把"消费方式"误写成"资产种类"，会导致同一能力出现两份可漂移的资产，并使 `memory_type` 承担它本不该承担的消费轴职责。

### 让所有记忆都可执行（everything is memory 的激进解释）

拒绝。它把身份/生命周期统一偷换成执行能力统一，绕过权限、沙箱、来源审查，直接放大"错误或恶意代码伪装成记忆"的风险，违反 [VISION.md](../../VISION.md) 关于类别坍缩与非目标的约束。

### MTP RUN 编译为自由命令字符串，由 Agent 在终端执行

拒绝。它等价于"skill 文档 + 终端命令"，把寻址、版本、授权、溯源和回流全部交还给环境与一次性 prompt，无法产生 memory-native 的结构性优势，MTP RUN 也随之失去存在意义。

### 保留 CODE_SNIPPET 的"content 即代码"，同时不引入 Artifact 引用

部分拒绝。当前 CODE_SNIPPET 可作为过渡形态保留，但作为长期语义被本决策的方向（content 指引 + Artifact 引用）替代；是否继续内联属于后续迁移 Plan 的裁定范围。

## Status

Accepted。本决策只冻结 MaaT 的语义边界：Memory 是统一资产，Skill/Plugin 是外部消费形态，MemoryCompiler target 是资产视图，MTP RUN 必须表达受约束的执行依据。它不冻结三层模型的物理实现、可执行资料的存储类型、具体执行 IR、强沙箱、信任晋升或终端实践的自动晋升机制。若这些边界需要改变，应以新的证据和独立 ADR 重新评估，而不是在局部代码中悄然改变语义。

## Related documents

- [VISION.md](../../VISION.md)
- [Memory Tool Protocol](../../contracts/mtp.md)
- [MemoryCompiler](../../patchouli/memory-compiler.md)
- [Artifacts 与来源追踪](../../patchouli/artifacts.md)
- [MaaT 与 Skill/Plugin 边界的资产消费模型（Idea）](../../ideas/memory-as-tool-skill-plugin-boundary.md)
- [ADR-0002：全局唯一身份与按需并发保护](./0002-unique-identities-and-minimal-concurrency.md)
