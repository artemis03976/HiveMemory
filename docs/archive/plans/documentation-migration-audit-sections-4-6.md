---
title: Documentation Migration Audit — Inventory Sections 4–6
status: archived
owner: project
scope: documentation-migration-audit-sections-4-through-6
archived_at: 2026-07-29
superseded_by:
  - docs/archive/plans/documentation-migration-inventory.md
  - docs/DOCUMENTATION.md
---

# 文档迁移逐篇审计：清单第 4～6 节

## 1. 审计目的与口径

本记录保存 `documentation-migration-inventory.md` 第 4～6 节在物理归档前的逐篇复核结果。审计不以“当前文档出现过相同关键词”为通过条件，而是逐项检查旧文档中的当前事实、设计动机、关键取舍、未来计划和历史假设是否获得了合适去向。

结论使用以下口径：

- **保留**：文档仍是当前入口，已按代码与现行文档体系核验；
- **拆分后归档**：事实、决策、Plan 或 Todo 已进入各自真相源，旧混合文档可以归档；
- **合并后归档**：仍有效内容已经进入当前文档，旧稿只保留历史价值；
- **直接归档**：内容属于历史设计或已完成记录，不再承担当前事实；
- **拒绝继承**：断言从未落地、已被新所有权纠正，或会把历史目标误写为当前保证。

代码与测试核验基线为 2026-07-29 工作树。本记录只处理第 4～6 节，不对清单第 7 节及以后文档给出归档结论。

## 2. 第 4 节：顶层入口与治理文档

### `docs/DOCUMENTATION.md`

结论：**保留，审计通过**。

原有分类、状态、目录职责、PR 闭环、ADR 与 Archive 规则完整。此次补充 YAML 元数据和逐篇迁移门禁，要求先把旧文档的独立主张分类为 current / ADR / Plan / Todo / history / rejected，再核对承接位置、反向链接和拒绝原因，最后才允许物理移动。

当前入口：`docs/DOCUMENTATION.md`。

### `docs/PROJECT.md`

结论：**保留并完成重写复核**。

迁移前原文第一章的“蜂巢记忆”构想、长期连续性问题、普通 RAG 的不足和成功标准，已保留在当前第 1～2 章；记忆资产、Context 编译、Memory as a Tool、热冷路径、责任分离和原始证据优先等设计理念也已保留。原第三章中仍有效的“MemoryAtom 是语义事务”与 index/payload/artifact 冰山分层已下沉到 `architecture/data-model.md`。

拒绝继承为当前事实的内容包括：固定 `<800ms P95`、百万向量容量与费用估算、自动 L3 revive、完整 Git-like 历史、固定来源置信度阶梯、旧三子系统目录和已经失效的版本状态。这些内容要么是历史目标，要么没有全系统代码保证。

当前入口：`docs/PROJECT.md`、`docs/architecture/data-model.md`。

### `docs/VISION.md`

结论：**保留，校正后审计通过**。

项目定位、Memory-Native 命题、三层理念、可证伪假设、验证体系、风险与非目标均具有持续价值。此次补充规范元数据，并修正两个工程偏差：当前边界包含 System 组合根与 Gateway/Patchouli/Alice 三个同级子系统；协议和实施文档入口改为 `contracts/`、`plans/` 与 Archive，不再推荐退役的 `protocols/`、`mod/`。

当前入口：`docs/VISION.md`。

### `docs/ROADMAP.md`

结论：**保留，审计通过**。

当前文档已区分已发布 `v0.5.0`、未发布 `v0.6.0`、Planned/Candidate/Deferred，并把详细设计指向当前文档或 Plan。数据模型治理已新增为 **Unscheduled** 计划，明确不构成 v0.6.x 版本承诺。

拒绝重新引入旧 PROJECT/设计稿中的静态版本时间表和未验证性能承诺。

当前入口：`docs/ROADMAP.md`、`docs/plans/data-model-mutability-governance.md`。

### `docs/SETUP.md`

结论：**合并后归档，审计通过**。

Python/Node/Docker 前置条件、Qdrant、开发与 Docker 端口、安装、配置来源、health/readiness、测试和常见故障已经分别由 `help/setup.md`、`help/configuration.md`、`help/troubleshooting.md` 承接。Help 文档已按当前 `8769/5173` 开发入口与 Docker `8000` 入口核验。

不继承旧 v0.1 定位、旧 LLM worker/librarian 环境变量、Patchouli 旧目录树和可能过时的单一路径命令。最终路径：`docs/archive/legacy-docs/SETUP.md`。

### `docs/TODO.md`

结论：**拆分后归档，审计通过**。

旧文档只有两个独立事项：Memory Garden 的“semantic”搜索仍是本地 substring；前端没有统一用户身份状态。它们已分别进入 `todo/frontend-memory-semantic-search.md` 与 `todo/frontend-identity-ownership.md`，并链接当前 Frontend 事实。

旧稿中的 Zustand 示例和具体文件名不继续作为完成方案；尤其 `UserStore` 不能被描述成认证或租户安全边界。最终路径：`docs/archive/legacy-docs/TODO.md`。

### `docs/ObservabilityDesign.md`

结论：**合并后归档，审计通过**。

仍有效的设计动机包括：前台/后台并发日志在单时间线交错、trace/span/task_type 分组、业务指令与观测旁路分离，以及静默失败必须可见。它们已进入 `system/observability.md`；RuntimeEvent 信封、ring buffer、replay/gap、慢订阅者隔离和 best-effort 语义由当前代码与 Contracts 重建，前端分组呈现由 Frontend 文档承接。

不继承旧 `logging.Filter + SSELogHandler` 作为既定实现、绝对“完美追踪”表述或已废弃组件名。最终路径：`docs/archive/legacy-docs/ObservabilityDesign.md`。

## 3. 第 5 节：Architecture

### `docs/architecture/README.md`

结论：**保留，审计通过**。

索引已只链接当前 overview、boundaries、data model、ADR 和相关治理 Plan；不再让迁移期混合文档或 evolution redirect 成为当前依据。

当前入口：`docs/architecture/README.md`。

### `docs/architecture/DataModelImmutabilityStatusAndRoadmap.md`

结论：**拆分后归档，审计通过**。

旧文档的事实层已经进入 `architecture/data-model.md`：`FrozenDict` 的递归范围、Identity/Turn/Topic/Gateway/Pending 的冻结现状、GatewayExecutionState 与 SemanticBuffer 的受控可变、可变累积后冻结、浅层冻结外壳和公共 DTO 引用风险。迁移前 PROJECT 中 MemoryAtom 的语义事务与冰山分层也一并承接。

长期裁定进入 ADR-0001：按模型角色选择可变性，跨边界使用与实体脱钩的只读投影。六阶段治理工作进入未排期 `plans/data-model-mutability-governance.md`。

明确没有继承为现状的断言：所有 DTO 已递归不可变、字段 frozen 等于对象图冻结、MemoryAtom 已有统一写入 API、用户输入自动不可变、relations/版本/置信度策略已经完整工作。最终路径：`docs/archive/legacy-docs/architecture/DataModelImmutabilityStatusAndRoadmap.md`。

### `docs/architecture/evolution/README.md`

结论：**直接归档，审计通过**。

该文件只承担迁移期 redirect，当前索引和历史索引已经分别由 `architecture/README.md` 与 `archive/legacy-architecture/README.md` 承担。最终路径：`docs/archive/legacy-architecture/evolution-index.md`。

### `SystemArchitecture_v2.0.md`

结论：**已直接归档，复核通过**。

冷热路径分离、检索与生成不同时间尺度、Patchouli 作为记忆管理者等理念仍可在 PROJECT 与 Patchouli 当前文档中找到。The Eye/三位一体、旧队列和旧目录属于历史架构，不能作为当前组件图。现路径：`docs/archive/legacy-architecture/SystemArchitecture_v2.0.md`。

### `SystemArchitecture_v3.0.md`

结论：**已直接归档，复核通过**。

Memory as a Tool、runtime 必须强制权限、记忆操作需要稳定协议等理念已由 PROJECT、VISION、MTP 与 Alice 文档承接。Patchouli OS、用户态/内核态绝对映射、旧 verb/runtime 位置和“文本协议优于所有 Function Calling”的绝对论断不再继承。现路径：`docs/archive/legacy-architecture/SystemArchitecture_v3.0.md`。

### `SystemArchitecture_v4.0.md`

结论：**已合并后归档，复核通过**。

System/Patchouli/Alice 分层、组合根、局部总线、公开总线、应用服务和生命周期顺序已经进入 architecture、system、patchouli 与 alice 当前文档。当前架构进一步把 Gateway 提升为同级子系统，纠正了 v4 阶段的旧边界。现路径：`docs/archive/legacy-architecture/SystemArchitecture_v4.0.md`。

### `SystemArchitecture_v4_RouterToApplicationService_Refactor.md`

结论：**已合并后归档，复核通过**。

FastAPI router 作为 transport adapter、`server/deps.py` 提供窄 service dependency、router 不访问 System/Runtime/Storage 内部对象、System 不是 God Facade、application service 不成为新万能 runtime 等取舍，已显式进入 `system/application-services.md`。实施步骤本身只保留为历史记录。现路径：`docs/archive/legacy-architecture/SystemArchitecture_v4_RouterToApplicationService_Refactor.md`。

## 4. 第 6 节：System、Contracts 与 i18n

### `docs/protocols/README.md`

结论：**退役并归档，审计通过**。

当前稳定契约入口已经由 `contracts/README.md` 承担，原索引中的 MTP、错误、scheduler 与 i18n 均有新的当前入口。最终路径：`docs/archive/legacy-docs/protocols/README.md`。

### `docs/protocols/MemoryToolProtocol.md`

结论：**重写后归档，审计通过**。

当前 `contracts/mtp.md` 已根据 parser、models、runtime、formatter 和测试重建六个 verb、权限、响应、PendingAtom、CALL 与身份边界。此次补回的有效理念包括：`⟪/⟫` 减少语法冲突、`VERB | TARGET | ARGS` 显式划分动作/对象/细节、一次只执行首条完整指令、READ 列表减少执行轮次，以及 Agent 使用各 verb 的行动门槛。

拒绝继承：MTP 优于所有 function calling、保留 chain-of-thought、任意多命令批处理、Patchouli Kernel 旧所有权、RUN 已是安全沙箱。当前还明确记录 formatter 的 XML escaping 缺口。最终路径：`docs/archive/legacy-docs/protocols/MemoryToolProtocol.md`。

### `docs/protocols/MTPErrorStructureDesign.md`

结论：**合并后归档，审计通过**。

结构化 `MTPErrorInfo`、agent/system fault、stable code、warning、cause 隐藏、handler 异常集中转换、`MTPFormatter` 唯一 Agent-facing 构造点、CALL 结构化回填等已进入 `contracts/error-model.md` 与 `contracts/mtp.md`。当前文档还澄清 `response_content` 不是完整 `formatted_response`，并记录 XML escaping 未完成。

最终路径：`docs/archive/legacy-docs/protocols/MTPErrorStructureDesign.md`。

### `docs/protocols/PatchouliUnifiedMaintenanceSchedulerDesign.md`

结论：**合并后归档，审计通过**。

线程式 scheduler、`asyncio.run()` 和临时 event loop 的失败原因，纯 asyncio 全局时钟、非重入、skip、启停/drain 和“统一调度不等于统一业务”已经进入 `system/runtime-and-bus.md`；observer idle flush 与 gardening 分别由 Patchouli Perception/Lifecycle 文档承接。

旧稿由 Patchouli 持有系统 scheduler 的所有权已被纠正：System 拥有 `GlobalMaintenanceScheduler` 生命周期，Patchouli 拥有并注册维护业务。最终路径：`docs/archive/legacy-docs/protocols/PatchouliUnifiedMaintenanceSchedulerDesign.md`。

### `docs/protocols/i18n/README.md`

结论：**退役并归档，审计通过**。

语言解析、文本域、边界与限制已由 `system/i18n.md` 统一索引。最终路径：`docs/archive/legacy-docs/protocols/i18n/README.md`。

### `I18nFoundationDesign.md`

结论：**合并后归档，审计通过**。

Language/normalize/resolve、Profile/component/global 优先级、分域 getter 和中英文基础设施已由当前代码核验。Relay 与 Generation getter 已补入当前文本领域；只翻译自然语言、不翻译协议标记和业务数据的原则也已显式保留。

未继承为现状：所有组件都完成迁移、fallback config 自动贯穿所有 getter、gettext/catalog 已有必要。最终路径：`docs/archive/legacy-docs/protocols/i18n/I18nFoundationDesign.md`。

### `I18nStatusAndRoadmap.md`

结论：**拆分后归档，审计通过**。

其中“当前完成度”已与 `src/hivememory/i18n/` getter、配置、MTP/MemoryCompiler 调用点和测试交叉核验后进入 `system/i18n.md`；历史路线不再作为未排期承诺。当前只保留真实限制：zh/en、fallback_language 未全链路接线、supported_languages 不自动校验所有 getter。

最终路径：`docs/archive/legacy-docs/protocols/i18n/I18nStatusAndRoadmap.md`。

### `KoakumaMTPBackfillTextI18nInventory.md`

结论：**合并后归档，审计通过**。

MTP runtime、syscall、formatter、MemoryCompiler 文本所有权和结构化错误边界已由 `system/i18n.md`、`contracts/mtp.md` 与 `contracts/error-model.md` 承接。逐调用点迁移清单只保留历史实施价值。

最终路径：`docs/archive/legacy-docs/protocols/i18n/KoakumaMTPBackfillTextI18nInventory.md`。

### `MemoryCompilerI18nMigrationPlan.md`

结论：**直接归档，审计通过**。

文件正文含原始 U+FFFD 编码损坏，不从损坏段落复制当前事实。其开头确认的 Phase 1～4 主迁移结果，已经由 `I18nStatusAndRoadmap.md`、当前 getter、MemoryCompiler/TimeFormatter 代码和测试交叉核验；当前剩余限制已写入 `system/i18n.md`，没有发现需要凭损坏原文另建的新 Plan/Todo。

最终路径：`docs/archive/legacy-docs/protocols/i18n/MemoryCompilerI18nMigrationPlan.md`。

## 5. 本批结论

清单第 4～6 节逐篇审计通过。物理迁移只包含本记录列出的第 4～6 节旧入口；当前文档留在原位，已经位于 `archive/legacy-architecture/` 的 v2/v3/v4 材料不重复移动。所有归档材料都必须从 Archive 索引反向指向当前替代入口。

第 7 节及以后仍保持原路径和既有状态，等待各自批次再次审计后处理。
