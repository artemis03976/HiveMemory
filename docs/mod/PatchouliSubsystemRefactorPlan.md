# Patchouli 子系统重构规划

**文档状态**: 规划中  
**关联版本**: v0.5.3（预计）  
**前置条件**: v0.5.2 全部 Phase 完成（AsyncQdrantClient + generation/retrieval async-native）

---

## 1. 背景：LibrarianCore 是 v4 架构演进的残留

### 1.1 历史演变

旧系统（v3）的结构非常直接：`PatchouliSystem` 对外承接 API，`PatchouliKernel` 承载运行时，`Koakuma` 负责 Agent 工具调用，`RetrievalFamiliar` 负责检索，**剩余所有记忆编排工作被封装进 LibrarianCore**。

v4 架构演进时，Agent 部分（Alice）被完全分离出去，但 Patchouli 的记忆管理部分基本未动——LibrarianCore 成为了"Agent 分离后剩余的一切"的容器。

`MemoryGenerationTaskController` 的出现（v0.5.0）是第一次有意识地从 LibrarianCore 中解构职责，但当时没有意识到这是解构 LibrarianCore 本身的起点。**LibrarianCore 才是需要被解构的那一个。**

### 1.2 现有结构的问题

LibrarianCore 同时做了两类性质不同的事：

| 职责 | 性质 | 问题 |
|---|---|---|
| 感知回调路由 | 编排逻辑 | 应属于流水线，不应是类方法 |
| Topic context 拼装 | 领域逻辑碎片 | 从引擎泄漏进来的逻辑 |
| 记忆生成任务管理 | 运行时编排 | 已拆出为 TaskController，但仍作为内部组件 |
| 对外暴露任务 API | 代理层 | 只是透传 TaskController 接口 |

`LibrarianCore` 这个名字隐含了"图书管理员管理与图书库的一切交互"的隐喻，理论上三级记忆系统的管理都应由它完成。而如今，**整个 Patchouli 子系统已经是这个隐喻的具象化**——它不应该是一个类，而是整个子系统的设计思路。

### 1.3 三级记忆体系的正确定位

三级记忆体系（短期 buffer → 中期向量库 → 长期冷存储）是**面向存储的概念框架**，而不是服务边界的划分依据。

记忆操作本身具有天然的跨层性：一次完整的用户交互会同时经过感知（短期）→ 查重（中期）→ 生成（写中期）→ 生命力更新（中期 lifecycle）→ 归档（长期）。强行按存储层划分三个平行服务，会倒逼重建一个等价于 LibrarianCore 的协调者。

正确用法：三级模型指导"数据该存哪里"与"状态如何转移"，而不是"代码怎么拆"。

---

## 2. 目标架构

### 2.1 总体结构

```
PatchouliRuntime（极薄的对外协调层）
├── MemoryLibrary            ← 三级存储 + 状态转移
├── RetrievalFamiliar        ← 全层检索服务（扩展）
└── MemoryIngestionPipeline  ← 生成写入管道
      └── MemoryGenerationTaskController（已存在）
```

**图书馆隐喻的对应**：
- `MemoryLibrary`：书库，管书的物理存放和流转
- `RetrievalFamiliar`：使魔，代理人去库里取书
- `MemoryIngestionPipeline`：写书流程，新书入库的完整流水线

### 2.2 MemoryLibrary

统一的三级存储层，负责记忆的物理存放和各层之间的状态转移。

```
MemoryLibrary
├── ShortTermStore    ← SemanticBufferManager（当前在 PerceptionLayer 内）
│     - 内存态 buffer
│     - topic 创建/归档/查询
│     - flush 触发（短期 → 中期 转移入口）
│
├── MidTermStore      ← QdrantMemoryStore（当前直接暴露）
│     - 向量数据库 CRUD
│     - 精确/模糊检索（由 RetrievalFamiliar 调用）
│
└── LongTermStore     ← ArchiveStore（待实现）
      - 冷存储（文件系统 / SQL）
      - 归档记录管理
      - 复活键（revival_keys）管理
```

**状态转移协议**（Library 内部管理，外部只感知事件）：

```
ShortTermStore.flush() → MidTermStore.upsert()   # 短期 → 中期
MidTermStore.archive() → LongTermStore.persist() # 中期 → 长期
LongTermStore.revive() → MidTermStore.upsert()   # 长期 → 中期（复活）
```

### 2.3 RetrievalFamiliar（扩展）

从"仅限中期向量检索"扩展为"三层统一检索入口"。

| 检索类型 | 对应层 | 当前状态 |
|---|---|---|
| Topic buffer 查询、话题快照 | 短期 | 分散在 LibrarianCore 方法中 |
| 精确取回（alias）、模糊语义检索 | 中期 | 已实现 |
| 冷存储复活检索（revival_keys） | 长期 | 待实现 |

将 LibrarianCore 中现有的 topic context 相关方法（`get_topic_context`、`get_active_topics_snapshots`）迁移至 RetrievalFamiliar，对外提供统一的"取书"接口。

### 2.4 MemoryIngestionPipeline

已基本成型（v0.5.0）。主要工作是将其从 LibrarianCore 内部组件升格为独立的流水线服务。

```
InteractionArtifactBuilder
    ↓ (topic raw interaction snapshot)
MemoryGenerationTaskController
    ↓ (compute → artifacts → persist 三步)
MemoryLibrary.MidTermStore
```

LibrarianCore 目前承担的"感知回调路由"职责（`_on_generate_memory`、`run_active_generation`）迁移至 Pipeline 的入口。

---

## 3. Config 重构

### 3.1 目标结构

```python
class PatchouliConfig(BaseModel):
    storage: QdrantConfig              # 向量数据库连接
    embedding: EmbeddingConfig         # Embedding 服务
    perception: PerceptionConfig
    generation: GenerationConfig
    lifecycle: LifecycleConfig
    retrieval: RetrievalConfig
    artifacts: ArtifactStoreConfig

class AliceConfig(BaseModel):
    runtime: AliceRuntimeConfig        # loop executor、frame scheduler 等

class SharedConfig(BaseModel):
    llm: LLMConfig                     # 跨子系统共用的 LLM 服务配置

class HiveMemoryConfig(BaseModel):
    patchouli: PatchouliConfig
    alice: AliceConfig
    shared: SharedConfig
```

### 3.2 迁移要点

- 当前 `HiveMemoryConfig` 中的字段按归属拆分，保持字段名不变，只改路径
- `PatchouliRuntime` 接收 `PatchouliConfig`，`AliceRuntime` 接收 `AliceConfig`
- `load_app_config()` 继续返回 `HiveMemoryConfig`，向下传递时各自解包子 config
- YAML/环境变量结构同步调整：`patchouli.qdrant.*`、`alice.runtime.*` 等

---

## 4. 迁移时序

### 前置：v0.5.0 完成（当前进行中）

- [x] Phase 1: ArtifactStore + 数据模型
- [x] Phase 2: InteractionArtifact 接入真实流
- [x] Phase 3: MemoryCreation/Version artifact + 三步流水线
- [ ] Phase 4: ColdPathTaskRunner（异步任务调度）
- [ ] Phase 5: RetrievalRequestEnvelope（检索生命周期事件）

### Step 1: Config 重构（可与 Phase 4/5 并行）

风险最低，边界最清晰。拆分 config 后，PatchouliRuntime 和 AliceRuntime 的职责边界将在代码层面显式化，为后续结构迁移提供稳定基础。

### Step 2: MemoryLibrary 建立（v0.6.0 核心）

1. 将 `SemanticBufferManager` 从 PerceptionLayer 提取为 `ShortTermMemoryStore`
2. 将 `QdrantMemoryStore` 包装为 `MidTermMemoryStore`（可能只是命名和分层，不改底层实现）
3. 定义 `MemoryLibrary` 统一接口，实现状态转移方法
4. `LongTermMemoryStore` 接口占位，供 lifecycle 归档写入

### Step 3: RetrievalFamiliar 扩展

将 LibrarianCore 中 topic context 相关方法迁移至 RetrievalFamiliar，使其成为真正的全层检索入口。

### Step 4: MemoryIngestionPipeline 独立化

将 `_on_generate_memory`、`run_active_generation` 等入口逻辑从 LibrarianCore 提取，形成独立的 Pipeline 入口。

### Step 5: LibrarianCore 退场

完成上述步骤后，LibrarianCore 剩余的职责已基本清空。此时可以：
- 将其保留为向后兼容的轻量代理（方法委托到 Library/Pipeline）
- 或直接删除，由 PatchouliRuntime 承担剩余的薄路由逻辑

---

## 5. 验收标准

- `LibrarianCore` 不再直接持有 `generation_engine` 引用
- `LibrarianCore` 不再直接构建 `GenerationContext` / transcript
- `MemoryLibrary` 可作为独立单元测试（不依赖 Patchouli 上层）
- `RetrievalFamiliar` 的公共接口覆盖短期 / 中期 topic 查询
- Config 按子系统分组，`PatchouliRuntime` 只接收 `PatchouliConfig`
- 所有现有 v0.5.0 测试继续通过

---

## 6. 注意事项

**lifecycle 的归属**：LifecycleEngine 同时维护中期记忆（vitality 分数）和触发长期归档，不应强行分裂。它作为独立引擎保持不变，由 MemoryLibrary 的状态转移事件触发，不归属于任何单一存储层。

**渐进式迁移**：每个 Step 都应可独立完成和测试，不存在"必须一次性全部完成"的大版本切换。PatchouliRuntime 的接口对外保持稳定。

**不在本次规划范围内**：记忆 split/merge 机制、LongTermStore 的完整实现、MTP READ 历史编译。
