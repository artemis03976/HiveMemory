# 模块与组件初始化规范化方案建议

这份文档旨在解决当前项目中初始化方式不统一、参数传递混乱以及测试困难的问题。请以此作为后续重构和新模块开发的准则。

(**本文档中的代码均为示例，实际方法名与类名以项目中的实际实现为准**)

---

**版本**: 1.0
**适用范围**: 所有后端 Python 模块 (`engines/*`, `patchouli/*`, `infrastructure/*`)
**核心目标**: 消除配置漂移，统一生命周期，实现完全的可测试性（Testability）。

---

## 1. 核心原则 (Core Principles)

### 1.1 显式依赖注入 (Explicit Dependency Injection)
*   **原则**: 组件**不应**在内部实例化其依赖项（如 LLM Client, Database Connection）。所有依赖必须通过构造函数 `__init__` 传入。
*   **目的**: 解耦组件与基础设施，方便单元测试时注入 Mock 对象。

### 1.2 配置对象化 (Configuration as Objects)
*   **原则**: 使用 Pydantic Model 传递配置。组件只接收属于自己的**特定配置对象**，严禁传递巨大的全局 `GlobalConfig`，严禁在 `__init__` 中解包 `**kwargs`。
*   **目的**: 明确组件需要什么参数，利用 Pydantic 进行类型校验和默认值管理。

### 1.3 自底向上装配 (Bottom-Up Assembly)
*   **原则**: 先创建叶子节点（Leaf Components），再创建编排器（Orchestrator）。编排器接收的是**已初始化好的子组件实例**，而不是子组件的配置。
*   **目的**: 降低编排器的复杂度，使其符合“开闭原则”（修改子组件构造逻辑不需要修改编排器）。

### 1.4 统一组合根 (Composition Root)
*   **原则**: `PatchouliSystem` 是全系统唯一的“装配车间”。除此类和测试代码外，业务逻辑中不应出现 `new Class(...)` 的实例化操作。

---

## 2. 详细实现规范

### 2.1 配置层规范 (Configuration Layer)

利用 Pydantic 的嵌套特性，构建与组件层级对应的配置树。

由于重构后仅主配置类 HiveMemoryConfig 负责从环境变量和 config.yaml 中加载配置，因此**仅 HiveMemoryConfig 需要继承自 BaseSettings**，其余子配置类都继承 BaseModel 以避免不必要的开销

**✅ 标准写法：**
```python
from pydantic import BaseModel

# 1. 叶子配置
class ExtractorConfig(BaseModel):
    model_name: str = "gpt-4o"
    temperature: float = 0.0

# 2. 聚合配置
class GenerationConfig(BaseModel):
    # 嵌套子配置
    extractor: ExtractorConfig
    # 自身的配置
    max_retries: int = 3
```

**❌ 禁止写法：**
*   禁止传递 `Dict` 或 `Any` 作为配置。
*   禁止组件读取 `os.getenv`（环境变量应在 Config 加载阶段处理）。

### 2.2 组件层规范 (Component Layer)

#### A. 基础组件 (Leaf Component)
只接收**基础设施依赖**和**自身配置**。

```python
class Extractor:
    def __init__(
        self, 
        llm_service: BaseLLMService,  # [依赖] 基础设施
        config: ExtractorConfig       # [配置] 专用配置对象
    ):
        self.llm = llm_service
        self.config = config

    def run(self, text: str):
        # 使用配置
        return self.llm.chat(model=self.config.model_name, ...)
```

#### B. 封装/编排类 (Orchestrator)
只接收**已初始化的子组件实例**。

```python
class GenerationOrchestrator:
    def __init__(
        self,
        config: GenerationConfig,
        extractor: BaseExtractor,      # [依赖] 已就绪的组件实例
        deduplicator: BaseDeduplicator # [依赖] 已就绪的组件实例
    ):
        self.config = config
        self.extractor = extractor     # 直接使用，无需创建
        self.deduplicator = deduplicator
```

### 2.3 工厂模式使用规范 (Factory Guidelines)

为了避免工厂泛滥，需严格遵守以下判定标准：

| 组件类型 | 特征 | 初始化方式 | 示例 |
| :--- | :--- | :--- | :--- |
| **单态组件** | 全局只有一种实现，逻辑固定 | **直接实例化** | `Extractor`, `Deduplicator` |
| **多态组件** | 有多种实现，需根据 Config 动态选择 | **工厂函数** | `BasePerceptionLayer` (Simple/Flow) |

**多态组件的 Config 写法 (利用 Discriminated Unions):**
```python
class SimplePerceptionConfig(BaseModel):
    type: Literal["simple"] = "simple"

class FlowPerceptionConfig(BaseModel):
    type: Literal["flow"] = "flow"

class PerceptionConfig(BaseModel):
    # Pydantic v2 支持 Discriminated Unions，可以根据 type 字段自动校验并选择正确的子配置
    engine: Union[SimplePerceptionConfig, FlowPerceptionConfig] = Field(..., discriminator="type")
```

---

## 3. 系统装配规范 (The Composition Root)

`PatchouliSystem` 负责所有的“脏活累活”（Wiring）。为了保持代码整洁，建议使用 **私有构建方法 (`_build_*`)** 来组织逻辑。

```python
# src/hivememory/patchouli/system.py

class PatchouliSystem:
    def __init__(self, config_path: str):
        # 1. 加载配置树
        self.cfg = load_config(config_path)

        # 2. 初始化基础设施 (单例)
        self.storage = QdrantMemoryStore(self.cfg.qdrant)
        self.llm = LiteLLMService(self.cfg.llm)

        # 3. 组装能力层 (Engines)
        # 3.1 多态组件：使用外部工厂函数
        self.perception_engine = create_perception_layer(
            config=self.cfg.perception,
            storage=self.storage,
            llm=self.llm
        )

        # 3.2 单态复杂组件：使用内部私有构建方法
        self.generation_engine = self._build_generation_engine()
        
        # 3.3 简单组件：直接实例化
        self.gateway_engine = GatewayService(self.cfg.gateway, self.llm)

        # 4. 组装人格层 (Personas) - 注入组装好的 Engine
        self.eye = TheEye(self.gateway_engine)
        self.core = TheCore(
            perception=self.perception_engine,
            generation=self.generation_engine
        )

    def _build_generation_engine(self) -> GenerationOrchestrator:
        """
        [私有构建器] 负责 Generation 模块内部的复杂组装
        """
        # A. 先创建子组件 (传入对应的子配置)
        extractor = Extractor(
            llm=self.llm, 
            config=self.cfg.generation.extractor
        )
        
        deduplicator = Deduplicator(
            storage=self.storage,
            config=self.cfg.generation.deduplicator
        )

        # B. 注入到编排器
        return GenerationOrchestrator(
            config=self.cfg.generation,
            extractor=extractor,          # 注入实例
            deduplicator=deduplicator     # 注入实例
        )
```

---

## 4. 迁移/重构清单 (Migration Checklist)

在进行代码重构时，请按以下顺序检查：

1.  [ ] **Config Check**: 检查所有 Config 类是否继承自 `BaseModel`？是否正确嵌套？
2.  [ ] **Init Check**: 检查所有类的 `__init__`，是否移除了 `**kwargs`？是否移除了内部的对象实例化（如 `self.llm = OpenAI(...)`）？
3.  [ ] **Orchestrator Check**: 检查主要模块的封装类，是否改为接收 `Instance` 而非 `Config`？
4.  [ ] **Factory Cleanup**: 删除只有一种实现的组件的 Factory 函数，改为在 `PatchouliSystem` 中直接构建。
5.  [ ] **Wiring**: 在 `PatchouliSystem` 中实现各模块的 `_build_xxx` 方法，完成依赖注入链。

遵循此规范，你的系统将获得极佳的**一致性**，并且在未来添加新功能或切换底层模型时，只需修改配置或 `PatchouliSystem` 的装配逻辑，而无需侵入业务代码。