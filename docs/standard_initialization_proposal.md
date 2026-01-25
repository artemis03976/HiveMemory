# 模块与组件初始化规范化方案建议

采用 **“依赖注入 (DI) + 配置对象 (Configuration Object)”** 的标准范式。

以下是针对 HiveMemory 的具体规范建议：

---

### 1. 黄金法则：构造函数签名规范

所有的组件（Class）的 `__init__` 方法必须严格遵守以下两个原则：

1.  **依赖项（基础设施）必须显式注入**：不要在组件内部实例化 LLM 或 DB。
2.  **配置项（超参数）必须通过 Pydantic Model 注入**：不要解包成零散的 `**kwargs`，也不要传巨大的 Global Config。

#### ❌ 错误的写法 (目前的混乱来源)
```python
# 情况 A: 散装参数（改配置累死人）
class Extractor:
    def __init__(self, model_name="gpt-4", temp=0.7, max_tokens=2000): ...

# 情况 B: 隐式依赖（无法 Mock，无法换模型）
class Extractor:
    def __init__(self, config: GlobalConfig):
        self.llm = OpenAI(api_key=config.key) # 错误！自己在内部创建了连接
        self.threshold = config.gen.threshold
```

#### ✅ 推荐的标准写法
```python
# 1. 定义专属的小配置对象
class ExtractorConfig(BaseModel):
    model_name: str = "gpt-4o"
    temperature: float = 0.0
    threshold: float = 0.7

# 2. 构造函数：服务传实例，参数传对象
class Extractor:
    def __init__(
        self, 
        llm_service: BaseLLMService,  # [依赖] 基础设施 (已初始化好的)
        config: ExtractorConfig       # [配置] 纯数据对象
    ):
        self.llm = llm_service
        self.config = config  # 直接存下来，以后加参数不用改 __init__

    def run(self):
        # 使用时直接调 config
        if score > self.config.threshold: ...
```

**针对你关于“测试方便性”的疑虑**：
Pydantic 对象其实比散装参数更好 Mock。在测试里你只需要：
`cfg = ExtractorConfig(threshold=0.9)`，不需要构造巨大的 GlobalConfig。

---

### 2. 解决多态组件的配置：Pydantic 区分联合 (Discriminated Unions)

Pydantic v2 支持 **Discriminated Unions**，可以根据 `type` 字段自动校验并选择正确的子配置。

#### 配置层设计
不要把所有参数平铺。

```python
from typing import Literal, Union
from pydantic import BaseModel, Field

# 1. 定义具体实现的配置
class SimplePerceptionConfig(BaseModel):
    type: Literal["simple"] = "simple"  # 鉴别器
    buffer_size: int = 10

class FlowPerceptionConfig(BaseModel):
    type: Literal["semantic_flow"] = "semantic_flow" # 鉴别器
    embedding_threshold: float = 0.75
    relay_enabled: bool = True

# 2. 定义多态父配置
class PerceptionConfig(BaseModel):
    # Pydantic 会自动根据 YAML 里的 type 字段决定实例化哪一个
    driver: Union[SimplePerceptionConfig, FlowPerceptionConfig] = Field(
        ..., discriminator="type"
    )
```

#### 工厂层设计
工厂函数不再负责解析参数，只负责**路由**。

```python
def create_perception_layer(
    config: PerceptionConfig,    # 接收父配置
    storage: QdrantMemoryStore,  # 接收依赖
    llm: BaseLLMService          # 接收依赖
) -> BasePerceptionLayer:
    
    # 这里的 config.driver 已经是具体的子类型了 (Simple 或 Flow)
    if isinstance(config.driver, SimplePerceptionConfig):
        return SimplePerceptionLayer(config=config.driver)
        
    elif isinstance(config.driver, FlowPerceptionConfig):
        return SemanticFlowPerceptionLayer(
            config=config.driver, 
            storage=storage, 
            llm=llm
        )
```

---

### 3. 统一装配：PatchouliSystem (Composition Root)

所有的“布线”工作（Wiring）应该收敛到一个地方，即 **Composition Root (组合根)**。在这个项目中，就是 `PatchouliSystem`。

**除此类之外，其他任何业务代码不应出现 `new Class(...)` 的操作。**
除 HiveMemoryClient / PatchouliSystem 外，其他模块不得自行读取 YAML、不得自行 load_app_config() 、不得自行创建基础设施依赖

```python
# src/hivememory/patchouli/system.py

class PatchouliSystem:
    def __init__(self, config_path: str = "config.yaml"):
        # 1. 加载配置树 (仅在此处加载一次)
        self.cfg = load_config(config_path)

        # 2. 初始化基础设施 (Infrastructure)
        # 这些是单例，并在系统内共享
        self.storage = QdrantMemoryStore(self.cfg.qdrant)
        self.llm_service = LiteLLMService(self.cfg.llm)
        self.embed_service = LocalEmbeddingService(self.cfg.embedding)

        # 3. 初始化能力层 (Engines) - 使用工厂
        # 注意：这里我们把 config 拆解了传进去，而不是传 self.cfg
        self.perception_engine = create_perception_layer(
            config=self.cfg.perception, 
            storage=self.storage,
            llm=self.llm_service
        )
        
        self.retrieval_engine = RetrievalEngine(
            config=self.cfg.retrieval, # 这是一个 RetrievalConfig 对象
            storage=self.storage,
            embedder=self.embed_service
        )

        # 4. 初始化人格层 (Personas) - 注入 Engine
        self.eye = TheEye(
            engine=create_gateway_engine(self.cfg.gateway, self.llm_service)
        )
        self.phantom = ThePhantom(
            engine=self.retrieval_engine
        )
        self.core = TheCore(
            perception=self.perception_engine,
            # ...
        )
```

---

### 4. 总结：规范清单

为了解决你现在的痛点，请在接下来的重构中执行以下清单：

1.  **Config 原则**:
    *   每个组件（如 `Extractor`）必须有对应的 `ExtractorConfig` Pydantic 类。
    *   构造函数只接收 `(dependencies..., config: SpecificConfig)`。
    *   **禁止**在构造函数中解包 `**kwargs`。

2.  **Dependency 原则**:
    *   所有“重量级”对象（DB, LLM Client, Redis）必须在外部初始化，通过参数传入。
    *   **禁止**在组件内部读取 `os.getenv` 或创建新的连接。

3.  **Factory 原则**:
    *   工厂函数只用于处理**多态**（if type == A return ClassA）。
    *   对于非多态组件，直接在 `PatchouliSystem` 里实例化，不需要写工厂。

4.  **Configuration Structure**:
    *   使用 Pydantic 的 `Union` + `discriminator` 来处理多态配置，避免 `config.type` 和参数不匹配的问题。

通过这套方案，你的代码将变得极其**可测试**（Mock 依赖，构造 Config 对象）且**配置安全**（Pydantic 会在启动时拦截配置错误，而不是运行时报错）。这是 Python 后端项目的最佳实践。
