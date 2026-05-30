# HiveMemory i18n 基建设计规划

**状态**: Draft  
**范围**: 全局语言配置与语言解析基础设施  
**目标阶段**: Memory Compiler 大规模 i18n 文案迁移之前

## 1. 背景

HiveMemory 当前已经存在若干局部语言开关，但还没有统一的语言环境。这会导致 Gateway、MTP、Agent prompt 组装、Retrieval 渲染、Relay 和 Memory Compiler 的 prompt 与注入文本上出现语言不一致。
当前可见问题包括：
1. Gateway 曾有局部配置 gateway.analyzer.prompt_language，现已移除并跟随全局 i18n。
2. MTP prompt 有局部配置 koakuma.mtp_prompt.language。
3. AgentProfile 自身携带 language 字段。
4. Relay 目前硬编码使用 zh。
5. Memory Compiler 是记忆到 Agent 的统一表达出口，但当前模板和运行时
   提示混合了中文与英文。
Memory Compiler 是后续 i18n 最关键的消费者，因为它直接决定编译后的记忆文本如何进入 Agent 上下文。在迁移具体文案之前，系统应先建立统一的全局
语言配置与语言解析规则。

## 2. 目标

本阶段只处理语言配置基础设施。
目标：
1. 在应用配置中增加全局语言配置。
2. 新增轻量级 hivememory.i18n 包，提供语言类型与语言解析能力。
3. 逐步移除组件级局部语言配置，统一跟随全局 i18n 和 AgentProfile。
4. 为 Memory Compiler、Agent prompt assembler、Gateway、Relay 和运行时
   文案后续接入 resolved language 提供清晰路径。

非目标：

1. 本阶段不迁移大段 prompt 模板。
2. 不引入 gettext、外部 locale 文件、YAML catalog 或 JSON catalog。
3. 不做自动语言检测。
4. 不修改持久化记忆 schema 或数据库记录。
5. 不在基础设施阶段迁移所有文案模板。

## 3. i18n 包设计

新增目录：

```text
src/hivememory/i18n/
|-- __init__.py
|-- types.py
`-- resolver.py
```

### 3.1 `types.py`

职责：
1. 定义系统内部统一语言枚举。
2. 归一化历史配置、用户友好输入和 locale 风格输入。
3. 延续当前项目中已经广泛使用的 zh / en 约定。

初始 API：

```python
from enum import StrEnum


class Language(StrEnum):
    ZH = "zh"
    EN = "en"


DEFAULT_LANGUAGE = Language.ZH
FALLBACK_LANGUAGE = Language.EN


def normalize_language(value: str | Language | None) -> Language | None:
    ...
```

建议兼容的输入：

1. 中文：`zh`、`zh-cn`、`cn`、`chinese`
2. 英文：`en`、`en-us`、`english`

`normalize_language()` 对未知输入返回 None，不直接抛错。这样边界调用可以保持宽容，后续如需严格校验，可在配置模型层单独增加 validator。

### 3.2 `resolver.py`

职责：
1. 提供统一语言解析函数。
2. 避免各组件分别实现自己的 fallback 链。

初始 API：

```python
def resolve_language(
    *,
    explicit: str | Language | None = None,
    profile_language: str | Language | None = None,
    component_language: str | Language | None = None,
    default_language: str | Language | None = None,
    fallback: Language = DEFAULT_LANGUAGE,
) -> Language:
    ...
```

推荐优先级：

```text
explicit
> profile_language
> component_language
> default_language
> fallback
```

语义：
1. `explicit`：单次调用显式指定，例如未来的 MemoryCompileOptions.language。
2. `profile_language`：AgentProfile 指定的人偶表达语言。
3. `component_language`：兼容历史局部配置，例如
   历史组件级语言配置。Gateway 和 MTP 的组件级语言配置已移除。
4. `default_language`：全局默认语言。
5. `fallback`：硬回退值，初始建议为 zh。

如果 Agent prompt assembler 需要严格保持旧行为，可以短期将
`component_language` 放在 `profile_language` 之前。但长期语义上，AgentProfile
和 Koakuma 的 MTP prompt 配置更具体，因此建议最终采用上述优先级。

### 3.3 `__init__.py`

只导出稳定 API：

```python
from .types import Language, normalize_language
from .resolver import resolve_language

__all__ = ["Language", "normalize_language", "resolve_language"]
```

## 4. 全局配置变更

在 src/hivememory/system/config.py 中新增 I18nConfig：

建议模型：

```python
class I18nConfig(BaseModel):
    default_language: str = Field(default="zh", description="全局默认语言 (zh/en)")
    fallback_language: str = Field(default="en", description="缺失文案时的回退语言")
    supported_languages: List[str] = Field(
        default_factory=lambda: ["zh", "en"],
        description="支持的语言列表",
    )
```

在 HiveMemoryConfig 中新增字段：

```python
i18n: I18nConfig = Field(default_factory=I18nConfig)
```

在 configs/config.yaml 中新增顶层配置：

```yaml
i18n:
  default_language: "zh"
  fallback_language: "en"
  supported_languages:
    - "zh"
    - "en"
```

当前局部配置先保留：

```yaml
gateway:
  analyzer:

koakuma:
  mtp_prompt:
    language: "zh"
```

这些字段在迁移期作为兼容性 override。待所有消费者都接入全局 i18n 后，可再考虑将其重命名为 language_override，或逐步弃用。

## 5. 第一批接入点

本阶段应避免大规模迁移文案，但可以通过构造函数和 options 预留语言参数，让后续迁移更顺畅。

### 5.1 Agent Prompt Assembler

当前问题：
1. AgentPromptAssembler 只接收 koakuma_config。
2. `_prompt_language()` 最终回退到硬编码 "zh"。
3. 文件内已有 TODO 指出需要全局 i18n。

建议改造：

```python
class AgentPromptAssembler:
    def __init__(self, koakuma_config: Any, default_language: str = "zh") -> None:
        self._koakuma_config = koakuma_config
        self._default_language = default_language
```

然后 `_prompt_language()` 使用 `resolve_language()`：

```python
return resolve_language(
    profile_language=getattr(profile, "language", None),
    component_language=getattr(prompt_config, "language", None),
    default_language=self._default_language,
).value
```

Alice runtime 构造处改为：

```python
self._prompt_assembler = AgentPromptAssembler(
    config.koakuma,
    default_language=config.i18n.default_language,
)
```

这样 Worker Agent prompt 路径可以先获得全局语言 fallback，而无需立即迁移具体 prompt 模板。

### 5.2 Memory Compiler API 预留

Memory Compiler 是下一阶段最重要的迁移目标。本阶段可以先加语言参数，不改变现有输出。

在 MemoryCompileOptions 中新增：

```python
language: Optional[str] = None
```

在 MemoryCompiler 中新增默认语言：

```python
class MemoryCompiler:
    def __init__(self, default_language: str = "zh") -> None:
        self.default_language = default_language
```

在 `compile()` 和 `wrap()` 中，如果 `options.language` 为空，则复制 options 并补上默认语言：

```python
if opts.language is None:
    opts = opts.model_copy(update={"language": self.default_language})
```

本阶段 handler 不需要消费该字段。此改动的目的只是稳定后续文案迁移所需的 API。

### 5.3 Gateway

Gateway 已有局部语言配置。本阶段不建议强制改变行为。

短期行为：
保持原样，或将 gateway.analyzer.prompt_language 与 config.i18n.default_language 一起调用 resolve_language()。

长期可选方案：

### 5.4 Relay

Relay 目前硬编码中文 prompt。它是明确的后续消费者，但可以等全局配置合入后再接入。

未来行为：

```python
get_relay_system_prompt(language=resolved_language.value)
```

### 5.5 Runtime 和 MTP 错误消息

Koakuma 和 KernelLoopExecutor 仍有较多英文硬编码运行时消息。本阶段不迁移，但需要标记为后续 catalog 目标：

1. Alias not found 类消息。
2. READ / RUN / WRITE / UPDATE 参数错误。
3. `[System MTP Execution Result]`
4. `[System IPC Return]`
5. `[Sub-Agent Reply]`
6. `[Artifacts Generated / Updated]`

## 6. 建议实施顺序

1. 新增 `src/hivememory/i18n/__init__.py`、`types.py`、`resolver.py`。
2. 在 HiveMemoryConfig 中新增 I18nConfig。
3. 在 configs/config.yaml 中新增 i18n 顶层配置。
4. 为 `normalize_language()` 和 `resolve_language()` 增加单元测试。
5. 修改 AgentPromptAssembler，接收 default_language 并使用 resolve_language()。
6. 修改 AliceRuntime，传入 config.i18n.default_language。
7. 在 MemoryCompileOptions 和 MemoryCompiler 预留语言字段与默认语言。
8. 增加回归测试，确认默认行为仍为中文，显式英文 override 仍生效。

## 7. 测试计划

最低测试范围：

1. `normalize_language()` 能识别标准值和常见 alias。
2. `normalize_language()` 对未知值返回 None。
3. `resolve_language()` 符合既定优先级。
4. HiveMemoryConfig 能从 YAML 加载 i18n，并支持环境变量覆盖，例如 HIVEMEMORY__I18N__DEFAULT_LANGUAGE=en。
5. AgentPromptAssembler 在无局部语言时回退到全局默认语言。
6. 现有 MTP prompt 的 zh / en 显式测试继续通过。
7. 新增语言 options 后，`MemoryCompiler()` 现有输出不变。

## 8. 风险与控制

### 8.1 破坏已有局部语言行为

控制方式：
1. 保留当前局部字段。
2. 将局部字段作为 component_language。
3. 增加 MTP prompt 语言行为的回归测试。

### 8.2 AgentProfile 和 Koakuma MTP 语言优先级不清晰

控制方式：
1. 在文档中明确长期推荐优先级。
2. 如必须严格兼容旧行为，可先保持旧优先级，再通过后续迁移切换。

### 8.3 增加全局配置但消费不全

控制方式：
1. 本阶段明确定位为 foundation。
2. 后续消费者单独追踪：Memory Compiler、Gateway、Relay、Koakuma runtime messages、KernelLoopExecutor IPC text。

### 8.4 过早过度设计 i18n

控制方式：
1. i18n 包仅包含 `types.py` 和 `resolver.py`。
2. 在 Memory Compiler 文案迁移开始前，不引入 catalog。

## 9. 后续工作

本基建合入后，下一阶段建议从 Memory Compiler 开始：

1. 为 Memory Compiler 增加轻量文本 catalog。
2. 迁移 Memory Compiler envelope header / footer。
3. 迁移 MemoryAtom 字段标签、confidence 标签、stale warning 和截断提示。
4. 迁移 PendingAtom ACK / READ / redirect 文案。
5. 将 Retrieval renderer 和 Koakuma 的 resolved language 传入 MemoryCompiler。
6. 扩展到 Gateway、Relay 和 MTP runtime messages。
