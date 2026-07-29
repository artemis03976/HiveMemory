---
title: System Configuration and Registries
status: current
owner: system
scope: configuration-loading-and-global-registries
code_paths:
  - src/hivememory/system/config/
  - src/hivememory/system/provider_registry.py
  - src/hivememory/system/model_registry.py
  - configs/config.yaml
related_contracts:
  - docs/architecture/boundaries.md
  - docs/contracts/subsystem-contracts.md
last_reviewed: 2026-07-28
---

# System 配置与注册表

配置是 System 的装配输入，不是另一套运行时控制 API。它需要同时满足两个现实：开发者希望通过 YAML 和环境变量管理整套服务，子系统又必须拥有自己那部分语义和默认值。当前做法是由 `HiveMemoryConfig` 统一加载和校验，再把子模型注入对应宿主；System 不在请求路径里重新解释 Gateway、Patchouli 或 Alice 的内部配置。

## 1. 配置树

`HiveMemoryConfig` 当前包含：

| 区域 | 主要内容 | 责任边界 |
|:---|:---|:---|
| `system` / `logging` | 名称、调试标志、日志输出 | System/基础设施 |
| `scheduler` | tick、关闭等待、observer/perception/GC 任务开关与间隔 | System runtime |
| `runtime_events` | 是否启用、ring buffer 和订阅队列大小 | System observability |
| `i18n` | 默认语言、fallback 字段、支持语言列表 | 全局文本解析 |
| `shared` | LLM、embedding、provider credentials | Registry 与共享模型能力 |
| `gateway` | interceptor、commands、workflow、topic router、query analysis | Gateway |
| `passive_ingress` | dedup、turn buffer、outbox 上限 | System passive ingress |
| `memory_compiler` | 编译策略 | MemoryCompiler 所有者 |
| `patchouli` / `alice` | 各自运行时和存储配置 | 对应子系统 |

System 只直接拥有顶层基础设施和 passive ingress 配置；Gateway 的 workflow timeout、Patchouli 的 retrieval 和 Alice 的 MTP 权限仍由各自所有者解释。

## 2. 来源与优先级

当前 `BaseSettings` 的来源顺序为：

```text
显式构造参数
  > HIVEMEMORY__* 环境变量
  > .env / configs/.env
  > 旧环境变量别名映射
  > provider 动态凭证扫描
  > configs/config.yaml（或 HIVEMEMORY_CONFIG_PATH 指定文件）
  > file secrets
```

嵌套环境变量使用 `HIVEMEMORY__` 前缀和 `__` 分隔，例如：

```text
HIVEMEMORY__SCHEDULER__TICK_SECONDS=1
HIVEMEMORY__GATEWAY__WORKFLOW__DEFAULT_REQUEST_TIMEOUT_MS=8000
HIVEMEMORY__PROVIDERS__DEEPSEEK__API_KEY=...
```

旧的 `LLM__...`、`QDRANT__...` 等形式仍通过显式 alias 映射兼容。动态 provider 名不能由静态 alias 枚举，因此由 `provider_credentials_settings_source()` 单独扫描并归入 `shared.providers`。

默认配置文件不存在时使用默认值和环境变量；显式指定的 `HIVEMEMORY_CONFIG_PATH` 不存在会抛 `FileNotFoundError`。YAML 解析失败当前记录 error 并返回空配置源，最终由 Pydantic 默认值和其他来源继续构造。

## 3. Registry 解析

`SystemAssembler` 在装配时：

1. 用共享 provider 配置创建 `ProviderRegistry`；
2. 创建引用该 registry 的 `ModelRegistry`；
3. 解析 Gateway 和 Librarian 的 LLM config，把 `model_id` 对应的 model/provider/api key/api base 补齐；
4. 将解析后的配置传入子系统。

这一步把凭证解析从业务请求热路径移开，也避免 Gateway 和 Patchouli 对同一模型引用各自得出不同结果。注册表提供模型和 provider 元数据，但不拥有某次请求的执行状态。

## 4. 配置所有权与兼容边界

- Gateway timeout 由 `gateway.workflow.default_request_timeout_ms` 控制；Chat 应用只能传入更小的 request timeout，不能扩大系统默认 deadline；
- Passive idle interval/timeout 由 `scheduler.tasks` 单一持有，`passive_ingress` 不重复定义同一事实；
- RuntimeEvent 的 buffer/queue 配置只影响观测容量，不改变业务状态；
- 项目版本属于构建事实，不是运行配置；`system.version` 已从配置模型和示例 YAML 中移除，版本唯一来源是 `src/hivememory/_version.py`；
- `i18n.default_language` 在配置校验后同步到进程级 resolver，但请求级显式语言仍由调用方或 Profile 传递。

配置扩展必须先判断它属于哪一个所有者；不能为了方便在 `HiveMemoryConfig` 添加一个字段，然后让多个子系统各自解释不同含义。

## 5. 当前限制

- 部分历史环境变量仍保留兼容映射，清理前不能假设只有嵌套新格式；
- `ConfigDict(extra="ignore")` 在多个顶层子模型上保持兼容，未知字段不一定立即暴露为配置错误；
- registry 只在进程装配时形成解析结果，运行中 provider/model 配置变更不会自动热重载；
- `I18nConfig.fallback_language` 当前没有被统一传入 `resolve_language()`，相关限制见[i18n 文档](./i18n.md)。

## 6. 配置变更检查

1. 新字段是否有唯一所有者和唯一真相源？
2. 是否改变了来源优先级或旧 alias 的含义？
3. 凭证是否仍只通过环境变量/secret 进入，而不会写入受版本控制的 YAML？
4. 该字段是装配期事实还是请求期控制？是否被错误地在两处同时读取？
5. 配置失败是应该阻止装配，还是允许明确的保守默认？是否有测试证明？

## 7. 验证入口

- `tests/unit/system/test_config_agent_runtime.py`
- `tests/unit/system/test_model_registry.py`
- `tests/unit/system/test_provider_registry.py`
- `src/hivememory/system/config/__init__.py`
