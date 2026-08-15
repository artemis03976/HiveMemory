---
title: Test Design Standards
status: governance
owner: project
scope: test-authoring-standards-anti-patterns-and-quality-gates
code_paths:
  - tests/
related_docs:
  - docs/todo/testing-suite-audit.md
  - pyproject.toml
  - .github/workflows/ci.yml
last_reviewed: 2026-08-15
---

# 测试设计规范

## 1. 背景与目的

HiveMemory 现有测试数量充足（约 2009 个用例、覆盖率 88.99%），但逐文件审查发现大量测试**无法失败**——
恒真断言、构造后回读刚赋的值、断言 mock 返回值、在测试内重新实现一遍生产逻辑、把关键断言包在 `if` 条件里
条件不满足即静默通过，甚至整个函数没有任何 assert。这些测试提供的是虚假的安全感：数量在增长，但一次真实回归
依然可能畅通无阻。

本规范约束**今后新编写与修改的测试**，目标是让每一条测试都满足一个最低标准：**它必须在至少一个维度上能够失败**，
失败时才真正暴露缺陷。存量的清理与归类见[测试体系问题清单](../../todo/testing-suite-audit.md)，本规范不重复逐条修复计划。

本文是跨版本持续生效的工程质量治理主题，不绑定某个版本的实施方案。只有满足[升级门槛](#10-升级为独立-plan-的门槛)
的机制建设才从本文提取为独立 Plan。

## 2. 核心原则

1. **测试必须能失败**：一条无法在任何输入或实现变更下失败的测试不是测试，应当删除。写完测试后自问
   "我改坏生产代码，这条测试会红吗？"——如果不会，说明断言没有约束力。
2. **断言可观察行为，而非实现**：验证输入到输出、状态转移、副作用；不验证"某个 mock 被调了几次、
   传了什么中间参数"、不验证对象私有属性的内部结构。实现方式变了但行为不变，测试不应红。
3. **被测边界内使用真实对象**：本次声称要验证的对象和协作边界必须真实；边界之外的协作者、外部网络、
   第三方服务和时间源可以使用 fake/mock。用 mock 冒充被测对象、再用断言回读 mock 返回值，是最常见的
   无效测试来源。
4. **一个测试验证一件事**：单一失败原因。不要在一个测试里塞入多个不相关的断言，也不要用 `or`/条件守卫
   把多个期望折叠成"任一成立即通过"。
5. **测试金字塔**：底层大量单元测试，中层少量集成测试，顶层更少端到端测试。关键不是数量，而是每一层的
   被测对象必须是真实的。

## 3. 测试分层与 marker

测试分类采用两个相互独立的维度：

1. **主类型**描述测试覆盖范围。每个测试必须且只能属于 `unit`、`integration`、`e2e` 之一；
2. **运行条件标签**描述执行测试需要的资源。`real_infra`、`live_llm`、`slow` 可以与主类型叠加。

测试类型由本次实际验证的边界决定，而不是由文件大小、mock 数量、组件数量或执行耗时决定。

### 3.1 互斥的主类型

| 主类型 | 定义 | 本次必须真实的对象 | 允许替换的边界 | 运行时机 |
|:---|:---|:---|:---|:---|
| `unit` | 验证一个内聚行为单元的输入输出、状态迁移或副作用 | 被测行为单元 | 其协作者、外部网络、第三方服务、时间源 | 每次 PR 必跑 |
| `integration` | 验证两个以上真实内部组件的协作，或验证适配器与真实依赖的兼容性 | 本次声称要验证的全部协作边界 | 本次边界之外的外部端口 | 每次 PR 必跑；带资源标签时进入对应任务 |
| `e2e` | 从系统公开入口进入，穿过主要架构边界，到达用户可观察终态 | 内部系统链路 | 不可控外部服务可使用协议级 fake | PR 冒烟 / 手动 / nightly |

`unit` 中的“单元”不机械等同于一个函数或一个类；一个状态机、Reducer 或内聚领域服务都可以是一个行为单元。

`e2e` 不以调用链长度单独判断。通常应同时满足：

- 从 HTTP、WebSocket、事件入口或系统 facade 等公开入口进入；
- 不直接调用内部私有方法绕过正常链路；
- 穿过主要架构边界；
- 最终断言响应、事件流、持久化结果或权限拒绝等用户可观察结果。

仅串联若干内部类、但没有从公开入口走到可观察终态的测试，仍然属于 `integration`。

### 3.2 正交的运行条件标签

| 标签 | 含义 | 典型运行时机 |
|:---|:---|:---|
| `real_infra` | 需要预先部署或下载的真实基础设施，如 Qdrant、真实 Embedding/Reranker 模型 | 专用 CI / 手动 / nightly |
| `live_llm` | 调用真实 LLM Provider，需要密钥、可能产生费用且存在非确定性 | 手动 / nightly |
| `slow` | 超过项目约定的快速反馈时长 | 手动 / nightly |

`real_infra` 当前保持为一个泛化标签。只有当 CI 确实需要分别调度 Qdrant、Embedding 等资源时，才新增更细标签，
避免提前制造分类负担。

### 3.3 分类硬性约束

- 每个测试必须且只能有一个主类型；同一文件若混合了不同主类型，应拆分文件；
- `tests/unit/`、`tests/integration/`、`tests/e2e/` 是主类型的唯一归类来源，不要求在每个测试函数上重复声明；
  在自动 marker 机制落地前，以目录选择为准；落地后由收集阶段统一补充并校验；
- `unit` 不得带 `real_infra` 或 `live_llm`；调用真实 Provider 或真实基础设施的测试至少属于 `integration`；
- mock/fake 是否存在不能单独决定测试类型。替换了本次声称要验证的对象或协作边界，测试就必须降级；
  只替换边界之外的外部端口，不影响其作为 `integration` 或 `e2e`；
- 验证 `Service -> Bus -> Handler` 协作时，这些组件必须真实，但最外层存储可以是 `InMemoryStore`；
  验证 `QdrantStorageAdapter` 时，Qdrant 必须真实，并标记为 `integration + real_infra`；
- `e2e` 的内部系统组件必须真实；可以在协议边界使用确定性假 LLM。使用真实 LLM 时叠加 `live_llm`，
  使用真实 Qdrant/Embedding/Reranker 时叠加 `real_infra`；
- `live_llm` 测试必须能运行。任何因构造签名过时、缺 import 等原因无法运行的 live 测试等同死代码，
  必须修复或删除。

### 3.4 典型归类示例

| 场景 | 归类 |
|:---|:---|
| `MemoryDeduplicator` 使用 fake store 验证 CREATE/UPDATE/DISCARD 决策 | `unit` |
| `MemoryGenerationService + RealBus + InMemoryStore` 验证内部组件协作 | `integration` |
| `QdrantMemoryStore` 连接真实 Qdrant 验证写入、过滤与查询 | `integration + real_infra` |
| LiteLLM 适配器调用真实 Provider 验证请求/响应协议 | `integration + live_llm` |
| 从 `/api/v1/chat` 进入并最终验证记忆落库，LLM 使用确定性协议 fake | `e2e` |
| 同一完整链路使用真实 Qdrant 与真实 LLM | `e2e + real_infra + live_llm` |

### 3.5 运行集合

| 集合 | 包含范围 | 目的 |
|:---|:---|:---|
| PR 快速集 | 全部 `unit` + 不带 `real_infra`/`live_llm`/`slow` 的 `integration` | 快速、确定性的开发反馈 |
| 确定性 E2E | 不带真实资源标签的 `e2e` | 验证公开入口到终态的主链路 |
| 真实基础设施 | 带 `real_infra`、不带 `live_llm` 的 `integration`/`e2e` | 验证 Qdrant、Embedding、Reranker 等兼容性 |
| Live LLM | 所有带 `live_llm` 的 `integration`/`e2e` | 验证真实 Provider 协议或完整 Live 链路 |
| Nightly | 确定性 E2E + `real_infra` + `live_llm` + 必要的 `slow` | 周期性发现环境、兼容性与长链路问题 |

本地默认运行 PR 快速集。真实资源任务应显式选择，不得因为默认排除而长期无人运行。

## 4. 断言规范

### 4.1 禁止的断言模式（反模式清单）

以下模式经审计证实是无效测试的主要来源。新代码中出现即应在 code review 中拦截：

| 编号 | 反模式 | 识别特征 | 为什么有害 |
|:---|:---|:---|:---|
| A1 | 恒真/同义反复 | `assert x == x`、`assert obj.a is obj.a`、构造对象后断言其字段等于刚赋的值 | 永远成立，零约束力 |
| A2 | 写死逻辑 | 测试内重新实现一遍生产公式/序列化/映射，再断言与生产结果一致 | 只能证明"测试副本与实现一致"，生产改错时测试副本同步改才不挂 |
| A3 | mock 镜像 | `mock.return_value = X` 后 `assert result is X` | 断言的是自己塞进去的值，业务处理被清空也全绿 |
| A4 | 只验 mock 调用 | 断言 `assert_called_once_with(...)`、`await_count == N`、中间参数细节 | 验证实现而非行为，重构即挂 |
| A5 | 宽松断言 | `assert result is not None`、`len(x) > 0`、`score >= 0`、`>= 0` | 只证明"没抛异常"，不验证内容正确性 |
| A6 | 条件守卫 | `if cond: assert ... else: logger.warning(...)` | 条件不满足时静默通过，测试永远绿 |
| A7 | 无断言 | 测试体只有调用 + 打印/日志 | 纯脚本冒充测试 |
| A8 | 私有实现断言 | 访问 `_private`、`not hasattr(obj, "_x")`、私有字典结构 | 实现改动即挂，与行为无关 |

### 4.2 正确写法对照

**A1 恒真 —— 反例与正例**

```python
# 反例：构造后回读刚赋的值
def test_retrieval_query_filters(self):
    query = RetrievalQuery(filters=QueryFilters(memory_type=MemoryType.USER_PROFILE))
    assert query.filters.memory_type == MemoryType.USER_PROFILE  # 恒真

# 正例：验证生产转换/默认值/校验行为
def test_retrieval_query_defaults_memory_type(self):
    query = RetrievalQuery(filters=QueryFilters())  # 不显式传 memory_type
    assert query.filters.memory_type == MemoryType.FACT  # 验证生产默认值
```

**A2 写死逻辑 —— 反例与正例**

```python
# 反例：复制生产公式
def test_access_boost(self):
    expected = self.config.access_boost_coef * math.log(1 + 5)
    assert abs(actual - expected) < 0.5

# 正例：断言确定性的期望值，公式变更时测试无需同步修改
def test_access_boost(self):
    result = self.engine.calculate(...)
    assert result.vitality == pytest.approx(EXPECTED_ABSOLUTE_VALUE, rel=0.05)
```

**A3/A4 mock 镜像与只验调用 —— 反例与正例**

```python
# 反例：mock 返回值镜像
def test_service_create(self):
    bus.request = AsyncMock(return_value=atom)
    result = await service.create(...)
    assert result is atom  # 恒真

# 反例：只验调用细节
def test_service_create(self):
    await service.create(...)
    bus.request.assert_awaited_once_with(ROUTE, arg1=..., arg2=...)

# 正例：用真实 bus 或真实 store，验证可观察结果
def test_service_create(self):
    store = InMemoryStore()
    service = Service(bus=RealBus(store=store))
    await service.create(...)
    assert store.get(...) is not None  # 验证真实副作用
    assert store.get(...).title == "..."  # 验证内容正确
```

**A6 条件守卫 —— 反例与正例**

```python
# 反例：LLM 未触发路径时静默通过
def test_write_creates_memory(self):
    if "WRITE" in commands:
        assert "WRITE" in commands
    else:
        logger.warning("WRITE 未触发")

# 正例：前置条件直接断言，不满足即失败
def test_write_creates_memory(self):
    commands = run_loop(...)
    assert "WRITE" in commands  # 未触发即失败
```

### 4.3 异常断言

- 禁止 `pytest.raises(Exception)` 捕获一切异常；必须收敛到具体异常类型（`ValueError`、`ValidationError`、
  `TypeError`），必要时用 `match=` 校验错误信息。
- 恒真风险提醒：异常断言同样可能失效，例如把"必然抛异常"的操作包进 `raises` 后不再验证后续行为。

### 4.4 浮点数断言

涉及浮点比较必须使用 `pytest.approx(..., rel=...)`，不得使用 `==`。时间、分数、衰减系数等一律遵循此规则。

## 5. mock 使用规范

选择优先级（从高到低）：

1. **真实对象**：被测对象及其直接依赖，只要不触发网络/慢 IO，一律用真实对象；
2. **手写 fake**：需要隔离外部存储或时间源时，使用内存实现（如 `InMemoryStore`）或可冻结时钟，
   而非 `Mock()`；
3. **`Mock()`**：仅用于外部网络边界（LLM API、第三方 SDK）与"验证副作用发生"的场景。

使用 `Mock()` 时：

- 禁止断言 mock 的返回值（`assert result is mock.return_value`）；
- 断言"是否被调用"应服务于验证副作用（如"确实触发了发布"），而非验证中间参数；
- 必须搭配至少一条对**可观察结果**的断言，不能只有 `assert_called`。

## 6. 命名与组织规范

### 6.1 文件命名

- 测试文件与被测模块同构：`test_<module>.py`，放在与被测代码对应的目录树下；
- 目录层级是主类型的唯一归类来源：`tests/unit/` 只放单元测试，`tests/integration/` 只放多组件协作或
  真实适配器兼容性测试，`tests/e2e/` 只放从公开入口走到可观察终态的全链路测试；
- `real_infra`、`live_llm`、`slow` 只描述运行条件，不改变测试所在的主类型目录。

### 6.2 测试命名

- pytest 要求 `test_` 前缀；函数名应描述**被测行为 + 触发条件/期望结果**：
  `test_<行为>_<条件/结果>` 或 `test_should_<行为>_when_<条件>`；
- 命名必须与断言一致。审计发现 `test_multiple_filters`（实际只回读字段）、
  `test_fusion_strategy_combines_results`（实际只断言 `len > 0`）等命名与内容脱节的案例——命名承诺了什么，
  断言就必须验证什么，否则应改名或补断言。

### 6.3 docstring

- 每个测试类/文件用一行 docstring 说明被测对象与范围；
- 禁止从其他文件复制 docstring 导致与实际内容不符（审计发现 `system/application/` 下 5 个文件 docstring 全部
  复制自同一模板、`core/mtp/` 3 个文件 docstring 互相雷同）。

## 7. fixture 与隔离规范

1. **作用域最小化**：默认 `function` 作用域；只有初始化昂贵且无状态泄漏时才用 `session`/`module`。
2. **必须清理**：fixture 修改了全局状态（i18n 语言、contextvars、全局注册表、单例）时，必须用 `try/finally`
   或 teardown 恢复，防止污染后续测试。
3. **禁止死代码**：fixture 中构造了但从未被任何测试引用的对象、helper、import 必须删除；不要用
   `_ = unused` 之类 hack 掩盖死参数。
4. **禁止真实时钟与固定 sleep**：需要时间相关行为时注入可控时钟（monkeypatch 或参数注入）；
   需要等待异步落定时用事件/轮询 + 超时，不得 `time.sleep(x)` 后直接断言终态。
5. **临时文件用 `tmp_path`**：禁止在项目工作目录创建 `.test_tmp` 之类真实目录。
6. **禁止 `sys.path.insert` hack**：依赖根 `conftest.py` 与 `pyproject.toml` 的 `pythonpath` 配置。

## 8. 覆盖率要求

- CI 门禁为 `--cov-fail-under=85`，覆盖 `tests/unit` 与 `tests/integration`；门禁命令见
  [.github/workflows/ci.yml](../../../.github/workflows/ci.yml)，覆盖率源码范围配置见 [pyproject.toml](../../../pyproject.toml)。
- 覆盖率是**下限而非质量上限**：高覆盖率不保证断言有效。本规范第 4 节的断言规范优先级高于覆盖率数字。
- 新增代码应优先补齐关键分支（状态机迁移、错误路径、边界值），而不是用恒真断言刷覆盖率。
- 已知低覆盖模块（`gateway/commands/parser.py`、`infrastructure/llm/litellm_service.py`、`server/__main__.py`）
  的补测属于存量工作，见审计文档的 COVERAGE_GAP 条目。

## 9. 评审清单（新测试 PR 必查）

新增或修改测试时，逐条确认：

- [ ] 这条测试改坏生产代码时会变红？（能失败）
- [ ] 断言的是可观察行为结果，而非 mock 调用细节或私有属性？
- [ ] 本次声称要验证的对象与协作边界真实，fake/mock 只替换边界之外的依赖？
- [ ] 无恒真/同义反复断言（A1）、无写死逻辑（A2）、无 mock 镜像（A3）？
- [ ] 无 `is not None` / `len > 0` / `>= 0` 之类宽松断言（A5）？
- [ ] 无 `if ...: assert ... else: log` 条件守卫（A6）？测试体存在真实 assert（非 A7）？
- [ ] 异常断言收敛到具体异常类型？
- [ ] 无真实时钟/sleep，无全局状态未清理？
- [ ] 恰好归属于一个主类型，目录与主类型一致，正交运行标签与依赖条件匹配？
- [ ] 命名与断言一致，docstring 准确？
- [ ] 无死代码（未用 import、变量、fixture、helper）？
- [ ] 一个测试只验证一件事？

## 10. 升级为独立 Plan 的门槛

以下机制建设在满足条件后，从本文提取为独立 Plan：

1. **断言质量静态检查**：需要一个 lint 规则或脚本，能识别 A1-A8 反模式（如 `assert x == x`、`is not None`
   恒真断言）。在无自动检查前，由 code review 承担。
2. **mutation testing**：引入 `mutmut` 等工具，用"改坏代码测试能否抓住"来持续校验断言有效性；
   需先清理存量恒真测试，否则噪音过大。
3. **FLAKY 时钟注入改造**：统一为 `perception`、`scheduler`、`vitality` 等时间敏感模块注入可控时钟，
   消除 `time.sleep`/真实 `datetime.now()` 依赖。
4. **主类型 marker 自动化**：在收集阶段按 `tests/unit`、`tests/integration`、`tests/e2e` 自动补充主类型 marker，
   并校验每个测试恰好属于一个主类型、`unit` 不携带 `real_infra`/`live_llm`；
5. **e2e/真实资源独立 CI**：为确定性 `e2e`、`real_infra` 和 `live_llm` 分别建立合适的手动或 nightly 任务，
   使依赖真实资源的测试可被明确选择和实际运行，避免“收集了但长期被排除”。

## 11. 相关文档

- [测试体系问题清单](../../todo/testing-suite-audit.md)：存量的逐条问题与修复计划（P0-P3）；
- [pyproject.toml](../../../pyproject.toml)：pytest、marker 与覆盖率源码范围配置；
- [.github/workflows/ci.yml](../../../.github/workflows/ci.yml)：CI 测试执行方式。
