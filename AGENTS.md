# HiveMemory Coding Agent 规范

本文件是仓库根目录的 Coding Agent 工作规范：工作原则、所有权摘要、最容易被违反的约束与验证要求，并指向改动前需要先读的当前设计文档。详细设计以 `docs/` 中的当前文档、代码、测试和配置为准。

## 1. 工作原则

- 先理解边界，再修改实现。任何新行为都要先找到负责它的 System、子系统、公共契约和测试入口。
- 采用最小、可审查的改动。不要顺手重命名、格式化或迁移无关文件，也不要覆盖用户已有的未提交修改。
- 当前事实必须有证据。优先核对成熟代码、测试、配置和已完成基线；`docs/ideas/`、`docs/plans/`、`docs/todo/` 与 `docs/ROADMAP.md` 中的内容不能证明功能已经实现，`docs/research/` 记录的外部能力也不能证明 HiveMemory 已实现对应功能。
- 不猜测公共行为。路由、事件、错误、版本和配置字段必须从代码或当前契约文档确认。
- 不把“测试通过”写成未执行的事实；最终说明实际运行过的命令及其结果。
- 默认使用中文沟通和提交说明；代码、公共标识符、日志和错误信息遵循现有语言与命名。

### 1.1 控制复杂度

代码要由 owner 长期读懂和维护。每个新概念的理解成本由之后的每一位读者承担，所以默认不新增概念，或需要严格控制新概念的加入。这与 [ADR-0002](docs/architecture/decisions/0002-unique-identities-and-minimal-concurrency.md) 的按需原则一致。

- 新的类型、端口（Protocol）、DTO、状态、层或配置项，需要已经存在的生产使用方；测试不算生产使用方。为尚未出现的第二个实现或调用方预留抽象，只在 owner 明确要求时进行。
- 生产中不会出现的输入不写处理分支。罕见或尚未出现的边界情况，优先以结构化错误显式拒绝，等真实出现再处理；不要用兜底、重试或兼容分支掩盖它们。
- 方案先给出能满足需求的最简单版本，写明它暂不处理哪些情况；更完整的做法作为可选项说明，由 owner 选择。
- 重构只服务于当前任务，只调整任务需要触及的结构。没有生产使用方的代码、同一事实的重复表达、读不懂的结构，在交付说明中列出或登记为 Todo，不擅自扩大改动范围。
- 第 3、4 节的所有权与不变量在实现时必须遵守，但不排斥简化：发现某个概念可以删除或合并时应当提出，并说明它原本保护的行为改由什么保证，由 owner 决定。

### 1.2 可读性与代码风格

人类可读、可理解、可维护，优先于写法上的巧妙、未经测量的性能优化和形式上的完备。判断标准是：owner 不借助 Agent，能读懂这段代码做什么、为什么这样做，并且有把握修改它。

- 先读后写。动手前读要修改的模块和相邻代码，沿用其中的命名、结构、错误处理、日志、类型标注与注释写法；同类问题已有写法时照着写，不在仓库里引入第二种风格。
- 命名使用项目已有的领域术语，说明它是什么；同一概念不另起名字，不同概念不共用名字。
- 控制流保持直白：优先平铺的顺序代码与提前返回，避免深层嵌套、用异常承担常规分支、只转发一次的薄包装和多余的间接层。抽取函数是为了让调用处更好读，不是为了减少行数。
- 函数只做一件事，长度以一屏能读完为宜；正确性依赖的前提（例如两步之间不能有 `await`）尽量由结构保证，做不到时在代码旁用一句注释写明。
- 注释与 docstring 写行为、不变量和理由，篇幅与逻辑的复杂度相称，不逐行复述代码，也不大段讲述相邻模块的职责（其余要求见第 5 节）。
- 格式只以 ruff、black 与前端 lint 配置为准。现有写法明显妨碍阅读时，不在当前任务中另起一套风格，在交付说明中提出。

## 2. 开始任务前

1. 运行 `git status --short`，记录并保留用户的已有修改。
2. 阅读与任务相关的 `README.md`、`docs/PROJECT.md`、当前架构/契约文档和最近的测试；文档治理规则见 [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md)。
3. 搜索真实入口：优先使用 `rg` 和 `rg --files`，不要仅凭文件名或旧归档推断实现。
4. 明确任务属于代码、测试、配置、前端、文档还是多项联动，并列出要验证的行为。
5. 如果发现设计仍在变动，先放入对应的 Plan/Idea/Todo；不要把未收尾的设计写进当前事实文档。

## 3. 当前架构与所有权

当前组合根是 `src/hivememory/system/` 中的 `HiveMemorySystem`（装配、门面与生命周期）。HTTP 路由与 MCP 入口（`mcp_server`）是入口适配层，经门面取得服务调用，不应重新实现业务流程：管理能力由 `workspace.capability` 的 root API 提供（actor API 只经操作入口，不经门面；MCP 入口只用门面交出的操作入口的请求级提交），chat 任务进程的进程表与四阶段编排骨架位于 `workspace.process`，Import Bus 的应用服务与就绪检查位于 `system.application`。

包按层组织，依赖只能指向同层或更低层（由 `tests/unit/architecture/test_package_layers.py` 守护）：

| 层 | 包 | 职责 |
| --- | --- | --- |
| L0 | `core` / `config` / `utils` / `i18n` | 依赖中立的模型、Route 泛型原语、事件常量、错误类型、访问值类型与端口协议；配置段模型 |
| L1 | `components` | 进程内运行时机制：总线、调度器、work queue、运行时事件、串行门、trace context、取消收尾 |
| L2 | `engines` / `infrastructure` / `prompts` | 算法与外部技术适配器 |
| L3 | `workspace` / `patchouli` / `gateway` / `alice` + `agent_runtime` | 子系统；彼此只能导入对方公开的 `contracts` 子包 |
| L4 | `system` | 组合根、门面与系统级能力；除入口外不被任何包导入 |
| L5 | `server` / `mcp_server` | 入口（传输层 adapter）；`mcp_server` 是外部 harness 经 MCP 访问记忆的操作适配器 |

根配置与加载（`config.app`）只供 `system` 与 `server` 使用；下层以端口协议声明对上层能力的需要（如 `core.access.PrincipalAuthenticator`、`agent_runtime.ModelResolver`），由组合根实现并注入。engines 对上层的少量既有导入登记为已知例外，不得新增。

| 边界 | 负责 | 不负责 |
| --- | --- | --- |
| System | 组合根与门面、配置加载、生命周期、模型/Provider 注册表、接入登记与 Principal authentication、Import Bus、就绪检查 | 记忆算法、Gateway 分析、Agent loop、MTP 具体执行、Workspace 准入与行为授权、运行时机制实现 |
| Workspace | 认证入口与 Workspace 准入、逐次行为授权、能力层（root API 与 actor API）、会话资源、写入意图登记与读取视图、WorkspaceAsset working set 与上传/解析交接、任务进程（进程表、四阶段编排、CPU 分配与提示词编译、封口交互记录） | 记忆算法与 canonical 存储、接入登记、Agent loop |
| Gateway | 入口拦截、命令解析、话题/查询分析、检索计划和保守降级 | 命令执行、记忆存储、检索执行、回复生成、Interaction 提交 |
| Patchouli | Memory/Topic/Profile、检索、感知、生成、生命周期、prepare（话题与检索）/finalize 和长期状态 | 顶层 chat 编排、入口命令、Agent 生成循环、附件租借与执行上下文编译 |
| Alice | Agent run、frame、MTP/工具和 CALL 编排 | 长期记忆所有权、Gateway 分析、chat 任务进程编排、HTTP 生命周期 |
| Core/Contracts | 依赖中立的模型、协议枚举、Route 泛型原语、事件常量、访问值类型与端口协议 | 业务编排、I/O、可变运行时状态 |
| Components | 进程内运行时机制（总线、调度器、work queue、运行时事件、串行门、trace context、取消收尾） | 业务状态与业务判断 |

必须保持以下方向：`server -> 经门面取得的服务（workspace 能力层与任务进程编排 / System Import Bus） -> Global public routes -> 子系统`；`mcp_server -> 门面取得的操作入口（请求级提交） -> actor API`。route 名称与签名由提供方 `contracts` 声明，调用方直接导入提供方的类型化 route；core 只提供 `Route` / `route` 原语，事件常量仍在 core。跨子系统使用公共 route、公共模型或全局事件；不要持有对方 Runtime、Service、Controller、存储客户端或 local bus。

关键所有权约束：

- Patchouli 是 Memory、Topic、Artifact、Interaction 和记忆任务的权威所有者；Topic、Memory、Artifact 的 Workspace 归属由 Patchouli 领域规则校验。
- Alice 只拥有本次 Agent run 的 frame、turn events、工具调用与 ACK 别名清单；写入意图登记、完整引用解析、引用记录副作用以及可失效的原子/Profile 缓存属于 workspace。Alice 不请求 Patchouli 公开路由。
- System 拥有 Import Bus 的进程内控制状态，以及总线、调度器等共享设施实例的装配与关闭（机制实现在 `components`）；chat 任务进程的控制状态由 workspace 的进程表持有；`WorkspaceAsset` 及其 working set 属于 workspace，由组合根装配并在关闭时最后清理。
- Gateway 只产生 `GatewayDecision`；它可以读取辅助上下文，但不取得记忆所有权。
- RuntimeEvent 只用于 best-effort 观测，不能决定业务成功、替代 RPC 返回值或充当可靠命令。
- Cache、queue、registry、scheduler 和 EventBus 默认是进程级共享基础设施；`workspace_id` 观测标签不等于授权或分区。

Workspace 的访问、能力层与任务进程规则较多。改动这些区域前先读对应的当前设计文档；下面只列出最容易被违反的约束：

- **能力层**（[能力层](docs/workspace/capability.md)）：三个调用面互不借用。actor API 的六项 actor 操作与 MTP 的 READ、SEARCH、RUN、WRITE、UPDATE、CALL 一一对应，逐次按访问登记白名单授权；root API 只接受保留的 `system`，HTTP 管理路由只调用它；任务进程的阶段调用只校验准入与目标，只在 `workspace.process` 中发起。两个 API 目录互不导入，不要新增可登记给 actor 的管理 operation。操作入口是 actor API 唯一的调用方：执行线程经线程提交，不建进程的 MCP 入口以请求级 context 经请求级提交 `execute_request`，它只开放检索与读取（[MCP 记忆访问](docs/contracts/mcp.md)）。
- **任务进程与执行线程**（[任务进程](docs/workspace/task-process.md)、[执行线程](docs/workspace/execution-threads.md)）：进程 context 服务阶段调用，线程 context 只服务 Actor 阶段的 actor 操作。`ProcessThreads` 是派生、结束与解析线程的唯一入口，存活只有一个判定（线程未结束且进程仍在运行）。执行者经线程句柄 `OperationSubmitter` 提交 `OperationRequest`；请求只携带操作参数，不暴露访问 context，也不能切换目标 Workspace 或指定 `process_id`。
- **身份数据**（[身份与授权](docs/workspace/identity-and-authorization.md)）：访问 context、`IdentityScope`、`ExecutionLabels` 与操作快照各有形态，不得互相代替。`IdentityScope` 只由授权点组装，只在一次操作的调用链内传递，不写入记录、事件或后台任务；标签与操作快照只用于观测、提示词与执行策略，不参与授权。资源 owner 在公共边界把 scope 拆为资源归属与发起者；资源授权在 owner 处进行，先校验归属等于目标 Workspace，再按资源 policy 判断发起者的可见性。
- **会话**（[会话资源](docs/workspace/sessions.md)）：ConversationSession 属于 Workspace，不属于 Agent；同一会话同时只允许一个任务进程；会话原文不删改，压缩只产生检查点；会话管理只经 root API。
- **已知例外**：`WorkspaceAssetStore`、解析服务与既有资产 Reader/Command 端口仍以旧 `IdentityScope` 做归属检查，System 的 `AssetMaterializationReader` 为此在一次租借内组装 scope。此例外仅限该适配器，不得扩散，后续拆分见 [WorkspaceAsset 归属拆分 Todo](docs/todo/workspace-asset-ownership-identity-split.md)。

## 4. 关键流程不变量

两条链路的完整控制流见[系统架构概览](docs/architecture/overview.md)第 5、6 节；停止、取消与收尾的细节见[任务进程](docs/workspace/task-process.md)。

- 主动链路：`Gateway process -> Patchouli prepare（话题与检索） -> 任务进程 CPU 分配 -> Actor 执行（经 CPU 端口，当前为 Alice） -> 有输出时封口交互记录并写入会话 -> Patchouli finalize`。prepare 失败、进程异常、finalize 前断连或 CPU 无输出时不进入 finalize；cleanup 只补偿 prepare 新建且仍为空的话题，不是跨边界事务回滚。附件租借随进程关闭释放。
- 停止与取消：用户停止保留部分输出并提交，交付方离开与系统关闭丢弃本轮；进入 FINALIZE 后拒绝停止，提交不受交付方离开影响。停止被接受时全部线程即已结束，CPU 取消后的收尾不得依赖 actor 操作。
- Import Bus 链路：`POST /api/v1/import/turns -> 统一认证网关 + 导入授权 -> 按外部会话串行编排 -> Gateway PASSIVE_MEMORY（只取话题落位） -> 共用封口 seal_interaction -> Patchouli 公开提交路由 -> 等待感知应用完成后确认`。Import Bus 不运行 Alice、MTP、命令或回复生成，不检索记忆，也不返回记忆上下文。
- `MTP WRITE/UPDATE` 的 ACK 表示意图已在 workspace 登记，并已被 Patchouli 生成队列接纳，不表示正式记忆已经持久化；派发开始后，写入不因 CPU 失败、取消、停止、线程结束或进程关闭而撤回。正式持久化只由 Patchouli 生成链结算，不能在 Alice（含 Koakuma）内直接写正式 Memory。详见[读取视图与写入意图](docs/workspace/read-view-and-write-intents.md)。
- Agent 的操作权限只由 Workspace 访问登记的 actor 操作白名单表达，Agent Profile 不承载任何权限。CALL 派生的子 frame 以被调用方身份运行，只允许根 frame 发起，不得递归 CALL。详见 [MTP 契约](docs/contracts/mtp.md)与[执行线程](docs/workspace/execution-threads.md)。
- Gateway 的局部失败只能在仍满足终态不变量时保守降级；投影、终态校验、装配错误和 cancellation 不得被静默吞掉。
- RPC 用于需要确定返回值、失败传播或完成确认的操作；Pub/Sub 只用于发布者不依赖结果的通知。

修改上述边界、所有权、公开 route/event、取消/清理、Workspace 访问或 prepare/finalize 顺序时，必须同步审查 `docs/contracts/`、`docs/architecture/`、相关测试和文档晋升门禁。

## 5. Python 后端规范

- Python `>=3.12`，源码位于 `src/`，配置和依赖入口是 `pyproject.toml`。
- Black 行宽 100；Ruff 使用 `E,F,I,N,W,UP`，忽略 `E501`；新代码保持类型标注和清晰的公共模型边界。
- 优先使用现有 Pydantic/domain model、route constants、错误类型和 service；不要为同一概念创建第二个 DTO、状态机或配置来源。
- 异步代码必须正确传播 `asyncio.CancelledError`，给外部调用设置已有的 timeout/deadline，并在 service/runtime 关闭时释放自己创建的任务和资源。
- 时间敏感逻辑注入可控时钟或使用相对时间；测试中不得通过固定 `sleep` 等待状态落定。
- 错误要保留结构化类型、阶段和可诊断上下文；不要用宽泛 `except Exception` 把程序错误改成成功或空结果。
- 环境变量用于密钥、地址、端口和运行开关；业务参数使用 `configs/config.yaml`，模型清单使用 `configs/models.yaml`。绝不提交真实密钥、`.env`、Qdrant 数据或日志。
- 配置模型位于 `src/hivememory/config/`（按子系统分段）；组件只接收并导入自己的配置段，不持有根配置 `HiveMemoryConfig`，也不自行调用 `load_app_config`，缺配置时显式失败。
- 版本唯一来源是 `src/hivememory/_version.py`；涉及版本时运行 `python scripts/check_version_consistency.py`。
- 改动公共模型、route 名称、事件名、配置键或错误语义时，先更新对应契约和所有消费者，再运行跨边界测试。
- 代码注释、docstring 与字段 `description`（含测试与前端代码）直接写出行为、不变量与理由，不指向任何文档：不写 `docs/` 路径、文档章节号与编号条目，不写计划与批次代号、决定编号或依赖计划进度的“暂时/下一批”，迁移来源与版本历史交给版本控制；代码目录不放 README。引用只能从文档指向代码；规则见[文档治理规范](docs/DOCUMENTATION.md)第 11.2 节，由 `tests/unit/architecture/test_self_contained_comments.py` 守护。

## 6. 测试与验证

测试的分类、归属、断言、mock、命名与隔离规则以[测试设计规范](docs/governance/testing/test-design-standards.md)为准，新增、修改或删除测试前先读它。下面只列出最常被违反的几条：

- 测试主类型由目录决定（`tests/unit/`、`tests/integration/`、`tests/e2e/`）；`e2e`、`live_llm`、`slow` 只是运行条件，不用 marker 掩盖错误的归类。
- 断言可观察行为，改坏生产代码时测试必须会失败；不测私有字段、实现顺序或 mock 的调用细节。
- 每条不变量只在做出判断的最低可测层写权威测试；新增前先检索现有测试，已有覆盖时扩展原测试，不另建文件。
- 测试针对可能出现的生产缺陷：不为纯重命名、低风险可逆改动，或 Python、mypy、架构测试已经保证的性质写测试。
- 契约有意变更时，在同一改动中删除或改写旧契约的测试，不写墓碑测试。

常用验证命令：

```bash
# 安装开发依赖
python -m pip install -e ".[dev]"

# 定向测试（优先从受影响目录开始）
pytest tests/unit/<area> -q
pytest tests/integration/<area> -q

# 默认本地测试：按 pyproject.toml 排除 e2e/live_llm/slow
pytest

# CI 等价后端门槛
pytest tests/unit tests/integration -v --tb=short \
  -m "not live_llm and not e2e and not slow" -n auto \
  --cov=hivememory --cov-report=term-missing --cov-fail-under=85

# 静态检查（项目配置已写入 pyproject.toml）
ruff check .
black --check .
uv sync --frozen --extra dev
uv run --no-sync mypy --warn-unused-ignores src scripts/typecheck_bus_routes.py

# 前端变更
cd frontend && npm ci && npm run lint && npm run build
```

需要真实 Qdrant/LLM 的测试只在明确需要时运行：CI 中 E2E 是手动 workflow；本地可以使用 Docker 或 `scripts/hivememory-dev.sh` 管理 Qdrant、后端和 Vite。测试结论必须说明外部依赖是否可用。

## 7. 文档与设计变更

文档的分类、状态、晋升门禁、计划与归档规则以[文档治理规范](docs/DOCUMENTATION.md)为准，新增、修改、移动或删除文档前先读它。下面只列出最常被违反的几条：

- 当前事实文档只写已经落地的内容。开发分支最终收尾之前，候选模型、目标接口、迁移步骤和“未来将支持”的设计只留在 Idea、Plan 或 Todo 中，也不用“计划中”“尚未实现”之类的措辞提前写进事实文档。
- 当前设计文档既写可执行事实，也保留仍然有效的问题背景、所有权理由、关键取舍与失败语义；不要把代码目录逐文件翻译成文档。
- 同一概念只在一处维护，其他文档链接过去，不复制内容。
- 设计讨论只把阻塞下一份计划（见 [Plans 导航](docs/plans/README.md)）的问题写进 Idea，写明背景、选项与影响；不阻塞的发现只记一行搁置项，或在对话中提出，owner 要求时再展开。一个主题只设一个问题，不拆成子问题。owner 决定后问题即关闭，后续细节作为候选设计写入 Idea 并给出合理默认值，注明不是 owner 决定、可在计划评审时调整；回答问题不应让未决问题变多。
- 新增或修改文档后，检查相对链接、状态、owner、`last_reviewed` 与相关代码入口，并运行 `git diff --check`。

## 8. 完成任务前的自检

- [ ] 改动范围只覆盖任务所需文件，未覆盖用户已有修改。
- [ ] 所有跨边界调用遵循公共 route/model/event 和唯一状态所有者；新增导入符合包分层规则。
- [ ] Workspace 访问与身份数据（访问 context、`IdentityScope`、资源归属与发起者）、取消、超时、失败、幂等和资源清理语义已核对。
- [ ] 新增的类型、端口、DTO、状态或配置项都有生产使用方；没有为生产中不会出现的输入写处理分支。
- [ ] 改动的代码不借助 Agent 也能读懂；命名、结构、错误处理与注释沿用了周边代码的写法。
- [ ] 新测试属于正确主类型，断言可在生产代码改坏时失败。
- [ ] 新测试位于不变量的归属层，没有在其他层或文件重复已有覆盖；契约变更已同步删除或改写旧契约测试。
- [ ] 已运行与改动匹配的定向测试、静态检查和必要的全量门槛。
- [ ] 文档只陈述有证据的当前事实，计划仍留在正确的工作文档中。
- [ ] 新写或改动的注释与 docstring 不引用文档、计划代号或决定编号，读者不打开文档也能理解。
- [ ] `git diff --check` 和最终 `git diff` 已审阅。

交付说明应简洁列出：改了什么、为什么、新增与删除了哪些概念、运行了哪些验证、哪些检查因环境或外部依赖未运行、任何仍需关注的风险，以及发现但没有处理的简化点。
