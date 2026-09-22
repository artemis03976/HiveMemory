# HiveMemory Coding Agent 规范

本文件是仓库根目录的 Coding Agent 工作规范。它把项目当前的架构、测试和文档治理规则压缩成执行清单；详细设计仍以 `docs/` 中的当前文档、代码、测试和配置为准。

## 1. 工作原则

- 先理解边界，再修改实现。任何新行为都要先找到负责它的 System、子系统、公共契约和测试入口。
- 采用最小、可审查的改动。不要顺手重命名、格式化或迁移无关文件，也不要覆盖用户已有的未提交修改。
- 当前事实必须有证据。优先核对成熟代码、测试、配置和已完成基线；`docs/ideas/`、`docs/plans/`、`docs/todo/` 与 `docs/ROADMAP.md` 中的内容不能证明功能已经实现。
- 不猜测公共行为。路由、事件、错误、版本和配置字段必须从代码或当前契约文档确认。
- 不把“测试通过”写成未执行的事实；最终说明实际运行过的命令及其结果。
- 默认使用中文沟通和提交说明；代码、公共标识符、日志和错误信息遵循现有语言与命名。

## 2. 开始任务前

1. 运行 `git status --short`，记录并保留用户的已有修改。
2. 阅读与任务相关的 `README.md`、`docs/PROJECT.md`、当前架构/契约文档和最近的测试；文档治理规则见 [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md)。
3. 搜索真实入口：优先使用 `rg` 和 `rg --files`，不要仅凭文件名或旧归档推断实现。
4. 明确任务属于代码、测试、配置、前端、文档还是多项联动，并列出要验证的行为。
5. 如果发现设计仍在变动，先放入对应的 Plan/Idea/Todo；不要把未收尾的设计写进当前事实文档。

## 3. 当前架构与所有权

当前组合根是 `src/hivememory/system/` 中的 `HiveMemorySystem`。HTTP 路由是适配层，应调用 System application service，不应重新实现业务流程。

| 边界 | 负责 | 不负责 |
| --- | --- | --- |
| System | 组合、应用用例、生命周期、全局总线、运行控制、Passive Ingress、调度、注册表和接入认证 | 记忆算法、Gateway 分析、Agent loop、MTP 具体执行 |
| Gateway | 入口拦截、命令、话题/查询分析、检索计划和保守降级 | 记忆存储、检索执行、回复生成、Interaction 提交 |
| Patchouli | Memory/Topic/Profile、检索、感知、生成、生命周期、prepare/finalize 和长期状态 | 顶层 chat 编排、入口命令、Agent 生成循环 |
| Alice | Agent run、frame、MTP/工具、PendingAtom 运行时和 CALL 编排 | 长期记忆所有权、Gateway 分析、HTTP 生命周期 |
| Core/Contracts | 依赖中立的模型、协议枚举、route/event 常量 | 业务编排、I/O、可变运行时状态 |

必须保持以下方向：`server -> System application -> Global public routes -> 子系统`。跨子系统使用公共 route、公共模型或全局事件；不要持有对方 Runtime、Service、Controller、存储客户端或 local bus。

关键所有权约束：

- Patchouli 是 Memory、Topic、Artifact、Interaction 和记忆任务的权威所有者。
- Alice 只拥有本次 Agent run 的 frame、turn events、工具调用、alias/cache 和 PendingAtom 运行时视图。
- System 拥有 chat/passive 控制状态、全局调度与 `WorkspaceAssetStore` 的进程内 working set。
- Gateway 只产生 `GatewayDecision`；它可以读取辅助上下文，但不取得记忆所有权。
- RuntimeEvent 只用于 best-effort 观测，不能决定业务成功、替代 RPC 返回值或充当可靠命令。
- `IdentityScope`（Actor + Workspace）必须沿应用服务、公共 route、Interaction 和后台任务传播，并在资源 owner 处再次校验。
- Cache、queue、registry、scheduler 和 EventBus 默认是进程级共享基础设施；`workspace_id` 观测标签不等于授权或分区。
- `WorkspaceAsset` 是 System 的进程内资源；Topic、Memory、Artifact 的 Workspace 归属仍由 Patchouli 领域规则校验。

## 4. 关键流程不变量

- 主动链路：`Gateway process -> Patchouli prepare -> Alice run -> (仅 completed) Patchouli finalize`。
- 被动链路：`PassiveIngressService -> Gateway PASSIVE_MEMORY -> buffer/seal -> InteractionSubmissionQueue -> Patchouli perception`；被动模式不运行 Alice、MTP、命令或回复生成。
- prepare 失败或 Agent 取消/失败时，不默认进入 finalize；System 可请求 Patchouli cleanup，但 cleanup 只补偿 prepare 新建且仍为空的临时话题，不是跨边界事务回滚。
- `MTP WRITE/UPDATE` 的 ACK 只表示 PendingAtom 已登记；正式持久化由 Patchouli 后续结算，不能在 Koakuma/Alice 内直接写正式 Memory。
- MTP 权限由 Agent Profile 的允许 verb/tool 控制；CALL 只允许根 frame 发起，子 frame 不得递归 CALL。
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
- 版本唯一来源是 `src/hivememory/_version.py`；涉及版本时运行 `python scripts/check_version_consistency.py`。
- 改动公共模型、route 名称、事件名、配置键或错误语义时，先更新对应契约和所有消费者，再运行跨边界测试。

## 6. 测试设计与选择

测试主类型由目录决定：`tests/unit/` 验证快速隔离行为，`tests/integration/` 验证多组件/真实适配器协作，`tests/e2e/` 验证公开入口到可观察终态的链路。`e2e`、`live_llm`、`slow` 是运行条件，不要用 marker 掩盖错误的目录归类。

- 测试行为和可观察结果，不测试私有字段、实现顺序或仅仅“调用过 mock”。
- 依赖不触发网络或慢 I/O 时优先用真实对象；外部 LLM、SDK、网络边界才使用 mock，必要时使用手写 fake/in-memory store。
- 禁止恒真断言、复制生产公式、mock 返回值镜像、`is not None`/`len > 0`/`>= 0` 宽松断言、条件守卫、无断言测试和 `pytest.raises(Exception)`。
- 异常断言收敛到具体异常类型，必要时使用 `match=`；浮点比较使用 `pytest.approx`。
- fixture 默认 function scope；修改全局注册表、contextvar、语言或单例时必须恢复；临时文件使用 `tmp_path`。
- 一个测试验证一个明确行为；测试命名描述触发条件与结果，docstring 必须与真实范围一致。
- 新行为应覆盖成功路径、关键边界、取消/失败语义、权限/Workspace 隔离和必要的重试/幂等行为；不要为纯重命名或低风险可逆改动堆测试。

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
mypy src

# 前端变更
cd frontend && npm ci && npm run lint && npm run build
```

需要真实 Qdrant/LLM 的测试只在明确需要时运行：CI 中 E2E 是手动 workflow；本地可以使用 Docker 或 `scripts/hivememory-dev.sh` 管理 Qdrant、后端和 Vite。测试结论必须说明外部依赖是否可用。

## 7. 文档与设计变更

- 当前事实入口：`docs/PROJECT.md`、`docs/architecture/`、`docs/system/`、`docs/gateway/`、`docs/patchouli/`、`docs/alice/`、`docs/contracts/`、`docs/help/` 和 `docs/governance/`。
- `docs/ideas/` 是探索，`docs/plans/` 是绑定版本/里程碑且可验收的实施计划，`docs/todo/` 是小范围缺陷/技术债，`docs/ROADMAP.md` 是明确标注状态的规划摘要，`docs/archive/` 只保存历史。
- 开发分支尚未最终收尾时，不把候选模型、未稳定命名、目标接口、迁移步骤或“未来将支持”的设计写入当前事实文档。实现、测试、迁移、验收和代码审查完成后，才按最终代码晋升事实并归档 Plan。
- 当前设计文档既要写可执行事实，也要保留仍然有效的问题背景、所有权理由、关键取舍、不变量和失败语义；不要把代码目录逐文件翻译成文档。
- 新增/修改文档后检查相对链接、状态、owner、last_reviewed、相关代码入口和重复真相源；运行 `git diff --check`。
- 更新根 README、当前架构、契约或本文件时，确认没有把计划内容误写成现状，也没有覆盖用户已有的文档修改。

## 8. 完成任务前的自检

- [ ] 改动范围只覆盖任务所需文件，未覆盖用户已有修改。
- [ ] 所有跨边界调用遵循公共 route/model/event 和唯一状态所有者。
- [ ] Workspace/IdentityScope、取消、超时、失败、幂等和资源清理语义已核对。
- [ ] 新测试属于正确主类型，断言可在生产代码改坏时失败。
- [ ] 已运行与改动匹配的定向测试、静态检查和必要的全量门槛。
- [ ] 文档只陈述有证据的当前事实，计划仍留在正确的工作文档中。
- [ ] `git diff --check` 和最终 `git diff` 已审阅。

交付说明应简洁列出：改了什么、为什么、运行了哪些验证、哪些检查因环境或外部依赖未运行，以及任何仍需关注的风险。
