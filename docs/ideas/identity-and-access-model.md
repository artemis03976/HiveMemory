---
title: 身份与访问体系
status: idea
horizon: candidate
owner: project
scope: identity-and-access-mainline-from-principal-proof-to-workspace-admission-operation-authorization-and-resource-visibility
code_paths:
  - src/hivememory/core/models/identity.py
  - src/hivememory/core/access.py
  - src/hivememory/core/memory_access.py
  - src/hivememory/system/access/
  - src/hivememory/config/access.py
  - src/hivememory/workspace/authentication.py
  - src/hivememory/workspace/authorization.py
  - src/hivememory/workspace/registry.py
  - src/hivememory/workspace/process/threads.py
  - src/hivememory/workspace/capability/operations.py
  - src/hivememory/server/deps.py
  - src/hivememory/server/routers/
related_docs:
  - docs/workspace/identity-and-authorization.md
  - docs/workspace/authentication.md
  - docs/workspace/capability.md
  - docs/workspace/execution-threads.md
  - docs/workspace/task-process.md
  - docs/governance/security/identity-and-execution-safety.md
  - docs/governance/baselines/identity-access-current-state-2026-11-05.md
  - docs/ideas/fifth-generation-architecture.md
  - docs/ideas/external-actor-registration-and-runtime-access.md
  - docs/ideas/pending-intent-migration.md
  - docs/ideas/execution-environment.md
  - docs/ideas/ae2-hivememory-architecture-analogy.md
  - docs/archive/ideas/execution-thread-definition.md
  - docs/archive/todo/local-http-management-access.md
  - docs/archive/plans/http-management-credential.md
  - docs/todo/workspace-asset-ownership-identity-split.md
  - docs/todo/memory-visibility-policy-ui.md
  - docs/todo/frontend-identity-ownership.md
  - docs/todo/run-target-executability-check.md
last_reviewed: 2026-11-05
---

# 身份与访问体系

## 0. 文档性质

- **主线定位**（owner，2026-11-05）：身份与访问体系作为单独的主线方向维护。它是 Workspace 体系的基础，不只服务 controller 模式，覆盖从 principal 的登记与调用方证明，到进入 Workspace 的身份、操作调用的授权，再到记忆等资源的可见性。这条方向最好在发布前为系统打好地基，否则 controller 与 plugin 两种接入都难以继续。
- 本文收集这条线上尚未决定或尚未实施的部分，不描述当前行为，也不代表排期。理念见[第五代架构 Idea](./fifth-generation-architecture.md)第 3.5、3.6 节；当前的身份模型见[身份与授权](../workspace/identity-and-authorization.md)与[认证与访问登记](../workspace/authentication.md)，三个调用面见[能力层](../workspace/capability.md)。
- **2026-11-05 重写**：原文只覆盖跨 Workspace 方向。散落在[第五代架构 Idea](./fifth-generation-architecture.md)第 5 节（P-1、P-4a、P-5、P-7、P-8）、[外部 Actor Idea](./external-actor-registration-and-runtime-access.md)（E-1）与[写入意图后续方向 Idea](./pending-intent-migration.md)第 5 节的身份问题移入本文，沿用原编号，原位置改为链接；本文不新增问题编号。重写前的版本见 commit `3cfdec9d` 中的本文；2026-10-13 的精简删除了已实施的 I-1–I-11，删除前的版本见 commit `a384688`。
- 问题只列选项及其影响，选项顺序不代表倾向；标为“分析”的内容不是 owner 决定。

## 1. 为什么单独成线

身份与访问自 v0.7.0 起经过多轮返工与调整，每一轮都挂在当时的主线方向下进行：[A1 访问边界返工](../archive/plans/v0.7.0-a1-access-boundary-rework.md)、[身份与访问体系第二批](../archive/plans/v0.7.0-identity-access-batch-2.md)、[能力层分层](../archive/plans/v0.7.0-capability-surface-split.md)、[执行线程层](../archive/plans/v0.7.0-execution-thread-layer.md)、[记忆访问](../archive/plans/v0.7.1-memory-access-mcp.md)中 P-1c 与 P-9c 的复核，以及 [plugin 模式开放写入与更新](../archive/plans/v0.7.2-plugin-write-update.md)时的请求级授权（均为归档计划）。结果是：

- 已落地的模型分布在多份事实文档中：身份与授权、认证与访问登记、能力层、执行线程与 MCP 记忆访问；
- 未决的部分散落在第五代架构 Idea、外部 Actor Idea、写入意图 Idea、五份 Todo 与身份治理中；
- 例外与补丁逐渐累积：WorkspaceAsset 仍以旧 `IdentityScope` 做归属检查（AGENTS.md 登记的唯一例外），Profile 注入脚本以 `system` 进入 root API，HTTP 管理接口沿用本地单用户部署的信任假设（2026-11-05 起以管理凭据防误用）。

已经具备、本文不再讨论的地基：三个身份模型与两阶段认证、两阶段授权；密封的访问 context 及进程、线程、请求三种运行绑定（owner 2026-11-05 决定改为统一身份授权入口，候选设计见第 5 节）；资源归属与发起者的拆分；root API、actor API 与阶段调用三个调用面；外部 principal 以本地共享密钥证明身份（P-1c）；`allow_system_actor` 限定能声明 `system` 的来源；不建进程的访问每次操作一份请求级 context（P-9c；统一身份授权入口保留其目标、改变其实现，见第 5 节）。

**对发布的影响**（分析）：owner 定下的 v1.0 范围包括桌面客户端，以及 Claude Code、dsh、Codex 与 pi 四个 harness 的 plugin 模式接入（owner，2026-11-04）。桌面端常驻后端之后，本机进程、各 harness 中的 agent 与前端都是调用方，本地单用户的信任假设与只能静态修改的登记都会被直接触及。

## 2. 全链路与缺口

| 环节 | 已有（当前事实） | 缺口与去向 |
|:---|:---|:---|
| Principal 登记 | 两份配置文件在启动时装载（[认证与访问登记](../workspace/authentication.md#2-两类访问登记)第 2 节） | 运行时登记、修改与撤销：第 3 节 |
| 调用方证明 | 统一认证网关；外部 principal 以共享密钥证明；任务进程注册、Import Bus 与 MCP 入口经网关；HTTP 管理面以管理凭据防误用（[认证与访问登记](../workspace/authentication.md#1-两阶段认证)第 1 节） | HTTP 管理入口的防刻意随桌面客户端；controller 模式下请求与进程的归属校验：第 4 节 |
| 访问 context 与操作授权 | context 只含 actor、驻留 Workspace、principal 与运行绑定；白名单逐次授权；阶段授权；root 只认 `system` | 统一身份授权入口（方向已由 owner 决定，候选设计待细化）；工作 actor 是否带 adapter、本次运行的操作上限、进程控制的授权主体：第 5 节 |
| 资源可见性 | 只有 MemoryAtom 带资源 policy，且只表达读取；其余资源只按归属授权 | 修改与执行授权、写入意图的可见性：第 6 节 |
| 进入其他 Workspace | 第 3 阶段只接受目标等于驻留 Workspace | 受限穿透访问：第 7 节 |
| 例外、暂缓与已知限制 | — | 第 8 节 |

## 3. Principal 登记的来源与生效时点（E-1）

2026-11-05 由[外部 Actor Idea](./external-actor-registration-and-runtime-access.md) 移入。

**背景**：System 接入登记（`configs/system_principals.yaml`）与 Workspace Actor 访问登记（`configs/workspace_actors.yaml`）只在启动时装载，运行中不可变；除配置文件外，没有登记、修改或撤销接入的入口（[认证与访问登记](../workspace/authentication.md#8-已知限制)第 8 节）。

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 维持启动时从配置装载 | 新增或撤销外部 Actor 需要修改配置并重启 |
| B | 提供运行时登记入口（例如经管理员直接通道） | 需要定义登记、修改、撤销的授权与生效时点，以及已签发 context 的处理 |
| C | 登记数据来自外部身份系统 | 需要定义与外部系统的信任关系与同步方式 |
| D | 其他 | —— |

**owner 决定**（2026-09-27）：接入登记维持启动时从配置装载（选项 A）。运行时登记与其他配置文件的热更新一并由未来单独的计划实现。

- **补充**（owner，2026-09-30）：harness 登记的执行侧面同样从配置装载。登记使用单独的配置文件（owner，2026-10-01），2026-10-02 起 `principals` 与 `workspace_actors` 各用一个文件。
- **并入**（2026-10-10）：Workspace 访问登记中 Agent 操作白名单的运行时编辑（含前端开关）并入同一份未来计划（[ROADMAP](../ROADMAP.md) 第 5.5 节“配置热更新”）。届时需要处理线程描述中的操作快照与即时检查之间的差异。

**E-1a**：System 接入登记与 Workspace Actor 访问登记是否经同一入口维护：同一入口 / 分开维护 / 其他。owner 决定随运行时登记的计划一并决定。

**分析**：按 v1.0 范围，Codex 与 pi 接入时各需登记一个 principal 与一份访问记录；维持选项 A 时，每接入一个 harness 都要修改两份配置并重启后端。

## 4. 调用方证明

### P-1 经网络接入的执行者如何证明身份

2026-11-05 由[第五代架构 Idea](./fifth-generation-architecture.md) 移入。外部 principal 以接入凭据证明身份；HTTP 入口仍以 server 登记的 principal 接入，请求头中的用户、Workspace 与 Agent 只是声明（[认证与访问登记](../workspace/authentication.md#8-已知限制)第 8 节）。

- **P-1b 请求与进程的归属校验**：请求的 principal 必须与进程注册时一致 / 其他规则。只涉及 controller 模式，即外部 harness 经绑定执行线程的操作适配器回调时（[外部 Actor Idea](./external-actor-registration-and-runtime-access.md#controller-模式的步骤分析退回候选方向) 的后续步骤）。统一身份授权入口下，它是 controller 回调在边界上的比对（第 5 节）。
- **P-1c 身份证明的实现位置**：owner 决定（2026-10-29）由统一认证网关证明，每个外部 principal 一份本地共享密钥作为接入证据，一个 harness 对应一个 principal；已实施，决定与复核记录在[外部 Actor Idea](./external-actor-registration-and-runtime-access.md#p-1c-与-p-9c-身份证明与不建进程的访问-context)。

### HTTP 管理入口的调用方证明

问题、证据与影响见[本机 HTTP 管理接口 Todo](../archive/todo/local-http-management-access.md)（已归档）：管理凭据落地之前，前端使用的 Workspace 路由以 server 的 principal 经网关接入，`config`、`providers`、`models`、`logs`、`runtime_events` 不经网关，本机任何进程（包括 harness 中的 agent）都能调用，绕过 Workspace 访问登记的白名单，以 `system` 声明进入 root API。owner（2026-11-01）决定下一步修复；它会改变整个 HTTP API 的信任模型，Todo 要求实施前先在 Idea 或计划中定下认证方式，本节承接这一设计问题。

| 选项 | 内容 | 影响（分析） |
|:---|:---|:---|
| A | HTTP 管理接口增加本机管理凭据，前端随页面取得，外部进程拿不到 | 管理路由与前端使用的 Workspace 路由都要核对凭据；页面由同源后端提供时可以注入，Vite 开发模式与桌面壳需要各自的取得方式（桌面壳可由主进程下发） |
| B | 管理接口单独监听另一个端口或 Unix socket，与 `/mcp`、`/api/v1/import/turns` 分开 | 端口分离本身不阻止同机进程连接，需要配合 socket 的文件权限或凭据；WSL2 会把 Windows 上的本机地址转发到 WSL 中的服务 |
| C | 收紧为只读，或移除不经认证网关的写接口 | 可写面变小，但 Settings、Provider 与模型管理依赖这些写接口，需要另找路径 |
| D | 其他 | —— |

**owner 决定**（2026-11-05）：选项 A，管理凭据。现在先做到防误用；v1.0 的桌面客户端完成防刻意。

- **防误用**：agent 在工作中顺手调用管理接口时被拒，例如 2026-11-01 dsh 中的 agent 直接调用 `/api/v1/config`。
- **防刻意**：agent 专门在本机寻找凭据也拿不到。harness 的 agent 以同一个系统用户运行，用户能读的文件与环境变量它都能读，所以凭据要放在它拿不到的地方：桌面客户端主进程的内存，或需要人参与的口令。
- harness 的共享密钥目前同样只做到防误用（[认证与访问登记](../workspace/authentication.md#8-已知限制)第 8 节）。

防误用部分已落地（2026-11-05，[归档计划](../archive/plans/http-management-credential.md)）：`/api/` 下除导入之外的接口都要求管理凭据，后端默认只监听本机地址，当前事实见[认证与访问登记](../workspace/authentication.md#1-两阶段认证)第 1 节。本节只剩防刻意。

**防刻意的候选设计**（不是 owner 决定，随桌面客户端的计划细化）：桌面客户端的主进程启动后端时生成凭据，经管道而不是文件或环境变量交给后端，再交给自己的页面，凭据不落盘。带着正确管理凭据的 HTTP 请求成为经过证明的工作 actor（第 5 节）。

### P-5 被动请求的认证

2026-11-05 由[第五代架构 Idea](./fifth-generation-architecture.md) 移入。被动请求只存在于 controller 模式（[外部 Actor Idea](./external-actor-registration-and-runtime-access.md#11-两种接入模式owner2026-09-27) 1.1）。

- **P-5b 触发器的 principal**：触发器的登记者 / 系统内置 principal / 其他。owner 倾向（2026-09-27）由登记的 Agent 反推 principal；
- **P-5c 触发器的准入检查时点**：登记触发器时 / 每次触发时 / 两者都查。

## 5. 访问 context 与操作授权

访问 context 的授予内容只包含 actor、驻留 Workspace、principal 与运行绑定（[认证与访问登记](../workspace/authentication.md#3-访问-context密封的运行时凭据)第 3 节）。owner 决定（2026-11-05）以统一身份授权入口取代按运行类型分化的访问 context（下一节）；以下两个方向届时落在工作 actor 或进程记录上，各自随相应问题决定：

- **adapter**：context 暂存签发时的 principal（I-2），adapter 是否一并保存未决，随 P-1b 一并决定；统一入口下它是工作 actor 的字段；
- **本次运行的操作上限**：只有 P-4a 采用“进程持有白名单的子集”时才需要携带它；统一入口下记在进程记录上。

context 没有固定有效期，随绑定的运行结束失效。原先列为开放方向的“plugin 模式需要长连接时是否加有效期”已不再成立：plugin 模式的 MCP 访问每次工具调用签发一份请求级 context，调用结束即撤销（P-9c，[MCP 记忆访问](../contracts/mcp.md)），没有长连接持有 context。统一身份授权入口下不再签发请求级 context，见下一节“对 P-9c 的影响”。

### 访问 context 与授权入口的收拢

2026-11-05 新列，同日 owner 决定方向（选项 B）。

**owner 观察**（2026-11-05）：

- 最初设想 Workspace 体系下只需要一个权威的 Workspace context，随着开发推进，关系越来越乱。
- 访问 context 与 `IdentityScope` 的分割最为关键，也间接催生了线程 context：每次操作（工具调用）的发起者与操作对象只能在运行时确认。
- controller 模式以任务进程驱动，plugin 模式是纯粹的操作响应，这一区别加大了裂缝。
- 新需求不断出现，造成了“加”；为避免反复的决策矛盾，又造成了“不改”。结果是体系不统一，漏洞越来越大。
- owner 对进程 context、线程 context 与操作授权信息来源的层层概念不满意，代码可读性差到无法理解现状。

**现状**（[身份与访问体系现状调查](../governance/baselines/identity-access-current-state-2026-11-05.md)第 4、6、10 节）：

- 访问 context 按运行类型分为进程、执行线程（主线程与子线程）和请求三种。“线程结束后拒绝迟到的请求”同时由线程 context 的撤销与线程表的存活判定承担。
- 操作授权者有七个方法，其中四个共用同一组前置检查。阶段授权与导入授权不检查白名单，它们的安全依赖架构测试限定调用位置。
- 请求级提交另加了派发话题与 `request_level` 参数。MCP 写入先经导入授权落位，再经 actor 授权登记。
- 回答“某个外部调用方能做什么”，要依次查 principal 登记、Workspace 访问登记、入口开放的请求类型，以及它经过的授权方法。

**分析**（不是 owner 决定）：

- 发起者与目标在运行时确认，这是本质（第 7 节前提 2）。偶然的是身份随运行类型分化，撤销又兼任存活判定。
- 每出现一种运行形态（CALL 子线程、不建进程的请求、带派发话题的写入），就多一种 context 用途、一条授权分支。controller 与 plugin 的差别本是运行形态不同，身份体系也因此分裂。
- “不改”的一个来源是：决定只以结论保存，没有连同前提保存。“context 不公开字段”的理由（防止把驻留身份当作操作身份）至今成立；但密封、三种 context 与特权兑现这套手段，是在只有 Alice 任务进程一种运行形态时定下的。plugin 模式出现后，前提已经变了。

**伪造与寿命是两件事**（分析，2026-11-05 与 owner 讨论）：

- 因果链：
  1. 进程内的身份值（`ActorIdentity`、`IdentityScope`）谁都能构造；
  2. 为防伪造，授权的输入被设计成不能伪造、可以撤销的凭据，也就是密封的 context；
  3. 可撤销就要有寿命，寿命要有所有者，context 因此绑到运行上（进程、线程）；
  4. plugin 模式没有运行，于是造出“请求”这种运行（请求级 context）。写入没有进程话题，又补上派发话题与 `request_level`。

  请求级 context 的直接原因是“授权的输入是一个有寿命的凭据”。伪造只是这个设计的出发点。
- 现有的防伪造并不完整：
  - Patchouli 公共路由只认 `IdentityScope`、不再授权，而它是普通值。生产代码中除授权者外，资产桥接一处自行构造它；架构测试只禁止 Alice、Patchouli 内部与引擎引用它，没有限制其他位置构造它。
  - 密封 context 在 Python 中同样可以伪造，它的 docstring 也写明只维护可信进程内的调用纪律。

  取得 scope 的前门防得很严，直接构造 scope 的后门只靠纪律。
- 伪造真正发生的位置：
  - 进程边界之外，伪造是真实威胁：外部调用方可以任意填写 `x-user-id`、`X-Agent-ID`，或直接调用 HTTP 管理路由。这只能靠证明解决：第 4 节的调用方证明。principal 标识执行环境，agent 由经过证明的执行环境声明（第 8 节）。
  - 进程之内都是可信代码，不存在伪造，只存在写错，能做的是结构纪律：谁能构造 scope、谁能持有身份，再由测试守住。

  现状正好相反：进程内花了很大力气防伪造而防不住，边界上的伪造却敞开着。

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 维持现有的 context 与授权方法，只做局部清理，例如删除无生产使用方的代码、统一运行类型检查 | 改动小；按运行类型分化的结构保留，新的运行形态（controller 的外部执行单元、更多 harness）仍需新增 context 用途或授权分支 |
| B | 身份不随运行类型分化：主工作 actor、一次操作的授权与运行三者分开，候选设计见下 | 涉及认证网关、操作授权者、任务进程的线程表、操作入口、能力层、MCP 与导入入口，以及相应的架构测试；实施前要先逐条列出现有 context 保证的行为，并确认每一条的去处 |
| C | 其他 | —— |

**owner 决定**（2026-11-05）：选项 B。这条线的方向固定为统一身份授权入口，本问题关闭。以下细节是候选设计，不是 owner 决定，可在计划评审时调整。

**统一身份授权入口的候选设计**：

命名（owner，2026-11-05）：不使用 Caller 与 Callee。CALL 来自 Alice 唤起子 agent 的方式，不适合作为 Workspace 体系的通用概念。owner 倾向 MainWorkingAgent 或 MainWorkingActor，对应 SubWorkingAgent 或 SubWorkingActor。下文暂用 Actor 一组（分析）：root 管理的主工作 actor 是保留的 `system`，它不是 Agent。

**统一的认证授权主线**：plugin 请求、任务进程（现在的 Alice chat，以后的 controller 模式）、管理路由与导入共用同一个认证网关、同一个授权入口与同一张规则表，产出同一种 `IdentityScope`。

```text
第 1、2 阶段：证明调用方 + Workspace 准入 → 工作 actor
    ├─ plugin 请求：入口的局部值，请求结束即消失
    └─ 任务进程：存入进程记录；线程记录持有主工作 actor 或子工作 actor；授权前先判定线程存活
第 3 阶段：authorize(工作 actor, 操作, 目标 Workspace) → IdentityScope（按规则表）
第 4 阶段：能力层 → Patchouli 公共边界 → 资源 owner（不变）
```

- 认证之后不能跳过第 3 阶段。两阶段认证只回答调用方是谁、能否进入这个 Workspace，操作是否获准（白名单）在第 3 阶段判定。一次请求可以有多次授权，例如 MCP 写入以同一个局部 actor 先授权 `memory.write`，再用于落位。
- 任务进程内，阶段调用使用主工作 actor；actor 操作使用提交它的线程的 actor，主线程是主工作 actor，子线程是派生时经过准入的子工作 actor。
- 各入口在边界上的动作：

  | 入口 | 边界上做什么 | 身份从哪来 |
  |:---|:---|:---|
  | plugin 请求、管理路由、导入 | 证明调用方，加准入 | 这次认证的结果 |
  | 任务进程注册 | 证明调用方，加准入 | 这次认证的结果，存入进程记录 |
  | controller 回调：外部 harness 经绑定的线程回调 | 证明调用方，比对它与进程记录中的 principal（P-1b） | 进程或线程记录，不从请求重新建立 |

- 不在这条主线上的是后台任务。以 `system` 发起的 SETTLE 等任务不经认证，只携带 `belong_to` 与 `from_actor`，由资源 owner 检查，保持不变。
- 主线统一之后，边界上能证明什么，取决于第 4 节 HTTP 管理入口的调用方证明（owner 已决定以管理凭据修复）。principal 标识执行环境，agent 由经过证明的执行环境声明，不另设 principal 与 agent 的绑定（第 8 节）。管理凭据（防误用）已于 2026-11-05 落地；它如何表达进工作 actor（下文流程表中的“未证明”），由统一入口的计划决定。

主线中的概念：

- **主工作 actor（MainWorkingActor）**：入口认证一次得到，字段为 principal、是否经过证明与 actor。一次请求或一个任务进程有一个主工作 actor。它只用于进程内部的协调，只留在进程或线程记录中；plugin 请求没有记录，它是入口调用链内的局部值。它不交给能力层、Patchouli 或 CPU。
- **子工作 actor（SubWorkingActor）**：任务进程派生子线程时，为子线程的 actor 做准入后得到。它沿用主工作 actor 的 principal 与证明，只换 actor；两者字段相同，区别只在来源。当前只有 Alice 的 CALL 派生子线程。
- **授权入口**：`authorize(工作 actor, 操作, 目标 Workspace)` 是唯一的授权入口，按规则表判定并产出 `IdentityScope`。授权由持有记录的一侧发起：线程表、进程，或 plugin 的入口。交出去的只有 `IdentityScope`，能力层方法接收已授权的 scope，不再接收访问 context。目标仍由运行（进程记录中的 Workspace）或请求给出，操作请求不能指定。
- **运行**：任务进程与执行线程只负责存活与派发话题，不承载身份。CPU 拿到的是线程句柄，它只是 `(process_id, thread_id)` 的引用，不含身份。plugin 模式是“有主工作 actor、没有运行”，controller 模式是“有主工作 actor，加一个进程运行”，两者共用同一个授权入口。
- **一条纪律**：`IdentityScope` 与工作 actor 只能由授权入口与认证产出，只在一次调用链或运行记录内存在，不写入其他记录、事件与后台任务。由架构测试限制 `IdentityScope(...)` 的构造位置（现在没有这条测试）。现有测试“能力层公共方法不接收 `IdentityScope`”的本意是 scope 不能由调用方组装，届时改写为“scope 只由授权入口产出”。资产桥接这一例外随 WorkspaceAsset 的归属拆分移除（第 8 节）。

规则表。所有操作的共同前提：准入记录有效，目标 Workspace 的 owner 是 actor 的用户。

| 操作 | 规则 |
|:---|:---|
| root 管理 | actor 是 `system` |
| 六项 actor 操作 | actor 不是 `system`，且操作在访问登记的白名单中 |
| 阶段操作 | actor 不是 `system`；不查白名单 |
| 导入 | actor 不是 `system`；不查白名单 |
| 进程控制 | 与进程在同一 Workspace（P-7 未决，先沿用现状） |

现有概念的去处：

| 现有概念 | 它保护的行为 | 选项 B 下由什么保证 |
|:---|:---|:---|
| 进程 context | 阶段调用以进程 Agent 的身份进行；进程结束后不能再做阶段调用 | 进程记录持有主工作 actor；进程是否结束由进程表判定 |
| 主线程 context 及其撤销 | 停止后迟到的 actor 请求被拒绝 | 线程表的存活判定（现在已是唯一的存活判定） |
| 子线程 context | 子线程以自己的 actor 运行，按它的白名单授权 | 线程表派生子线程时做准入，线程记录持有子工作 actor |
| 请求级 context 及其撤销 | MCP 每次工具调用一份身份，调用结束后失效 | 主工作 actor 只是这次调用的入口局部值，调用返回即消失（见下文对 P-9c 的影响） |
| 密封与特权兑现接口 | 防止代码把驻留身份直接当作操作身份 | 授权入口是唯一能产出 `IdentityScope` 的地方，构造位置由架构测试限制 |
| 四个授权方法与 `request_level` | 各调用面的不同规则；线程 context 不能借请求级提交绕过存活判定 | 授权入口与规则表；请求级提交接收主工作 actor，线程提交接收线程句柄，Alice 只持有句柄 |

**主要流程的变化**。以一个场景推演：前端用 coder 对话，coder 先检索，再派生子线程运行 reviewer，reviewer 写入一条记忆；用户中途停止时，reviewer 还有一个读取请求在路上；同时 dsh 经 MCP 写入一条记忆。

| 流程 | 现在 | 选项 B |
|:---|:---|:---|
| chat 注册 | 网关签发密封的进程 context，存进进程记录 | 网关认证得到主工作 actor `(server, 未证明, (default, coder))`，连同 Workspace 存进进程记录 |
| 阶段调用：Gateway、Profile、prepare、附件、finalize、cleanup | `authorize_stage(进程 context, 目标)` | `authorize(主工作 actor, 阶段操作, 进程的 Workspace)` |
| 进入 Actor 阶段 | 线程表从进程 context 派生主线程 context，与线程描述一起交给 CPU | 线程表登记主线程记录 `(main, coder, 运行中)`，交给 CPU 线程句柄 |
| coder 检索 | 线程表判定存活；`authorize_actor_operation` 确认 context 种类，再兑现、准入、检查目标与白名单 | 线程表按句柄判定存活，并取出线程记录中的 actor；`authorize(coder, memory.search, 进程的 Workspace)` |
| 派生 reviewer 子线程 | 以 coder 授权 `agent.call` 并读取图纸；线程表从进程 context 派生子线程 context，并为 reviewer 做准入 | 以 coder 授权 `agent.call` 并读取图纸；线程表为 reviewer 做准入，登记子线程记录，持有 reviewer 的子工作 actor |
| reviewer 写入 | 子线程 context 授权 `memory.write`，再解析一次线程，取得进程标识与话题 | 按句柄取出线程记录，`authorize(reviewer, memory.write, 进程的 Workspace)`，从同一条记录取得进程标识与话题 |
| 子线程结束 | 撤销子线程 context | 线程记录标为已结束 |
| 用户停止 | 以 `(default, system)` 取得请求级 context；比对两份 context 的驻留 Workspace；撤销全部线程 context | 主工作 actor `(server, 未证明, (default, system))`；`authorize(…, 进程控制, 进程的 Workspace)`；全部线程记录标为已结束 |
| reviewer 迟到的读取 | context 已撤销、进程也已停止，两道判定都拒绝 | 线程表看到线程已结束，一道判定 |
| 进程关闭 | 撤销进程 context | 进程记录移出进程表 |
| dsh 的 `memory_write` | 请求级 context；先以 `authorize_import` 落位（不查白名单），再以 `request_level=True` 授权 `memory.write` | 主工作 actor `(harness:dsh, 已证明, (default, dsh))`；先 `authorize(…, memory.write, 请求的 Workspace)`，通过后再落位并登记 |
| 管理路由、导入 | 请求级 context；`authorize_root` 或 `authorize_import` | 主工作 actor；`authorize(…, root 管理)` 或 `authorize(…, 导入)` |

不变的部分：两阶段认证；root 与 actor 两个调用面；白名单是 Agent 权限的唯一来源；授权后仍产出 `IdentityScope`，Patchouli 在公共边界拆为 `belong_to` 与 `from_actor`，资源授权不变；目标由运行或请求给出。回答“某个外部调用方能做什么”，只需看规则表与访问登记。

**对 P-9c 的影响**：选项 B 重新打开 P-9c 的实现，但保留它的目标：每次工具调用一份身份，调用结束即失效。plugin 的入口本身就是授权点：

1. 认证得到主工作 actor，它是入口的局部值；
2. 以它调用授权入口，产出 `IdentityScope`；
3. 以 scope 调用 actor API；
4. 请求返回，两者一同消失。

不再需要请求级 context 这层包装，也不需要 `request_level`：Alice 只持有线程句柄，请求提交与线程提交的入口类型本来就不同。

**代价与待核对**（分析）：

1. **身份的失效方式**。
    - 现在的 context 带撤销标记，运行结束时被撤销。误留到运行结束之后的 context 再拿去授权会失败。
    - 进程与线程路径：工作 actor 只留在记录中，不往外交，授权由记录一侧发起。记录移出进程表或线程表即失效，不需要撤销标记。
    - plugin 请求路径：主工作 actor 是入口的局部值，只有被误放进比请求活得更久的后台任务或闭包时才会留存。这与今天的 `IdentityScope` 处境相同，它本就是不可撤销的普通值，依赖的也是上文候选设计中“一条纪律”的不变量。选项 B 在这条路径上没有带来新的风险，只是让同一条不变量多覆盖一个类型。
    - 也可以为每个请求建一份轻量记录，让“记录移除即失效”在请求路径上同样成立；这等于为 plugin 重新引入一种运行，不建议。
2. **持有限制仍然需要**。
    - 阶段操作与导入不查白名单。现在，阻止 Alice 或其他代码借 `authorize_stage` 绕过白名单的，是架构测试限定这些方法只能出现在哪些模块（调用点限制），授权方法本身不检查是谁在调用。
    - B 下规则表写明了“阶段操作不查白名单”，持有工作 actor 的代码都能以它请求阶段操作。所以仍需要结构规则：工作 actor 只在入口、进程记录与线程记录中，Alice 与 CPU 只持有线程句柄。现在已有对应的架构测试。
    - 变化是从“四个授权方法各自限定调用模块”收缩为“谁能持有工作 actor、谁能构造 scope”，不是取消结构规则。在单个 Python 进程内，区分特权最终都要靠结构纪律。
3. **改动面**：认证网关、操作授权者、线程表、操作入口、能力层 26 个方法的签名（`access` 改为已授权的 `IdentityScope`）、MCP 与导入入口、`server/deps.py`，以及约 40 条身份相关架构测试中的大部分。
4. **细节核对**：例如 UPDATE 的冷读跨越线程结束时不登记写入，现在依赖“同步段只解析一次线程”，B 下必须保持。实施前要逐条列出现有 context 保证的行为，确认每一条的去处。

暂不处理：跨 Workspace 的目标（第 7 节），目标仍只能等于运行或请求所在的 Workspace；P-4a 若选择“进程持有白名单的子集”，子集记在进程记录上，由授权入口读取。

**与其他问题的关系**：主工作 actor 中“是否经过证明”与 actor 的含义，取决于第 4 节 HTTP 管理入口的调用方证明，两者应一并设计，避免收拢之后再加一层。principal 标识执行环境、不与 agent 绑定（第 8 节），不需要另行设计绑定。P-1b 成为主线中 controller 回调在边界上的比对，P-7 的授权主体成为规则表中的条目。

### P-4a 进程能否持有白名单的子集

2026-11-05 由[第五代架构 Idea](./fifth-generation-architecture.md) 移入。

- 支持：注册时声明所需操作集合，入口校验它是白名单子集，能力层按进程的集合授权；本次运行的操作上限由访问 context 携带（统一身份授权入口下记在进程记录上）；
- 不支持：进程一律继承 actor 在该 Workspace 的白名单。

### P-7 进程控制操作的授权主体

2026-11-05 由[第五代架构 Idea](./fifth-generation-architecture.md) 移入。取消由持有进程句柄的入口与进程控制授权处理；状态查询等其余控制操作的授权主体尚未决定。

选项：仅进程的注册 principal / 注册 principal 与管理员 / Workspace 内任何获准者 / 其他。

## 6. 资源可见性与资源授权

现在只有 MemoryAtom 带资源 policy（PUBLIC、PRIVATE、TEAM，只表达读取）；Topic、Artifact、WorkspaceAsset 与记忆任务只按归属授权（[身份与授权](../workspace/identity-and-authorization.md#5-资源归属与寻址)第 5 节）。

约束（2026-10-04，由“资源上的来源字段不参与授权”得出）：与创建者相关的权限，例如“只有创建者能修改”，应在资源创建时写入资源 policy，授权时不读取来源；否则来源会重新成为访问条件。

### P-8 资源边界是否增加修改与执行授权

2026-11-05 由[第五代架构 Idea](./fifth-generation-architecture.md) 移入。`MemoryAccessPolicy` 只表达读取。

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 资源策略扩展到修改与执行 | 需要扩展 `MemoryAccessPolicy` 或新增策略模型，并确定默认值与存量迁移 |
| B | 不增加 | 修改权限只由操作白名单与读取可见性共同约束 |
| C | 只对部分资源类型增加 | 需要划分资源类型 |

### 写入意图的可见性与 policy

2026-11-05 由[写入意图后续方向 Idea](./pending-intent-migration.md) 第 5 节移入。

**问题**：意图不设 policy，同 Workspace 内持有别名即可回读（UPDATE 意图跟随基础原子的可读性）。别名由标题或内容片段加 4 位十六进制后缀组成，“只有写入者知道别名”是惯例，不是访问控制，真正的边界是 Workspace。`WriteFocus` 只有 content、reason、title，不携带目标 policy。

需要回答：

- WRITE 若能声明目标 policy，落库前的意图可见性是否跟随该 policy；
- 增加意图 policy 时，不要求外部构造 RuntimeScope，也不把整个 scope 相等重新变成权限条件；
- 读取正式原子时，是否提示调用者自己仍在进行的 UPDATE 修订；不提示时，Pending 正文不能静默叠加到已提交版本上。

管理页查看与编辑记忆策略的前端组件见 [Memory 可见性策略前端组件 Todo](../todo/memory-visibility-policy-ui.md)，它依赖 Memory CRUD 契约先支持 `access_policy` 的读写。

## 7. 受限穿透访问

owner（2026-10-03）：`IdentityScope` 由操作的发起者与目标 Workspace 组成，正是为了以后 actor 的一次操作可以跨过它所在的 Workspace，对其他 Workspace 做有限访问。actor 进入 Workspace 与 actor 真正进行操作，两者的 Workspace 可以不同：coder 在 WA 中执行任务，创建任务进程时确认 coder 能进入 WA；执行过程中 coder 想去 WB 查找资源，这次操作的发起者是 coder，目标是 WB，与 coder 驻留在 WA 无关。

现在第 3 阶段只接受目标等于驻留 Workspace；授权点接口已经显式接收目标 Workspace，实现受限穿透访问时只需放宽这条规则，不必改动各授权点的签名。

需要回答：

- **授予模型**：谁能对哪个非驻留 Workspace 做什么。分析（2026-10-10）：一次跨 Workspace 的操作要同时满足 agent 自身的白名单与目标一侧的授予；
- **agent 的身份**：Agent Profile 是某个 Workspace 中的记忆原子，`agent_id` 即其 alias，只在该 Workspace 内唯一，不同 Workspace 中的同名 agent 是两条独立的原子。“同一 agent 在不同 Workspace 有不同能力”只会出现在穿透访问中，那时 actor 身份需要带上驻留 Workspace（分析，2026-10-10）；
- **目标的表达**：执行一侧如何给出目标，见本节末尾的 T-5b；
- **资源边界**：资源 owner 仍按目标 Workspace 校验归属；同一进程内会同时出现多个目标 Workspace 的资源，派生缓存按派生源坐标键控，不能从 scope 的存在推导隔离；
- **与子网的关系**：Mount、Bridge 与能力子网是更完整的形态（[AE2 Idea](./ae2-hivememory-architecture-analogy.md)，GitHub Epic #92 的主子网部分），穿透访问的授予模型应能向它扩展。

以下前提与问题于 2026-10-19 从原《执行单元、执行线程与执行环境》Idea（现[执行环境 Idea](./execution-environment.md)）移入，沿用原编号。

**前提 2**（owner，2026-10-05）：一次操作的发起者与目标分开看待。

- **发起者**是客观事实，由执行单元一侧补全，agent 不在工具调用中声明；当前由发起操作的执行线程给出，见 [执行线程](../workspace/execution-threads.md)。
- **目标 Workspace** 是 agent 的意图：现在由补全的一层给出，只接受驻留 Workspace；实现跨 Workspace 的受限穿透访问后，agent 需要能够自己带上操作目标（T-5b）。

### T-5b 跨 Workspace 目标的表达形式

- **选项**：按指令给出目标，一条指令一个目标 / 按引用限定，alias 带上 Workspace 限定 / 其他。
- **影响**：
  - alias 按 (Workspace, alias) 分区，跨 Workspace 的引用必须带限定；
  - agent 只能给出 Workspace key 一类名称，`WorkspaceIdentity` 中的 owner 由补全的一层解析，解析失败与无权访问需要对 agent 表现一致；
  - （分析）MTP 文本不能指定 Workspace 的现有约束，随穿透访问设计另行评审；发起者不能由工具调用指定（前提 2）长期成立。

## 8. 例外、暂缓与已知限制

以下各项不是本文的开放问题，按各自的去向跟踪，这里只列出它们属于这条线。

**已登记的例外与暂缓**

- WorkspaceAsset 的 Store、解析服务与旧端口仍接收 `IdentityScope`，归属拆分暂缓：[WorkspaceAsset 归属拆分 Todo](../todo/workspace-asset-ownership-identity-split.md)；
- chat 请求体中 `session_id` 的未来语义：[外部会话与 Topic 投影 Idea](./external-session-and-topic-projection.md)；
- **root 的声明范围**（owner，2026-10-20）：能以保留的 `system` 进入 root API 的只有用户经 HTTP 入口的管理访问，其他接入来源不能声明 `system`。已落实（2026-10-29，[记忆访问计划](../archive/plans/v0.7.1-memory-access-mcp.md)）：接入登记的 `allow_system_actor` 逐个声明能否声明 `system`，默认不能，server 与 Profile 注入脚本显式允许，外部 harness 不能（[认证与访问登记](../workspace/authentication.md#1-两阶段认证)）。Profile 注入脚本（`hivememory:profile-injector`）不在这条规则的范围内（owner，2026-10-20）：它用于向默认 Workspace 快速注入预设 Agent，理想情况下只在项目初次部署时使用，不应经过权限系统；现状仍作为登记来源经网关以 `system` 进入 root API，项目接入部署流程时再调整。

**相关 Todo 与治理工作包**

- RUN 的错误可以探测记忆的类型与是否存在，不可执行的目标也会记录一次引用：[RUN 目标的可执行性检查 Todo](../todo/run-target-executability-check.md)；
- 前端身份切换时的持久化、失效与清理：[前端身份状态所有权 Todo](../todo/frontend-identity-ownership.md)，对应[身份治理](../governance/security/identity-and-execution-safety.md) Phase S4；
- 显式 Profile 解析失败、权限拒绝与未指定 Profile 的稳定结果，以及失败事件不泄漏不可见正文：身份治理 Phase S1 第 4、5 项；
- 身份与威胁模型清单：身份治理 Phase S0，尚未开始。

**事实文档登记的已知限制**（[认证与访问登记](../workspace/authentication.md#8-已知限制)第 8 节、[身份与授权](../workspace/identity-and-authorization.md#8-已知限制)第 8 节）

- 接入凭据没有签发、轮换、撤销与过期，修改经重启生效；
- principal 标识执行环境，不与 Agent 绑定。这是设计，不是限制：依据是[执行环境 Idea](./execution-environment.md) 前提 9 与外部 Actor Idea 的 E-6a（owner，2026-10-06），以及一个 harness 对应一个 principal（P-1c）。同一执行环境可以以不同 Agent 运行，Agent 由经过证明的执行环境声明，权限与可见性只跟随 Agent。
  - 仍然存在的限制是可声明的 Agent 不要求已登记：用户级记录接纳任意具体 Agent；chat 需要对应的 Profile，导入与 MCP 不要求。
  - 现有登记中的 `claude_code` 与 `dsh` 把 harness 名称当作 Agent，是前提 9 之前“harness 即 agent”看法的残留，它们的白名单实际上按执行环境限权。plugin 模式目前不能选择 Agent，暂时可用；支持选择 Agent 时需要重新处理。
- 用户级访问记录让同一用户的所有具体 Agent 共享一份白名单；
- Patchouli 的交互提交与主动记忆意图提交路由只接收 `IdentityScope`、不做操作授权，依赖唯一的生产调用方先完成授权。

## 9. 地基整理的起点（分析）

以下是分析，不是 owner 决定，也不预排计划。

- 这条线目前没有一份统一的清单，能逐项列出各入口、路由、actor 操作、缓存、后台任务与资源操作的身份输入与授权所有者，也无法据此区分哪些是设计、哪些是补丁。身份治理 Phase S0 定义的正是这份清单，适合作为整理的第一步，第 8 节的例外与已知限制可以据此判断去留。
- 按发布影响，第 4 节的 HTTP 管理入口与第 3 节的登记生效时点直接关系桌面端与多个 harness 同机运行；第 5 节的 P-1b、P-5、P-7 与第 7 节主要服务 controller 模式与多 Workspace，可以留到对应方向重启时再决定。

## 10. 相关文档

| 文档 | 关系 |
|:---|:---|
| [身份与授权](../workspace/identity-and-authorization.md)、[认证与访问登记](../workspace/authentication.md)、[能力层](../workspace/capability.md) | 身份模型、访问边界与调用面的当前事实 |
| [身份治理](../governance/security/identity-and-execution-safety.md) | 跨版本的身份与执行安全目标、成熟度与工作包 S0–S4 |
| [身份与访问体系现状调查](../governance/baselines/identity-access-current-state-2026-11-05.md) | 2026-11-05 冻结的现状证据：各入口的身份来源、授权点、资源授权的实际效果、补丁与例外；第 5 节收拢问题的现状依据 |
| [第五代架构 Idea](./fifth-generation-architecture.md) | 3.5、3.6 的理念；P-1、P-4a、P-5、P-7、P-8 的原所在（2026-11-05 移入本文） |
| [外部 Actor Idea](./external-actor-registration-and-runtime-access.md) | E-1 的原所在（2026-11-05 移入本文）；P-1c、P-9c 的决定与复核；controller 模式的身份需求（E-3b、绑定线程的操作适配器） |
| [写入意图后续方向 Idea](./pending-intent-migration.md) | 意图可见性与 policy 的原所在（2026-11-05 移入本文） |
| [执行线程](../workspace/execution-threads.md)（已归档的[执行线程 Idea](../archive/ideas/execution-thread-definition.md)） | 操作的发起者由执行线程给出；线程的目标现为驻留 Workspace，跨 Workspace 目标的表达见第 7 节 T-5b |
| [执行环境 Idea](./execution-environment.md) | 前提 2 与 T-5b 的原所在（2026-10-19 移入本文） |
| [AE2 Idea](./ae2-hivememory-architecture-analogy.md) | 子网与跨 Workspace 的更完整形态 |
