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
  - docs/archive/ideas/pending-intent-migration.md
  - docs/ideas/execution-environment.md
  - docs/ideas/ae2-hivememory-architecture-analogy.md
  - docs/archive/ideas/execution-thread-definition.md
  - docs/archive/todo/local-http-management-access.md
  - docs/archive/plans/http-management-credential.md
  - docs/archive/plans/unified-identity-authorization-entry.md
  - docs/archive/plans/workspace-asset-ownership-split.md
  - docs/archive/plans/memory-visibility-management.md
last_reviewed: 2026-11-20
---

# 身份与访问体系

## 0. 文档性质

- **主线定位**：身份与访问体系作为单独的主线方向维护。它是 Workspace 体系的基础，不只服务 controller 模式，覆盖从 principal 的登记与调用方证明，到进入 Workspace 的身份、操作调用的授权，再到记忆等资源的可见性；最好在发布前为系统打好地基，否则 controller 与 plugin 两种接入都难以继续。
- 本文是这条线的设计：已定的设计、尚未实施的部分与开放问题。理念见[第五代架构 Idea](./fifth-generation-architecture.md)第 3.5、3.6 节；当前的身份模型见[身份与授权](../workspace/identity-and-authorization.md)与[认证与访问登记](../workspace/authentication.md)，三个调用面见[能力层](../workspace/capability.md)。
- 正文是已定的设计，候选设计标“候选”；已经实现的部分以事实文档为准，本文只链接。问题沿用原编号（E-1、P-1、P-4a、P-5、P-7、P-8、T-5b），只列选项及其影响，选项顺序不代表倾向。

## 1. 为什么单独成线

身份与访问经过多轮返工，每一轮都挂在当时的主线方向下进行（访问边界返工、能力层分层、执行线程、MCP 记忆访问、plugin 模式的写入与更新）。结果是：已落地的模型分布在多份事实文档中，未决的部分散落在几份 Idea、Todo 与身份治理中，例外与补丁逐渐累积。单独成线，是为了让这条线上的判断只在一处维护，每次改动都对照全链路。

已经具备、本文不再讨论的地基：三个身份模型与两阶段认证、两阶段授权；统一身份授权入口：认证得到工作 actor，授权只有 actor 执行期间的 `authorize_agent` 与系统级操作的 `authorize_root` 两个方法，交互的参与者作为数据显式传递（第 5 节）；资源归属与发起者的拆分；外部 principal 以本地共享密钥证明身份（P-1c）；`allow_system_actor` 限定能声明 `system` 的来源；不建进程的访问每次工具调用认证一次、身份只在本次调用内有效（P-9c）。

**对发布的影响**（分析）：v1.0 范围包括桌面客户端，以及 Claude Code、dsh、Codex 与 pi 四个 harness 的 plugin 模式接入。桌面端常驻后端之后，本机进程、各 harness 中的 agent 与前端都是调用方，本地单用户的信任假设与访问登记的维护方式都会被直接触及。

## 2. 全链路与缺口

| 环节 | 已有（当前事实） | 缺口与去向 |
|:---|:---|:---|
| Principal 登记 | 两份配置文件在启动时装载，运行中可以经管理接口编辑并写回；外部来源的凭据可以生成与轮换，principal 可以限定可声明的 Agent（[认证与访问登记](../workspace/authentication.md#2-两类访问登记)第 2 节） | 第 3 节 |
| 调用方证明 | 统一认证网关；外部 principal 以共享密钥证明；任务进程注册、Import Bus 与 MCP 入口经网关；HTTP 管理面要求管理凭据，开发方式下防误用，桌面客户端中防刻意（[认证与访问登记](../workspace/authentication.md#1-两阶段认证)第 1 节） | controller 模式下请求与进程的归属校验（P-1b）；harness 的接入凭据只防误用：第 4 节 |
| 工作 actor 与操作授权 | 工作 actor 只含 principal、actor 与驻留 Workspace；actor 执行期间按白名单授权，系统级操作（root API、任务进程的系统阶段与取消、导入与确定话题）以 `system` 授权（[身份与授权](../workspace/identity-and-authorization.md#4-操作授权)第 4 节） | 工作 actor 是否带 adapter、本次运行的操作上限、进程控制中取消以外的授权主体：第 5 节 |
| 资源可见性 | 只有 MemoryAtom 带资源 policy，且只表达读取，只有用户能在管理页修改（[管理页面](../frontend/management-views.md#111-可见范围)）；其余资源只按归属授权 | 修改与执行授权：第 6 节 |
| 进入其他 Workspace | 第 3 阶段只接受目标等于驻留 Workspace | 受限穿透访问：第 7 节 |
| 例外、暂缓与已知限制 | — | 第 8 节 |

## 3. Principal 登记的来源与生效时点（E-1）

**设计**：

- 两类登记（接入登记与 Workspace 访问记录）在启动时从各自的配置文件装载，运行中可以编辑，改动写回生效的登记文件。
- 两类登记分开维护（E-1a）：接入登记经 System 的管理接口，访问记录经 root API。
- 接入凭据由 HiveMemory 生成与轮换；环境变量照常可用并优先。
- 个人配置不进入仓库：两个生效文件不受版本控制，仓库只发布 `*.example.yaml` 模板，默认路径的生效文件缺失时从模板生成。
- 运行时收回的操作：线程描述中的操作快照不随之更新，收回的操作在调用时由即时授权拒绝。

**现状**：已实现，见[认证与访问登记](../workspace/authentication.md#2-两类访问登记)第 2 节与第 8 节（运行时编辑的生效时点）。主配置的热更新不在这条线上，见 [ROADMAP](../ROADMAP.md) 第 5.5 节“配置热更新”。

## 4. 调用方证明

### P-1 经网络接入的执行者如何证明身份

外部 principal 以接入凭据证明身份；HTTP 入口以 server 登记的 principal 接入，请求头中的用户、Workspace 与 Agent 只是声明（[认证与访问登记](../workspace/authentication.md#8-已知限制)第 8 节）。

- **P-1b 请求与进程的归属校验**（搁置，controller 模式重启时决定）：请求的 principal 必须与进程注册时一致 / 其他规则。只涉及 controller 模式，即外部 harness 经绑定执行线程的操作适配器回调时；在统一身份授权入口下，它是 controller 回调在边界上的证明与比对：在边界上证明调用方，再与进程记录中的 principal 比对，不从请求重新建立身份（第 5 节）。
- **P-1c 身份证明的实现位置**：由统一认证网关证明，每个外部 principal 一份本地共享密钥作为接入证据，一个 harness 对应一个 principal。已实现，决定的依据与复核见[外部 Actor Idea](./external-actor-registration-and-runtime-access.md#p-1c-与-p-9c-身份证明与不建进程的访问-context)。

### HTTP 管理入口的调用方证明

**设计**：HTTP 管理接口要求本机管理凭据，分两层：

- **防误用**：agent 在工作中顺手调用管理接口时被拒，例如 dsh 中的 agent 直接调用 `/api/v1/config`。
- **防刻意**：agent 专门在本机寻找凭据也拿不到。harness 的 agent 以同一个系统用户运行，用户能读的文件与环境变量它都能读，所以凭据要放在它拿不到的地方：桌面客户端主进程的内存，或需要人参与的口令。

理由：管理凭据落地之前，前端使用的 Workspace 路由以 server 的 principal 经网关接入，`config`、`providers`、`models`、`logs`、`runtime_events` 不经网关，本机任何进程（包括 harness 中的 agent）都能调用，绕过 Workspace 访问登记的白名单，以 `system` 声明进入 root API。端口或 socket 分离本身不阻止同机进程连接，收紧为只读又会让设置、Provider 与模型管理失去写路径，所以选择凭据。

**现状**：已实现。`/api/` 下除导入之外的接口都要求管理凭据，后端默认只监听本机地址（[认证与访问登记](../workspace/authentication.md#1-两阶段认证)第 1 节）；桌面客户端中，主进程启动后端时生成凭据，经 stdin 管道交给后端，再放进窗口的内存会话，不落盘，只有桌面窗口能管理（[桌面客户端](../frontend/desktop-client.md#4-管理凭据)）；开发方式（`ps1`、Vite）仍使用凭据文件，只防误用。harness 的接入凭据同样只防误用（[认证与访问登记](../workspace/authentication.md#8-已知限制)第 8 节）。

### P-5 被动请求的认证

搁置，controller 模式重启时决定。被动请求只存在于 controller 模式（[外部 Actor Idea](./external-actor-registration-and-runtime-access.md#11-两种接入模式owner2026-09-27) 1.1）。

- **P-5b 触发器的 principal**：触发器的登记者 / 系统内置 principal / 其他；owner 倾向由登记的 Agent 反推 principal。
- **P-5c 触发器的准入检查时点**：登记触发器时 / 每次触发时 / 两者都查。

## 5. 工作 actor 与操作授权

工作 actor 只包含 principal、actor 与驻留 Workspace（[认证与访问登记](../workspace/authentication.md#3-工作-actor)第 3 节）。以下两个方向落在工作 actor 或进程记录上，各自随相应问题决定：

- **adapter**：工作 actor 保存认证时的 principal（I-2），adapter 是否一并保存未决，随 P-1b 一并决定；
- **本次运行的操作上限**：只有 P-4a 采用“进程持有白名单的子集”时才需要携带它，记在进程记录上。

工作 actor 没有有效期：进程记录与线程记录持有的随进程注销被丢弃，入口为一次请求认证得到的只在本次调用内使用（P-9c，[MCP 记忆访问](../contracts/mcp.md)），没有长连接持有身份。

### 访问 context 与授权入口的收拢

**设计**：身份不随运行类型分化。主工作 actor、一次操作的授权与运行三者分开：

- 认证得到工作 actor，授权只有两个方法：actor 执行期间的 `authorize_agent`，与系统级操作的 `authorize_root`。能力层的 API 代表谁在执行这项操作。
- 由系统以 root 身份执行的操作：取消是用户下达的指令；Profile 与附件是用户为这一轮作出的选择，对用户可见而不是对 Agent 可见，不能以错误的身份执行；Gateway 与确定话题面对的话题属于 Workspace；交互提交是系统级操作，Agent 不能拥有这项能力，chat 的提交、plugin 模式的导入与将来的历史对话导入都在这一范畴。外部 harness 不能声明 `system`，管的是边界上的声明，与此无关。
- 任务进程的预检索以执行者的 actor 操作进行，按执行者的可见性返回记忆。
- 系统组件以自己登记的 principal 经进程内 adapter 认证 `(user, system)`，取得执行系统级操作的身份。
- 命名：实现以一个类型 `WorkingActor` 表达，主工作 actor、子工作 actor 与系统的 root 工作 actor 是它的角色；不使用 Caller 与 Callee，它们来自 Alice 唤起子 agent 的方式，不适合作为 Workspace 体系的通用概念。
- 不采用“认证直接产出 `IdentityScope`、删除入口检查”的最小版本：它会抹掉驻留身份与单次操作 scope 的区分。

理由：

- 发起者与目标在运行时确认，这是本质（第 7 节前提 2）；偶然的是身份随运行类型分化，撤销又兼任存活判定。原来每出现一种运行形态（CALL 子线程、不建进程的请求、带派发话题的写入），就多一种访问 context 用途、一条授权分支，controller 与 plugin 的差别本是运行形态不同，身份体系也因此分裂；回答“某个外部调用方能做什么”要依次查 principal 登记、访问登记、入口开放的请求类型与它经过的授权方法（[现状调查](../governance/baselines/identity-access-current-state-2026-11-05.md)第 4、6、10 节，冻结的快照）。
- **伪造与寿命是两件事**：为防伪造，授权的输入曾被设计成不能伪造、可以撤销的凭据，也就是密封的 context；可撤销就要有寿命，寿命要有所有者，context 因此绑到进程与线程上；plugin 模式没有运行，于是又造出“请求”这种运行。进程内的身份值谁都能构造，密封 context 在 Python 中同样可以伪造。伪造真正发生在进程边界之外：外部调用方可以任意填写请求头，或直接调用 HTTP 管理路由，这只能靠证明解决（第 4 节）；进程之内都是可信代码，只存在写错，能做的是结构纪律，由架构测试守住谁能构造、谁能持有身份。
- 决定要连同前提保存：“context 不公开字段”的理由（防止把驻留身份当作操作身份）仍然成立，但密封、三种 context 与特权兑现这套手段，是在只有 Alice 任务进程一种运行形态时定下的。

**现状**：已实现，见[身份与授权](../workspace/identity-and-authorization.md#4-操作授权)第 4 节、[认证与访问登记](../workspace/authentication.md#3-工作-actor)第 3 至 5 节与[执行线程](../workspace/execution-threads.md)。后台任务不在这条主线上：以 `system` 发起的 SETTLE 等任务不经认证，只携带 `belong_to` 与 `from_actor`。工作 actor 不设“是否经过证明”：证明分别发生在入口（管理凭据、接入凭据）；principal 标识执行环境、不与 agent 绑定（第 8 节）。

### P-4a 进程能否持有白名单的子集

搁置，controller 模式重启时决定。选项：支持，注册时声明所需操作集合，入口校验它是白名单子集，能力层按进程的集合授权，本次运行的操作上限记在进程记录上 / 不支持，进程一律继承 actor 在该 Workspace 的白名单。

### P-7 进程控制操作的授权主体

取消是用户下达的指令，由 root 执行（已实现，[身份与授权](../workspace/identity-and-authorization.md#4-操作授权)第 4 节）。状态查询，以及 controller 模式下由谁控制进程，搁置。选项：仅进程的注册 principal / 注册 principal 与管理员 / Workspace 内任何获准者 / 其他。

## 6. 资源可见性与资源授权

现在只有 MemoryAtom 带资源 policy（PUBLIC、PRIVATE、TEAM，只表达读取）；Topic、Artifact、WorkspaceAsset 与记忆任务只按归属授权（[身份与授权](../workspace/identity-and-authorization.md#5-资源归属与寻址)第 5 节）。

约束：资源上的来源字段不参与授权，所以与创建者相关的权限（例如“只有创建者能修改”）应在资源创建时写入资源 policy，授权时不读取来源；否则来源会重新成为访问条件。

### P-8 资源边界是否增加修改与执行授权

搁置。`MemoryAccessPolicy` 只表达读取。选项：资源策略扩展到修改与执行，需要扩展策略模型并确定默认值与存量迁移 / 不增加，修改权限只由操作白名单与读取可见性共同约束 / 只对部分资源类型增加，需要划分资源类型。

### 写入意图的可见性与 policy

**设计**：记忆的可见性只由用户在管理页修改；被动提取、Agent 写入与手动创建默认生成 PUBLIC；TEAM 保留在模型中，不开放。Agent 写入不声明策略，写入意图因此不加 policy：意图结算出的正式记忆对 Workspace 内所有 Agent 可见，结算前按 Workspace 归属回读与此一致。以后开放 Agent 写入声明策略时，要重新回答意图在结算前的可见性是否跟随声明的策略。

理由：意图的别名由标题或内容片段加 4 位十六进制后缀组成，“只有写入者知道别名”是惯例，不是访问控制，真正的边界是 Workspace；`WriteFocus` 只有 content、reason、title，不携带目标 policy。

**现状**：已实现，见[管理页面](../frontend/management-views.md#111-可见范围)。

搁置：读取正式原子时，是否提示调用者自己仍在进行的 UPDATE 修订（现在读取正式原子始终返回已提交版本，进行中的修订只经它自己的 pending 句柄可见）。

## 7. 受限穿透访问

**设计**：`IdentityScope` 由操作的发起者与目标 Workspace 组成，正是为了以后 actor 的一次操作可以跨过它所在的 Workspace，对其他 Workspace 做有限访问。actor 进入 Workspace 与 actor 真正进行操作，两者的 Workspace 可以不同：coder 在 WA 中执行任务，创建任务进程时确认 coder 能进入 WA；执行过程中 coder 想去 WB 查找资源，这次操作的发起者是 coder，目标是 WB，与 coder 驻留在 WA 无关。

- **agent 的身份**：agent 的定义是全局的，它作为工作者进入 Workspace，不是 Workspace 的资产。Agent Profile 是 AgentProfileStore 中的全局文件，`agent_id` 在实例内唯一；能否在某个 Workspace 工作、在那里能做什么，由该 Workspace 的访问登记决定。穿透访问中，“同一 agent 在不同 Workspace 有不同能力”仍由各 Workspace 的访问登记表达，工作 actor 的驻留 Workspace 区分它此刻在哪里工作。
- **前提 2：一次操作的发起者与目标分开看待**。发起者是客观事实，由执行单元一侧补全，agent 不在工具调用中声明，当前由发起操作的执行线程给出（[执行线程](../workspace/execution-threads.md)）；目标 Workspace 是 agent 的意图，现在由补全的一层给出、只接受驻留 Workspace，实现受限穿透访问后，agent 需要能够自己带上操作目标（T-5b）。

**现状**：第 3 阶段只接受目标等于驻留 Workspace；授权点接口已经显式接收目标 Workspace，实现受限穿透访问时只需放宽这条规则，不必改动各授权点的签名。

候选设计与需要回答的：

- **授予模型**：谁能对哪个非驻留 Workspace 做什么。候选：一次跨 Workspace 的操作要同时满足 agent 自身的白名单与目标一侧的授予；
- **资源边界**：资源 owner 仍按目标 Workspace 校验归属；同一进程内会同时出现多个目标 Workspace 的资源，派生缓存按派生源坐标键控，不能从 scope 的存在推导隔离；
- **与子网的关系**：Mount、Bridge 与能力子网是更完整的形态（[AE2 Idea](./ae2-hivememory-architecture-analogy.md)，GitHub Epic #92 的主子网部分），穿透访问的授予模型应能向它扩展。

### T-5b 跨 Workspace 目标的表达形式

- **选项**：按指令给出目标，一条指令一个目标 / 按引用限定，alias 带上 Workspace 限定 / 其他。
- **影响**：
  - alias 按 (Workspace, alias) 分区，跨 Workspace 的引用必须带限定；
  - agent 只能给出 Workspace key 一类名称，`WorkspaceIdentity` 中的 owner 由补全的一层解析，解析失败与无权访问需要对 agent 表现一致；
  - （分析）MTP 文本不能指定 Workspace 的现有约束，随穿透访问设计另行评审；发起者不能由工具调用指定（前提 2）长期成立。

## 8. 例外、暂缓与已知限制

以下各项不是本文的开放问题，按各自的去向跟踪，这里只列出它们属于这条线。

**已登记的例外与暂缓**

- chat 请求体中 `session_id` 的未来语义：[外部会话与 Topic 投影 Idea](./external-session-and-topic-projection.md)；
- **root 的声明范围**：能以保留的 `system` 进入 root API 的只有用户经 HTTP 入口的管理访问与系统自身的 principal `hivememory:system`，其他接入来源不能声明 `system`；接入登记的 `allow_system_actor` 逐个声明，默认不能，外部 harness 不能（[认证与访问登记](../workspace/authentication.md#1-两阶段认证)第 1 节、[第 5 节](../workspace/authentication.md#5-系统自身的-principal)）。预设 Agent 的写入不在这条规则的范围内：数据根第一次启动时，Patchouli 把随程序发布的预设直接写成 Profile 文件，不经认证网关、root API 与授权，这时还没有任何调用方（[桌面客户端](../frontend/desktop-client.md#7-推理模型与首次部署)）。

**相关 Todo 与治理工作包**

- `agent_id` 是名字而不是不透明的 id：它是 `ActorIdentity` 的一部分，访问登记、外部接入的 `allowed_agent_ids`、对话请求、会话记录、记忆来源与 CALL 都按它引用，只给 Agent Profile 加一个 id 解决不了改名；是否换成不透明的 id 归这条线，与 [Agent 别名改名 Todo](../todo/agent-alias-rename.md) 一起考虑。AgentProfileStore 保持 `agent_id` 作为不可改的标识。
- 前端身份切换时的持久化、失效与清理：不单独追踪，出现登录或 Workspace 切换时再计划，对应[身份治理](../governance/security/identity-and-execution-safety.md) Phase S4；
- 身份与威胁模型清单（身份治理 Phase S0）：身份输入与授权所有者的盘点已由[现状调查](../governance/baselines/identity-access-current-state-2026-11-05.md)完成；越权、Profile 过期等最小复现样本不单独立项，随相关条目实施时补上。

**事实文档登记的已知限制**（[认证与访问登记](../workspace/authentication.md#8-已知限制)第 8 节、[身份与授权](../workspace/identity-and-authorization.md#8-已知限制)第 8 节）

- 接入凭据没有过期；环境变量中的凭据修改后经重启生效；
- principal 标识执行环境，不与 Agent 绑定。这是设计，不是限制：依据是[执行环境 Idea](./execution-environment.md) 前提 9 与外部 Actor Idea 的 E-6a，以及一个 harness 对应一个 principal（P-1c）。同一执行环境可以以不同 Agent 运行，Agent 由经过证明的执行环境声明，权限与可见性只跟随 Agent。
  - 接入登记可以以 `allowed_agent_ids` 限定一个来源能声明哪些 Agent，插件页为 harness 登记来源时初始只允许插件的缺省 Agent，之后可以增加。仍然存在的限制是不限定的来源（前端使用的 server 来源）可以声明用户级记录接纳的任意具体 Agent；chat 需要对应的 Profile。
  - 插件的缺省 Agent 以 harness 名称命名（`claude_code`、`codex`、`pi`、`dsh`，[插件服务](../../src/hivememory/system/plugins/harnesses.py)），是“harness 即 agent”看法的残留，它们的白名单实际上按执行环境限权。plugin 模式目前不能选择 Agent，暂时可用；支持选择 Agent 时需要重新处理。
- 用户级访问记录让同一用户的所有具体 Agent 共享一份白名单；
- Patchouli 的交互提交与主动记忆意图提交路由只接收 `IdentityScope`、不做操作授权，依赖唯一的生产调用方先完成授权。

## 9. 地基整理的起点（候选）

- 整理的第一步是一份能逐项列出各入口、路由、actor 操作、缓存、后台任务与资源操作的身份输入与授权所有者、并区分设计与补丁的清单，即身份治理 Phase S0；[现状调查](../governance/baselines/identity-access-current-state-2026-11-05.md)完成了这份盘点。它是冻结的快照，其中已被统一身份授权入口、Import Bus 授权移入能力层、WorkspaceAsset 归属拆分与交互提交收束改掉的部分，以当前事实文档为准。
- 直接关系桌面端与多个 harness 同机运行的两项（HTTP 管理入口、登记的生效时点）已经实现；剩下的 P-1b、P-4a、P-5、P-7 与第 7 节主要服务 controller 模式与多 Workspace，可以留到对应方向重启时再决定。

## 10. 相关文档

| 文档 | 关系 |
|:---|:---|
| [身份与授权](../workspace/identity-and-authorization.md)、[认证与访问登记](../workspace/authentication.md)、[能力层](../workspace/capability.md) | 身份模型、访问边界与调用面的当前事实 |
| [身份治理](../governance/security/identity-and-execution-safety.md) | 跨版本的身份与执行安全目标、成熟度与工作包 S0–S4 |
| [身份与访问体系现状调查](../governance/baselines/identity-access-current-state-2026-11-05.md) | 冻结的现状证据：各入口的身份来源、授权点、资源授权的实际效果、补丁与例外；第 5 节理由的依据 |
| [第五代架构 Idea](./fifth-generation-architecture.md) | 3.5、3.6 的理念 |
| [外部 Actor Idea](./external-actor-registration-and-runtime-access.md) | P-1c、P-9c 的决定与复核；controller 模式的身份需求（E-3b、绑定线程的操作适配器） |
| [执行线程](../workspace/execution-threads.md)（已归档的[执行线程 Idea](../archive/ideas/execution-thread-definition.md)） | 操作的发起者由执行线程给出；线程的目标现为驻留 Workspace，跨 Workspace 目标的表达见第 7 节 T-5b |
| [执行环境 Idea](./execution-environment.md) | 前提 9 与 agent、执行环境的分别选择 |
| [AE2 Idea](./ae2-hivememory-architecture-analogy.md) | 子网与跨 Workspace 的更完整形态 |
