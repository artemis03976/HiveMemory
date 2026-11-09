---
title: dsh 插件并入统一插件目录
status: archived
archived_at: 2026-11-08
implemented_by: 分支 refactor/dsh-unified-plugin-dir
superseded_by: docs/system/import-bus.md
owner: project
target: v1.0 范围的 plugin 模式接入（不绑定具体版本号，owner 2026-11-04）
scope: dsh-plugin-into-unified-plugin-directory-with-shared-node-core
related_docs:
  - docs/system/import-bus.md
  - docs/contracts/mcp.md
  - docs/help/deepseek-harness-plugin.md
  - docs/archive/plans/pi-plugin-integration.md
code_paths:
  - integrations/hivememory/package.json
  - integrations/hivememory/node/
  - tests/unit/connectors/test_deepseek_harness_plugin.py
  - .github/workflows/ci.yml
updates:
  - docs/system/import-bus.md
  - docs/contracts/mcp.md
  - docs/help/deepseek-harness-plugin.md
  - docs/help/README.md
  - docs/help/setup.md
  - docs/help/troubleshooting.md
  - docs/architecture/overview.md
  - docs/PROJECT.md
  - docs/ROADMAP.md
  - README.md
  - README_EN.md
last_reviewed: 2026-11-08
---

# dsh 插件并入统一插件目录

> 状态：Archived（2026-11-08，迁移、测试与事实文档完成后归档；owner 重新安装到桌面端后在日常使用中确认，见第 8 节）。当前事实见 [Import Bus](../../system/import-bus.md#9-deepseek-harness-插件)第 9 节与[接入 DeepSeek Harness](../../help/deepseek-harness-plugin.md)。以下为实施时的计划原文。原状态：Active（2026-11-08 建立并实施，分支 `refactor/dsh-unified-plugin-dir`）。owner 决定（2026-11-08）：dsh 插件在 pi 插件之后并入统一插件目录 `integrations/hivememory/`，与 pi 扩展共用 Node 公共部分。标为“候选设计”的内容不是 owner 决定。

## 1. 背景和问题

pi 扩展接入时（[归档计划](./pi-plugin-integration.md)），JS 代码按公共部分（`node/core/`）与 pi 专属部分（`node/pi/`）放进了 `integrations/hivememory/`。公共部分是照着 dsh 插件的 `lib/` 改写的：MCP 调用与导入提交及结果分类、原子写入的状态、日志、记忆工具的回声规则与最近轮次的截断。现在仓库里同一份逻辑有两份 JS，dsh 插件仍单独在 `integrations/deepseek-harness/`，使用手册 skill 也要另外复制。

## 2. 目标与非目标

目标：

- dsh 插件搬进 `integrations/hivememory/node/dsh/`，改用 `node/core/` 的配置、客户端、状态、日志与回声规则，删除 `integrations/deepseek-harness/` 中重复的实现。
- `integrations/hivememory/package.json` 同时是 dsh bundle（`main`、`engines.dsh`、`dsh.bundle.patch`）与 pi 包；插件名仍为 `hivememory-connector`。
- dsh 插件的行为不变：回流、水位与补发、记忆工具、子 Agent 的处理都与现在相同；状态目录（`~/.hivememory/dsh-connector`）与其中的水位保留。
- 两个 harness 的 JS 测试合在一处，CI 只跑一个 Node 任务。

非目标：

- 由 bundle 指定 skill 目录、让 dsh 自动加载使用手册：需要先核对 dsh 的 skill 配置项，桌面端的 bundle 打包在 `app.asar` 中，本计划不做，dsh 仍手动复制 skill。
- dsh 的轮次投影、同步规则与记忆工具的行为调整。
- 服务端的契约与接入登记（不变）。

## 3. 设计（候选）

- **目录**：`node/dsh/` 放 dsh 专属部分：入口 `index.js`（Cordis 插件的 `name`、`inject`、`apply`）、`turns.js`（轮次投影）、`sync.js`（同步循环与水位）、`tools.js`（记忆工具）与 bundle 配置 `cordis.patch.yml`。
- **公共部分**：dsh 的取值为 principal `harness:dsh`、来源 `dsh`、默认 Agent `dsh`、凭据变量 `HIVEMEMORY_DSH_TOKEN`、默认状态目录 `~/.hivememory/dsh-connector`、客户端名 `hivememory-dsh-connector`。服务地址的默认值随公共部分改为 `http://127.0.0.1:8769`（原为 `http://localhost:8000`，设置了 `HIVEMEMORY_URL` 时没有区别）。`node/core/state.js` 增加连接器首次运行时间（dsh 用它判断首次见到的会话是否回填）；dsh 的水位登记与读改写留在 `node/dsh/sync.js`。
- **包清单**：`main` 指向 `node/dsh/index.js`，`engines.dsh` 沿用 `>=0.2.1-rc.5 <0.3.0-0`，`dsh.bundle.patch` 指向 `./node/dsh/cordis.patch.yml`；`pi` 字段不变。pi 只读 `pi` 字段，dsh 只读 `main`、`engines.dsh` 与 `dsh` 字段。
- **测试**：dsh 的测试移到 `node/test/`（文件名带 `dsh-` 前缀），与 pi 和公共部分的测试一起由 `npm test` 运行；与公共部分重复的客户端测试删除。Python 一侧的一致性测试改读新位置，记忆工具与 MCP server 一致的检查只保留一处（公共部分）。

## 4. 兼容与迁移

dsh 以 pnpm 的 `link:` 链接插件目录，目录改变后要重新安装一次。桌面端从托盘完全退出后：

```powershell
$dsh = "$env:LOCALAPPDATA\Programs\DeepSeek Harness\resources\runtime\cli\bin\dsh.cmd"
& $dsh plugin --profile desktop add 'D:\Projects\HiveMemory\integrations\hivememory'
```

插件名不变，状态目录与水位保留，重新安装后不会重复导入已经确认的轮次。没有重新安装时，桌面端下次启动会因链接目标不存在而加载失败。

## 5. 测试和验收

| 不变量 | 归属层 |
|:---|:---|
| dsh 的轮次投影、回声标记与结局、写入时的轮次信息与导入一致、水位与补发（行为不变） | `node/test/dsh-*.test.js`（原 dsh 测试随代码迁移） |
| 公共部分按 harness 取值：dsh 读自己的凭据变量 | `node/test/core.test.js` |
| 包清单指向的 dsh 入口、bundle 配置、pi 扩展与 skill 存在；dsh 的 principal 与 Agent 在发布的登记中 | `tests/unit/connectors/test_deepseek_harness_plugin.py` |

```bash
cd integrations/hivememory && npm test
pytest tests/unit/connectors -q
```

dsh 实测：以 dsh 桌面端自带的 CLI 在隔离的 `DSH_HOME` 中从新目录安装插件，对本机模拟服务确认插件加载、注册四个记忆工具；重新安装到已有旧链接的 profile 时链接改到新目录。owner 重新安装到桌面端后，日常使用中确认回流与记忆工具照常。

## 6. 风险

- 一份 `package.json` 同时承担 dsh bundle 与 pi 包：任何一方开始读取对方的字段（例如 pi 检查 `engines`）时会互相影响，升级 dsh 或 pi 后需要核对。
- 桌面端在重新安装之前重启会加载不到插件（第 4 节）。

## 7. 完成时必须更新的当前文档

- [Import Bus](../../system/import-bus.md)：第 2 节组成表、第 9 节 dsh 插件的位置与组成、第 11 节中“不共用”的说法、第 14 节代码与测试入口；
- [MCP 记忆访问](../../contracts/mcp.md) 中 dsh 插件的代码位置；
- 帮助：[接入 DeepSeek Harness](../../help/deepseek-harness-plugin.md) 的安装路径与配置默认值，[帮助索引](../../help/README.md)、[安装与启动](../../help/setup.md)、[故障排查](../../help/troubleshooting.md)；
- [架构概览](../../architecture/overview.md)、[PROJECT](../../PROJECT.md)、README 与 README_EN、[ROADMAP](../../ROADMAP.md)、Plans 索引。

## 8. 实施记录

分支 `refactor/dsh-unified-plugin-dir`（2026-11-08）：dsh 插件的入口、轮次投影、同步与记忆工具以 `git mv` 移到 `integrations/hivememory/node/dsh/`，bundle 配置移到 `node/dsh/cordis.patch.yml`；原 `lib/client.js`、`config.js`、`log.js`、`state.js` 删除，改用 `node/core/`（`core/state.js` 增加读改写与首次运行时间，水位登记留在 `node/dsh/sync.js`）；轮次投影中的工具调用改用公共的回声规则；包清单合并进 `integrations/hivememory/package.json`；dsh 的测试移到 `node/test/dsh-*.test.js`，与公共部分重复的客户端与配置测试删除（按 harness 取凭据的用例并入 `core.test.js`）；CI 的两个 Node 任务合为 `node-connectors`；`tests/unit/connectors/test_deepseek_harness_plugin.py` 改读新位置并检查包清单；第 7 节的事实文档。

**验证**（2026-11-08，Windows 原生）：

- `npm test`（`integrations/hivememory`）：72 passed（dsh、pi 与公共部分）；`pytest tests/unit tests/integration -m "not live_llm and not e2e and not slow" -n auto --cov=hivememory --cov-fail-under=85`：3270 passed，1 skipped，覆盖率 94.85%；`ruff check .`、`black --check .`、`git diff --check` 通过。
- dsh 实测（桌面端自带的 dsh CLI 0.2.1-rc.5，隔离的 `DSH_HOME`，本机模拟的 MCP 服务）：从新目录 `plugin add` 后，`--dump-config` 出现 `hivememory-connector` 配置层，`headless` 启动时插件以 `harness:dsh` 与客户端名 `hivememory-dsh-connector` 请求 `initialize` 与 `tools/list` 并注册四个记忆工具；对已经链接旧目录的 profile 再执行一次 `plugin add`，profile 的依赖改为链接新目录，插件照常加载。没有在 dsh 中运行模型，轮次回流以随代码迁移的测试覆盖。

**待确认**（owner）：桌面端从托盘退出后按第 4 节重新安装，之后在日常使用中确认回流与记忆工具照常。
