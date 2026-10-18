---
title: 构造根配置会改写进程默认语言
status: todo
owner: system
scope: process-default-language-side-effect-of-config-construction
code_paths:
  - src/hivememory/config/app.py
  - src/hivememory/server/routers/config.py
  - src/hivememory/i18n/resolver.py
  - src/hivememory/system/system.py
related_docs:
  - docs/system/i18n.md
  - docs/system/configuration.md
  - docs/archive/plans/v0.7.0-cpu-prompt-compilation.md
last_reviewed: 2026-10-17
---

# 构造根配置会改写进程默认语言

## 问题与证据

`HiveMemoryConfig` 的模型校验器 `sync_i18n_default_language`（`config/app.py`）在每次构造时调用 `set_default_language(self.i18n.default_language)`，把进程级默认语言改成该对象的值（[i18n](../system/i18n.md)第 42 行按现状记录了这一行为）。启动时加载运行配置需要这一步，但运行中构造的临时配置对象也会触发它：

- `GET /config/defaults` 经 `_build_config_without_path_override()` 构造一份不读取 YAML 的默认配置，只为返回默认值，却把进程默认语言重置为代码默认值（未设置环境变量时为 `zh`）；
- `POST /config` 构造提交的配置用于校验与持久化，持久化之前进程默认语言已经改变，持久化失败也不会恢复。

2026-10-17 复现：先 `set_default_language("en")`，再构造一次 `HiveMemoryConfig()`，`resolve_language()` 由 `en` 变为 `zh`。该校验器自 v0.5.1（`778604c5`）起存在。

## 影响范围

进程默认语言是未显式传入语言的文本的回落值。改写后，每次调用时才解析默认语言的文本随之改变，直到重启或下一次构造配置：

- 任务进程编译的 `RETRIEVAL_CONTEXT`：CPU 分配调用 `MemoryCompiler` 时不传语言，记忆段按调用时的进程默认语言渲染；
- 其他未显式传入语言的 i18n 文本；
- Profile 语言为空或无法识别时解析出的提示词语言（`CPUInputManifest.language`；未声明时 Profile 模型默认 `zh`，不回落到进程默认值）。

在构造时读取一次默认语言的组件（Gateway 的查询理解与话题路由、记忆生成的抽取器）仍保持启动时的语言。因此部署在配置文件中设为英文时，任何人打开一次“默认配置”页面，同一进程内的文本就会混用两种语言，同一会话的前后轮次也可能不一致。

可能的处理：把设置默认语言从模型校验器移到组合根，只在加载运行配置时执行一次；配置路由构造的临时对象不产生全局副作用；若 `POST /config` 需要让新语言立即生效，由保存成功后的显式步骤设置，并与配置热更新的范围一致。

## 完成条件

- 构造 `HiveMemoryConfig` 不改变进程默认语言；运行中的默认语言只在启动加载运行配置时（以及明确的热更新步骤中）设置；
- 回归覆盖：构造默认配置、调用 `GET /config/defaults` 与校验失败的 `POST /config` 之后，`resolve_language()` 保持原值；
- [i18n](../system/i18n.md)与 [System 配置](../system/configuration.md)中关于默认语言同步时机的描述随实现更新。

## 追踪

- 2026-10-17：提示词编译前移收尾审查时发现（[归档计划](../archive/plans/v0.7.0-cpu-prompt-compilation.md)实施记录），不属于该计划范围；未绑定版本或 Issue。
