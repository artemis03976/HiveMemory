---
title: Gateway 与 Librarian 的旧式直配路径
status: todo
owner: system
scope: component-llm-config-direct-model-and-key-fields-bypassing-registries
code_paths:
  - src/hivememory/config/shared.py
  - src/hivememory/config/app.py
  - src/hivememory/system/model_registry.py
  - src/hivememory/system/assembler.py
related_docs:
  - docs/system/configuration.md
  - docs/archive/todo/config-secrets-written-to-yaml.md
last_reviewed: 2026-11-01
---

# Gateway 与 Librarian 的旧式直配路径

## 问题与证据

Gateway 与 Librarian 的 LLM 配置（`shared.llm.gateway` / `librarian`，类型 `LLMConfig`）有两种写法：

- 当前写法：`model_id` 引用 `configs/models.yaml` 中的模型，组合根经模型注册表解析出模型名，再从 Provider 注册表补齐 `api_key` 与 `api_base`；
- 旧式写法：不写 `model_id`，直接在配置中给出 `model`、`api_key`、`api_base`。`ModelRegistry.resolve_for_llm_config` 遇到没有 `model_id` 的配置原样返回（代码注释称“向后兼容旧配置”），完全不经过两个注册表。

旧式写法的密钥还可以来自环境变量别名：`LEGACY_ENV_ALIASES` 把 `HIVEMEMORY__LLM__GATEWAY__API_KEY`、`HIVEMEMORY__LLM__LIBRARIAN__API_KEY` 等变量映射到这两段配置，嵌套形式 `HIVEMEMORY__SHARED__LLM__GATEWAY__API_KEY` 也同样生效。

2026-11-01 起 `config.yaml` 中不能再写非空的 `api_key`（[归档 Todo](../archive/todo/config-secrets-written-to-yaml.md)），但经环境变量仍能走旧式写法。随仓库发布的 `config.yaml` 两段都使用 `model_id`。

## 影响范围

- 同一件事有两种配置方式，经环境变量注入的模型与密钥不在 Provider 页面与模型页面中显示，排查时容易漏看；
- `LLMConfig` 同时承担配置文件中的组件配置与注册表解析后的运行时结果两种角色，`api_key` 字段因此留在主配置模型里，主配置接口需要为它打码、在保存时沿用运行时值。

可能的处理（未决定）：

- 组件的 LLM 配置只保留 `model_id` 与组件调参（`temperature`、`max_tokens`），删除 `model`、`api_key`、`api_base` 字段与对应的旧环境变量别名；注册表解析直接产出请求参数（`LLMRequestParams`），不再借用 `LLMConfig`；
- 用旧式写法的部署按 `model_id` 迁移，迁移说明写入 System 配置文档的升级注意。

## 完成条件

- Gateway 与 Librarian 的模型与凭证只经模型注册表与 Provider 注册表取得，配置与环境变量中不再有直接指定模型与密钥的入口；
- 主配置模型中不再有 LLM 的 `api_key` 字段；
- 在配置加载与组合根的测试层覆盖上述行为，删除只守护旧写法的测试（例如新旧环境变量的优先级、没有 `model_id` 时原样返回）。
