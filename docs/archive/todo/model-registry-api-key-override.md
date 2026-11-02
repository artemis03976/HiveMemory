---
title: 模型级密钥覆盖写入受版本控制的 models.yaml
status: archived
archived_at: 2026-11-01
implemented_by: PR #166
superseded_by: docs/system/configuration.md
owner: system
scope: model-registry-per-model-api-key-override-persisted-to-tracked-file
code_paths:
  - src/hivememory/core/models/model_definition.py
  - src/hivememory/system/model_registry.py
  - src/hivememory/server/routers/models.py
  - src/hivememory/server/models/model_registry.py
  - frontend/src/components/settings/ModelRegistrySettings.tsx
  - configs/models.yaml
related_docs:
  - docs/help/configuration.md
  - docs/frontend/management-views.md
  - docs/system/configuration.md
  - docs/archive/todo/config-secrets-written-to-yaml.md
last_reviewed: 2026-11-01
---

# 模型级密钥覆盖写入受版本控制的 models.yaml

> 状态：Archived（2026-11-01，已修复，处置见文末一节）。当前事实见 [System 配置与注册表](../../system/configuration.md#4-配置所有权与兼容边界)第 4 节与[配置指南](../../help/configuration.md#4-配置-model-registry)第 4 节。以下为登记时的原文，相对链接已随归档改正。

## 问题与证据

模型定义（`ModelDefinition`）带有可选的 `api_key` 与 `api_base`，作为单个模型的“高级覆盖”：解析模型凭证时，模型自身的值优先于 Provider 注册表（`ModelRegistry._resolve_credentials`）。

- 前端模型页面在展开“高级覆盖”后提供 `api_key` 输入，经 `/api/v1/models` 创建或更新模型；
- 注册表把模型定义原子写回 `configs/models.yaml`，这份文件受版本控制，覆盖的密钥因此以明文写入；
- 配置指南与前端管理视图只提醒不要用这个入口保存真实密钥，接口本身不阻止。

2026-11-01 核对时 `configs/models.yaml` 中没有模型写入 `api_key`。同日主配置的密钥已移出 `config.yaml`（[归档 Todo](./config-secrets-written-to-yaml.md)），模型级覆盖是剩下的、能经界面把密钥写进受版本控制文件的入口。

## 影响范围

- 在模型页面填写一次覆盖密钥，提交仓库时随 `models.yaml` 泄露；
- 同一个提供商的密钥可能分散在 Provider 凭证与多个模型定义中，模型自身的值优先，排查时不容易看出实际生效的是哪一个。

可能的处理（未决定）：

- 删除模型级的 `api_key` 覆盖，密钥只经 Provider 凭证提供；需要单个模型用不同密钥时，另登记一个 provider 名。`api_base` 不是密钥，可以保留覆盖；
- 保留覆盖，但存入不受版本控制的文件（例如与 `providers.secrets.yaml` 一样 gitignored），`models.yaml` 只保存非敏感字段；
- 与 `config.yaml` 一样，`models.yaml` 中出现非空 `api_key` 时加载失败。

## 完成条件

- 经模型页面或 `/api/v1/models` 的操作不会把密钥写入受版本控制的文件；
- 在模型注册表与模型路由的测试层覆盖上述行为；
- 配置指南与前端管理视图改写对应的提醒。

## 处置（2026-11-01）

owner 选择第一种处理：删除模型级的 `api_key` 覆盖，密钥只经 Provider 凭证提供，`api_base` 保留覆盖。

- `ModelDefinition` 删除 `api_key`；`ModelRegistry` 解析的密钥只来自模型 `provider` 对应的 Provider 凭证，`api_base` 仍是模型覆盖优先于 Provider；
- 模型 API 的创建与更新请求删除 `api_key`，响应删除 `api_key_masked`，更新时以空字符串清除密钥的特例随之删除；前端模型页面的高级覆盖只保留 API 地址，说明改为“需要不同密钥时另登记一个 Provider”；
- `configs/models.yaml` 删除 `api_key: null` 与对应的字段说明；`models.yaml` 中写入非空 `api_key` 的模型在加载时被跳过，日志指出模型 ID 与密钥应在的位置，只含 `api_key: null` 的旧文件照常加载，下次保存时这一字段被移除；
- 测试：[模型注册表的单元测试](../../../tests/unit/system/test_model_registry.py)覆盖写入密钥的模型被跳过、`api_key: null` 照常加载（修复前这一用例失败），以及模型覆盖 `api_base` 时密钥仍取 Provider 凭证；原“模型自身的密钥优先”与路由层的脱敏、清除密钥三个测试只守护旧契约，删除。路由层不能写入密钥由请求模型与 `ModelDefinition` 不再有这个字段保证，没有另写运行时测试；
- 文档：[System 配置与注册表](../../system/configuration.md#4-配置所有权与兼容边界)第 4 节写明模型定义没有密钥并加入升级注意，[配置指南](../../help/configuration.md#4-配置-model-registry)第 4、7 节与[管理页面](../../frontend/management-views.md#32-model-registry)第 3.2 节改写对应的提醒。

已知限制：被跳过的模型不在注册表中，修复 `models.yaml` 之前经模型页面保存任何模型，都会让它从文件中消失；这是注册表对所有无效模型定义的既有行为。
