---
title: 模型级密钥覆盖写入受版本控制的 models.yaml
status: todo
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
  - docs/archive/todo/config-secrets-written-to-yaml.md
last_reviewed: 2026-11-01
---

# 模型级密钥覆盖写入受版本控制的 models.yaml

## 问题与证据

模型定义（`ModelDefinition`）带有可选的 `api_key` 与 `api_base`，作为单个模型的“高级覆盖”：解析模型凭证时，模型自身的值优先于 Provider 注册表（`ModelRegistry._resolve_credentials`）。

- 前端模型页面在展开“高级覆盖”后提供 `api_key` 输入，经 `/api/v1/models` 创建或更新模型；
- 注册表把模型定义原子写回 `configs/models.yaml`，这份文件受版本控制，覆盖的密钥因此以明文写入；
- 配置指南与前端管理视图只提醒不要用这个入口保存真实密钥，接口本身不阻止。

2026-11-01 核对时 `configs/models.yaml` 中没有模型写入 `api_key`。同日主配置的密钥已移出 `config.yaml`（[归档 Todo](../archive/todo/config-secrets-written-to-yaml.md)），模型级覆盖是剩下的、能经界面把密钥写进受版本控制文件的入口。

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
