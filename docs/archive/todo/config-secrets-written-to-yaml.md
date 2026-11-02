---
title: 主配置保存把密钥写入受版本控制的 YAML
status: archived
archived_at: 2026-11-01
implemented_by: PR #159
superseded_by: docs/system/configuration.md
owner: system
scope: main-config-save-persists-secrets-into-tracked-config-yaml
code_paths:
  - src/hivememory/server/routers/config.py
  - src/hivememory/config/app.py
  - src/hivememory/config/shared.py
  - src/hivememory/system/assembler.py
  - configs/config.yaml
related_docs:
  - docs/help/configuration.md
  - docs/system/configuration.md
  - docs/todo/local-http-management-access.md
last_reviewed: 2026-11-01
---

# 主配置保存把密钥写入受版本控制的 YAML

> 状态：Archived（2026-11-01，已修复，处置见文末一节）。当前事实见 [System 配置与注册表](../../system/configuration.md)第 2–4 节与[配置指南](../../help/configuration.md#5-主配置与生效边界)第 5 节。以下为登记时的原文，环境变量名已改正。

## 问题与证据

`POST /api/v1/config` 用 `HiveMemoryConfig` 校验提交内容。校验时，配置来源除了提交内容，还会合入环境变量与 `.env`：`HIVEMEMORY__PROVIDERS__<名称>__API_KEY`、`HIVEMEMORY__LLM__GATEWAY__API_KEY`、`HIVEMEMORY__LLM__LIBRARIAN__API_KEY` 与 `HIVEMEMORY__QDRANT__API_KEY`。之后它把校验结果的完整导出原子写入 `configs/config.yaml`，这份文件受版本控制。

2026-11-01 核对：只在环境变量中设置的提供商密钥，出现在校验结果的导出中，保存时会被写入 YAML。当时仓库中的 `configs/config.yaml` 没有非空的 `api_key`。

同日 `/api/v1/config` 的响应改为只返回打码值，提交时打码值还原为当前密钥（[配置指南](../../help/configuration.md#5-主配置与生效边界)）。这只解决了响应泄露明文密钥的问题：还原之后写入 YAML 的仍是真实密钥。在此之前，前端会把读到的明文密钥原样提交回来，结果相同。

问题是在 dsh 接入的试用中发现的。dsh 中的 agent 从本机读取 `/api/v1/config`，取得了明文的 DeepSeek 密钥。这个密钥随工具结果进入了 dsh 的会话记录，也进入了待导入 HiveMemory 的一轮。owner 决定先轮换密钥，再让这一轮照常导入。

## 影响范围

- 在前端主配置页面保存过一次，`configs/config.yaml` 中就可能出现 `.env` 里的全部密钥，提交仓库时随之泄露；
- 配置指南目前只提醒保存后检查 YAML，不阻止写入。

可能的处理（未决定）：

- 写入 YAML 时去掉所有 `api_key`。密钥只来自 `.env`、环境变量与 Provider Registry，这与 `ProviderCredentials` 的约定一致（`config.yaml` 中留空）；代价是原本写在 `config.yaml` 中的密钥会在保存后丢失；
- 只写入提交中显式改动的字段，不写入从其他来源合入的值；需要区分字段来源。

## 完成条件

- 经 `POST /api/v1/config` 保存后，`configs/config.yaml` 不含来自环境变量或 `.env` 的密钥，运行中的配置仍持有这些密钥；
- 在配置路由所在的测试层（`tests/unit/server/routers/test_config.py`）补充上述行为的测试；
- 配置指南删去保存后检查 YAML 的提醒，改写为新的保存规则。

## 处置（2026-11-01）

排查发现问题不止于保存时合入环境变量。“密钥只放在 `providers.secrets.yaml` 或 `.env`”的约定在引入模型注册表时（PR #69，2026-07）定下，主配置接口更早（2026-03）就已存在，此后没有随之调整：

1. 组合根把注册表解析出的 Gateway、Librarian 配置写回运行时配置对象，其中带有从 Provider 注册表取到的真实 `api_key`，并丢掉了 `model_id`。dsh 的 agent 读到的密钥就来自这里；主配置页面一旦保存，`config.yaml` 会写入明文密钥与解析出的模型名，两个组件从此走“没有 `model_id`”的兼容路径，脱离模型注册表。
2. 保存时把环境变量与 `.env` 合入的值一并写入 `config.yaml`。
3. `config.yaml` 中的 `shared.providers.deepseek`（只有 `api_base`）与环境变量合并后作为 Provider 注册表的环境层，环境层整条优先，经 Provider 页面写入 `providers.secrets.yaml` 的同名凭证因此不生效。

owner 选定的范围（2026-11-01）是修复 1、2，增加加载时的拒绝，并把提供商凭证收归注册表：

- 组合根不再把解析结果写回配置对象，解析结果只用于构造组件的请求参数；
- 主配置接口不读写密钥：响应打码，保存时提交中的 `api_key` 一律沿用运行时值，写入 YAML 的内容不含 `api_key`；
- `config.yaml` 中出现非空 `api_key` 时加载失败，错误指出字段路径与密钥应在的位置；
- `shared.providers` 从主配置删除，提供商凭证由 `provider_credentials_from_env()` 从环境变量与 `.env` 读取后直接交给 Provider 注册表；`config.yaml` 中 DeepSeek 的 `api_base` 是提供商的默认地址，随这一段删除。

未纳入本次范围，各自登记为 Todo：Gateway、Librarian 的旧式直配路径（`model`、`api_key`、`api_base` 字段与 `HIVEMEMORY__LLM__*__API_KEY` 别名，2026-11-01 升级为[计划](../../plans/llm-config-retirement.md)），以及模型注册表“高级覆盖”把模型级密钥写入受版本控制的 `models.yaml`（[已归档](./model-registry-api-key-override.md)）。
