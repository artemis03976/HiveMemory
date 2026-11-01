---
title: 主配置保存把密钥写入受版本控制的 YAML
status: todo
owner: system
scope: main-config-save-persists-secrets-into-tracked-config-yaml
code_paths:
  - src/hivememory/server/routers/config.py
  - src/hivememory/config/app.py
  - src/hivememory/config/shared.py
  - configs/config.yaml
related_docs:
  - docs/help/configuration.md
  - docs/system/configuration.md
  - docs/todo/local-http-management-access.md
last_reviewed: 2026-11-01
---

# 主配置保存把密钥写入受版本控制的 YAML

## 问题与证据

`POST /api/v1/config` 用 `HiveMemoryConfig` 校验提交内容。校验时，配置来源除了提交内容，还会合入环境变量与 `.env`：`HIVEMEMORY__PROVIDERS__<名称>__API_KEY`、`LLM__GATEWAY__API_KEY`、`LLM__LIBRARIAN__API_KEY` 与 `QDRANT__API_KEY`。之后它把校验结果的完整导出原子写入 `configs/config.yaml`，这份文件受版本控制。

2026-11-01 核对：只在环境变量中设置的提供商密钥，出现在校验结果的导出中，保存时会被写入 YAML。当时仓库中的 `configs/config.yaml` 没有非空的 `api_key`。

同日 `/api/v1/config` 的响应改为只返回打码值，提交时打码值还原为当前密钥（[配置指南](../help/configuration.md#5-主配置与生效边界)）。这只解决了响应泄露明文密钥的问题：还原之后写入 YAML 的仍是真实密钥。在此之前，前端会把读到的明文密钥原样提交回来，结果相同。

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
