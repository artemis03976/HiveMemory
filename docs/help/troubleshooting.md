---
title: HiveMemory Troubleshooting
status: current
owner: project
scope: common-runtime-and-frontend-failures
code_paths:
  - docker/docker-compose.yml
  - src/hivememory/server/app.py
  - src/hivememory/system/config/
  - frontend/src/
related_contracts:
  - docs/contracts/error-model.md
last_reviewed: 2026-07-28
---

# HiveMemory 故障排查

排障前先确认运行形态：Docker 一体化使用 `8000`，本地开发使用后端 `8769` 与前端 `5173`。端口对不上会让后续所有症状看起来像业务错误。

## 1. 服务打不开

先检查：

- Docker：`docker compose -f docker/docker-compose.yml ps`；
- 本地：`hivememory-server` 进程是否仍在运行；
- liveness：Docker 访问 `http://localhost:8000/health`，本地访问 `http://localhost:8769/health`。

若 `/health` 成功但前端打不开：

- Docker 检查镜像是否成功执行 `npm run build`，以及 `HIVEMEMORY_SERVE_FRONTEND=true`；
- 本地开发必须另行启动 Vite，并访问 `http://127.0.0.1:5173`；`8769` 默认只提供 API，不直接提供未构建前端。

## 2. Qdrant 连接失败

```text
docker compose -f docker/docker-compose.yml ps
docker compose -f docker/docker-compose.yml logs qdrant
```

确认：

- Qdrant `/readyz` 健康；
- 本地后端使用 `127.0.0.1:6333/6334`；
- Compose 应用使用服务名 `qdrant:6333/6334`，该值已在 compose environment 中覆盖；
- 自定义 `HIVEMEMORY_CONFIG_PATH` 指向的文件存在；
- Qdrant collection/vector dimension 与 Embedding 输出一致。

## 3. `/health/ready` 长时间 503

`/health` 成功而 `/health/ready` 返回 `warming_up`，表示服务已经接受 HTTP，但 Embedding 或启用的 Reranker 尚未完成加载。

检查：

- 首次下载是否仍在进行；
- 网络是否能访问模型源，必要时设置可信的 `HF_ENDPOINT`；
- 模型 cache 目录是否有写权限和足够空间；
- 配置中的 embedding/reranker model name 是否有效；
- 后端日志是否出现模型加载异常。

Docker 镜像默认尝试在 build 阶段预下载模型；构建失败与运行时 warmup 失败是两个不同阶段。

## 4. LLM 请求失败

按顺序检查：

1. `configs/models.yaml` 中被引用的 model ID 是否存在；
2. `litellm_model` 与 Provider 是否匹配；
3. Provider key 是否通过环境变量或 `providers.secrets.yaml` 提供；
4. 同名环境变量是否覆盖了 UI 中的 YAML 凭证；
5. api base、网络、额度与模型权限是否有效；
6. 修改注册表或主配置后，相关组件是否需要重启。

Provider/Model 列表 API 只返回脱敏 key；看到掩码代表“存在一个值”，不证明凭证可用。

## 5. 前端 API、SSE 或 WebSocket 失败

开发模式确认：

- Vite 在 `127.0.0.1:5173`；
- 后端在 `localhost:8769`；
- `frontend/vite.config.ts` 的 `/api` proxy 未被改到其他地址。

整合部署确认浏览器与 API 同源 `8000`。如果使用自定义反向代理，必须同时转发普通 HTTP、SSE 长连接和 `/api/v1/ws/logs` WebSocket upgrade。

`VITE_BACKEND_ORIGIN` 当前只控制日志与 RuntimeEvent stream，不能改变所有 HTTP API。仅设置它而没有代理 `/api`，Chat、Memory 和 Settings 仍会请求前端当前 origin。

## 6. 页面显示了数据，但写操作全部失败

Memory、Agent、Settings 和开发模式 Memory task 存在 mock fallback。典型迹象：

- 后端日志没有对应 list 请求成功记录；
- Memory 显示固定示例内容，页面却没有错误；
- Agent 出现演示 Profile；
- Settings 显示旧版完整配置，但保存报错；
- task 只在开发环境显示固定任务。

当前 mock 标识并不统一。确认真实后端可达并重新加载页面后再判断数据是否存在；不要在 mock 页面上执行删除或编辑来验证后端能力。

## 7. Settings 某些分类空白或报错

当前前端主配置类型仍是旧扁平结构，后端 `/api/v1/config` 已使用新的嵌套配置树，多组分类尚未对齐。Provider 与 Model Registry 使用独立 API，通常不受此问题影响。

在对齐完成前：

- 直接编辑 `configs/config.yaml`；
- 或使用能够提交完整 `HiveMemoryConfig` 新结构的 API 客户端；
- 保存后重启并重新检查 readiness；
- 用版本控制 diff 确认没有意外删除未出现在旧表单中的配置段。

## 8. Chat 看不到历史或 Topic 切换无效

这是当前能力边界，不是数据加载延迟：

- 浏览器不持久化 messages 和 `currentTopicId`；
- 后端没有 session/message history API；
- 点击左侧 Topic 只改变标题和本地高亮，不会加载历史，也不会指定下一轮后端 Topic。

本轮真实 Topic 以 SSE `topic_info` 为准。若要调查话题资产，请使用 Topic API 与 Patchouli 当前文档，而不是把左侧栏当作会话历史浏览器。

## 9. Kernel Terminal 没有日志或事件

检查：

- `logging.websocket_enabled` 是否开启；
- logger namespace 和 level 是否被配置过滤；
- `runtime_events.enabled` 是否开启；
- WebSocket/SSE 是否被代理；
- 当前窗口是否为主窗口，或 BroadcastChannel 是否可用；
- 浏览器 localStorage 中的主窗口 heartbeat 是否因异常关闭短暂未过期。

日志和 RuntimeEvent 都是 best-effort 观测旁路。它们缺失不自动代表业务失败，业务结果仍应检查 HTTP/SSE 终态和后端状态。

## 10. 仍无法定位

收集以下信息后再提交 Issue：

- 运行方式与操作系统；
- 当前 Git commit/tag；
- 访问的端口与 `/health`、`/health/ready` 结果；
- 去除密钥后的配置差异；
- 相关后端日志、浏览器 console/network 错误；
- 是否出现 mock fallback；
- 最小复现步骤与期望/实际结果。

不要把 API key、完整 `.env` 或 `providers.secrets.yaml` 附在 Issue 中。
