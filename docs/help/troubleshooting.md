---
title: HiveMemory Troubleshooting
status: current
owner: project
scope: common-runtime-and-frontend-failures
code_paths:
  - docker/docker-compose.yml
  - src/hivememory/server/app.py
  - src/hivememory/server/management_credential.py
  - frontend/vite.config.ts
  - src/hivememory/config/
  - frontend/src/
  - integrations/hivememory/
  - integrations/hivememory/node/dsh/
related_contracts:
  - docs/contracts/error-model.md
  - docs/contracts/mcp.md
last_reviewed: 2026-11-09
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

从其他机器访问不到时，注意后端默认只监听 `127.0.0.1`，Docker Compose 只在宿主的本机地址上开放 `8000`（[安装与启动](./setup.md#43-启动后端)）。

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

`/health` 成功而 `/health/ready` 返回 `warming_up`，表示服务已经接受 HTTP，但 Embedding 或启用的 Reranker 尚未成功加载过，或空闲卸载之后的重新加载失败了。模型空闲卸载本身不会让它变为 `warming_up`。

检查：

- 首次下载是否仍在进行；
- 网络是否能访问模型源，必要时设置可信的 `HF_ENDPOINT`；
- 模型 cache 目录是否有写权限和足够空间；
- 配置中的 embedding/reranker model name 是否有效；
- 后端日志是否出现模型加载异常或 `推理模型预取失败`。

空闲卸载之后的第一次检索或记忆生成会先重新加载模型，单个模型约需 1～3 秒，这期间其他请求与流式输出可能停顿数秒；第一次检索还可能跳过重排，日志出现 `重排模型未加载` 的 warning，之后的检索恢复重排。这是空闲卸载的预期代价；内存充足时可以把驻留方式设为 `resident`（[配置](../system/configuration.md)）。外部脚本不应再用 `/health/ready` 判断模型是否已在内存中，它只表示服务可用。

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

上下文相关失败会在 Chat 中给出具体提示，稳定原因见[错误模型](../contracts/error-model.md#22-agent-run)：当前输入过大时缩短消息、减少附件或检索材料；较早的对话无法压缩进预算时可重试或新建会话（关闭压缩时只能新建会话或重新开启 `alice.compaction.enabled`）；轮内工具结果超预算时可继续对话或缩小任务；提供商超窗时核对模型登记的窗口与生成上限，减少输入或选择更大窗口。字符估算与提供商计数可能不同，不应仅为消除提示而虚增窗口。

压缩会在部分长会话轮次开始时增加等待，界面显示进度；原文不会被压缩标记隐藏或删除。标记刷新失败可重新选择会话；`agent.context.*` 只用于观测，是否接受检查点以会话 GET 为准。配置说明见[会话压缩配置](./configuration.md#52-alice-会话压缩)。

## 5. 前端 API、SSE 或 WebSocket 失败

开发模式确认：

- Vite 在 `127.0.0.1:5173`；
- 后端在 `localhost:8769`；
- `frontend/vite.config.ts` 的 `/api` proxy 未被改到其他地址。

整合部署确认浏览器与 API 同源 `8000`。如果使用自定义反向代理，必须同时转发普通 HTTP、SSE 长连接和 `/api/v1/ws/logs` WebSocket upgrade。

设置 `VITE_BACKEND_ORIGIN` 后，前端的 HTTP、SSE 与日志 WebSocket 都直连这个 origin，不经 Vite 代理，也就不带管理凭据，会被拒绝；页面 origin 不在后端 CORS 允许的本地开发 origin 中时，浏览器还会拦下跨域请求。日常使用不设置它。

接口返回 401 `request.management_credential_invalid`，或日志 WebSocket 连接被拒绝，说明请求没有带上一致的管理凭据：

- 开发模式：确认后端已启动、`configs/management.token` 已生成，且 Vite 读得到它（Vite 从 `frontend` 目录以 `../configs/management.token` 读取）；修改 `vite.config.ts` 之后需要重启 Vite；
- 后端提供页面时：重新加载页面，让后端重新下发凭据 cookie；后端重新生成过凭据时同样如此；
- 在终端中调用：带上 `Authorization: Bearer $(cat configs/management.token)`（[安装与启动](./setup.md#43-启动后端)）。

## 6. 页面显示了数据，但写操作全部失败

Memory、Agent、Settings 和 Memory task 的 mock fallback 只在 debug 模式下启用，即前端以 `VITE_DEBUG=true` 启动或构建时（[状态与传输](../frontend/state-and-transports.md#6-mock-降级边界)第 6 节）。默认不启用：后端读取失败时 Memory 列表显示错误，Agent 页提示“Agent 列表读取失败，请检查后端服务后刷新”，Settings 显示“配置加载失败”。

debug 模式下的典型迹象：

- 后端日志没有对应 list 请求成功记录；
- Memory 显示固定示例内容，页面却没有错误；
- Agent 出现演示 Profile；
- Settings 显示旧版完整配置，但保存报错；
- task 面板显示固定任务。

debug 模式的页面没有 mock 标识。验证后端能力时不要设置 `VITE_DEBUG`；确认真实后端可达并重新加载页面后再判断数据是否存在，不要在 mock 页面上执行删除或编辑。

## 7. Settings 某些分类空白或报错

当前前端主配置类型仍是旧扁平结构，后端 `/api/v1/config` 已使用新的嵌套配置树，多组分类尚未对齐。Provider 与 Model Registry 使用独立 API，通常不受此问题影响。

在对齐完成前：

- 直接编辑 `configs/config.yaml`；
- 或使用能够提交完整 `HiveMemoryConfig` 新结构的 API 客户端；
- 保存后重启并重新检查 readiness；
- 用版本控制 diff 确认没有意外删除未出现在旧表单中的配置段。

## 8. Chat 看不到会话历史

会话历史只在后端进程内保存，浏览器只保存当前会话 ID 与 Agent ID：

- 后端重启后旧会话不存在，读取历史返回 404；前端清除失效选择，此时需要新建会话；
- 网络或服务错误不会当作空历史，页面保留错误并禁止发送，可重新选择会话重试；
- 在另一个窗口运行的同一会话会返回 busy，等该任务进程结束后再继续；
- 会话与 Topic 独立，MemoryLibrary 的短期话题池用于整理记忆材料，不负责恢复聊天。

本轮回复的 Topic 标注来自 `done.topic_id`，重新加载时来自会话轮次。会话 CRUD 与历史契约见[公开路由与事件](../contracts/routes-and-events.md#61-会话管理与-chat-sse)。

## 9. Workspace Terminal 没有日志或事件

检查：

- `logging.websocket_enabled` 是否开启；
- logger namespace 和 level 是否被配置过滤；
- `runtime_events.enabled` 是否开启；
- WebSocket/SSE 是否被代理；
- 当前窗口是否为主窗口，或 BroadcastChannel 是否可用；
- 浏览器 localStorage 中的主窗口 heartbeat 是否因异常关闭短暂未过期。

日志和 RuntimeEvent 都是 best-effort 观测旁路。它们缺失不自动代表业务失败，业务结果仍应检查 HTTP/SSE 终态和后端状态。

## 10. 外部 harness 插件：工具不出现、导入停止或被拒

各插件的现象与处理见对应手册的排查一节：[接入 Claude Code](./claude-code-plugin.md#9-排查)、[接入 DeepSeek Harness](./deepseek-harness-plugin.md#8-排查)、[接入 Codex](./codex-plugin.md#9-排查)、[接入 pi](./pi-plugin.md#8-排查)。

请求返回 403 `agent_not_allowed_for_principal`：harness 声明的 Agent 不在这个来源可声明的 Agent 中，通常是改了 `HIVEMEMORY_AGENT_ID`；在设置页的“接入与权限”中把这个 Agent 加入该来源。设置页的修改被拒并提示重启后端：登记文件在后端运行时被手工改过，重启装载之后再在页面上修改。

管理接口返回 403 `system_actor_not_allowed` 与插件无关：自定义的 `system_principals.yaml` 没有为 server 的登记设置 `allow_system_actor: true`。

## 11. 仍无法定位

收集以下信息后再提交 Issue：

- 运行方式与操作系统；
- 当前 Git commit/tag；
- 访问的端口与 `/health`、`/health/ready` 结果；
- 去除密钥后的配置差异；
- 相关后端日志、浏览器 console/network 错误；
- 前端是否以 debug 模式（`VITE_DEBUG=true`）运行；
- 最小复现步骤与期望/实际结果。

不要把 API key、完整 `.env` 或 `providers.secrets.yaml` 附在 Issue 中。
