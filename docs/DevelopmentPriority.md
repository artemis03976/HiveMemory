# HiveMemory 系统开发优先级规划
## Development Priority & Defect Assessment

**文档状态**: Active
**评估日期**: 2026-03-08
**适用阶段**: Phase 4.5 → Phase 5 过渡期

---

## 一、当前系统健康度评估

### 1.1 测试状态总览

| 测试类型 | 通过 | 失败 | 错误 | 备注 |
|:---|:---|:---|:---|:---|
| 单元测试 | 1143 / 1148 | 3 | 0 | 覆盖率 76% |
| 集成测试 | 73 / 73 | 0 | 0 | 全部通过 |
| E2E 测试 | 0 / 134 | 0 | 2 (Import) | 2 个文件因 Import 错误无法收集 |

### 1.2 已发现缺陷清单

#### BUG-1: Koakuma L2 冷查找错误消息不一致 [低]
- **位置**: `patchouli/kernel/koakuma.py` → `_resolve_alias_l2()`
- **现象**: 当 `sys_` 前缀的未知工具触发 L2 冷查找时，因 alias 不是合法 UUID 导致底层抛出 `badly formed hexadecimal UUID string` 异常，返回的错误消息为 "L2 alias lookup failed" 而非预期的 "not found"。
- **影响**: Agent 收到的错误提示不够友好，无法正确引导其使用 SEARCH 指令。
- **修复建议**: 在 `_resolve_alias_l2()` 入口处增加 `sys_` 前缀的快速拒绝逻辑，直接返回 "Tool not found" 而非尝试 UUID 解析。

#### BUG-2: E2E 测试 Import 错误 — `Observation` 类已移除 [中]
- **位置**: `tests/e2e/pipeline/test_cold_path_e2e.py`, `tests/e2e/pipeline/test_pre_retrieval_e2e.py`
- **现象**: 两个 E2E 测试文件尝试从 `patchouli.protocol.models` 导入 `Observation` 和 `EyeGazeResult`，但这些类在感知层重构中已被移除/重命名。
- **影响**: 2 个 E2E 测试套件（共 134 个用例）完全无法运行。
- **修复建议**: 更新测试文件的 import 路径，使用重构后的新数据模型。

#### BUG-3: System Chat 测试断言与实现不匹配 [低]
- **位置**: `tests/unit/system/test_chat_logic.py`
- **现象**: `test_no_memory_no_injection` 和 `test_empty_mtp_prompt_no_injection` 两个测试的断言基于旧版 `PatchouliSystem.chat()` 的消息组装逻辑，与当前实现的消息顺序不一致。
- **影响**: 2 个单元测试失败，但不影响实际功能。
- **修复建议**: 更新测试断言以匹配当前 `system.py` 中 `_assemble_messages()` 的实际行为。

### 1.3 已知技术债务

| 编号 | 模块 | 描述 | 严重度 |
|:---|:---|:---|:---|
| TD-1 | `perception/relay_controller.py` | LLM 驱动的 Relay 摘要未实现，当前使用占位符 | 中 |
| TD-2 | `lifecycle/archiver.py` | 仅实现了 FileBasedArchiver，DBBasedArchiver 为 TODO | 低 |
| TD-3 | `generation/deduplicator.py` | Lifecycle 事件记录标记为 TODO | 低 |
| TD-4 | `librarian_core.py` | `start_gardening()` 定时维护方法为空桩 | 低 |
| TD-5 | `gateway/interceptors.py` | 系统指令硬编码，未从全局配置读取 | 低 |

### 1.4 模块就绪度矩阵

| 模块 | 实现度 | 测试覆盖 | 生产就绪 |
|:---|:---|:---|:---|
| Gateway Engine (The Eye) | 95% | ✅ 高 | ✅ |
| Perception Engine (STM/MMU) | 90% | ✅ 高 | ✅ (TD-1 除外) |
| Generation Engine (Librarian) | 95% | ✅ 高 | ✅ |
| Retrieval Engine | 95% | ✅ 高 | ✅ |
| Lifecycle Engine | 90% | ✅ 高 | ✅ (TD-2 除外) |
| MTP Protocol & Koakuma | 95% | ✅ 高 | ✅ |
| Patchouli Kernel | 95% | ✅ 中 | ✅ |
| SystemBus | 95% | ✅ 高 | ✅ |
| Worker Agent | 100% | ✅ 高 | ✅ |
| Infrastructure (LLM/Embed/Store) | 95% | 中 | ✅ |
| Client API | 95% | 中 | ✅ |

---

## 二、开发优先级规划

基于当前系统状态和 Phase 5 (Web 集成) 目标，按优先级排列如下：

### P0: 缺陷修复与测试修复 (立即执行)

> 在进入新功能开发前，确保现有系统的测试基线完全绿色。

1. **修复 BUG-1**: Koakuma L2 冷查找对 `sys_` 前缀的处理
2. **修复 BUG-2**: 更新 E2E 测试的 import 路径，恢复 134 个 E2E 用例的可运行性
3. **修复 BUG-3**: 对齐 `test_chat_logic.py` 的断言与当前实现
4. **补充**: 对 `PatchouliSystem.chat()` 完整链路编写集成级冒烟测试

### P1: FastAPI 后端服务层 (核心优先)

> 将 PatchouliSystem 的能力通过 HTTP API 暴露，这是前端集成的前置条件。

1. **项目结构**: 创建 `src/hivememory/server/` 目录
   - `app.py` — FastAPI 应用入口，生命周期管理 (startup/shutdown)
   - `deps.py` — 依赖注入 (PatchouliSystem 单例)
   - `models/` — Pydantic Request/Response 模型
   - `routers/` — 路由模块

2. **核心 API 路由 (v1)**:
   - `POST /api/v1/chat` — 主动对话接口 (封装 `PatchouliSystem.chat()`)
     - 支持 SSE (Server-Sent Events) 流式响应
     - 返回结构化的 MTP 执行轨迹 (供前端渲染)
   - `POST /api/v1/ingest` — 被动消息摄入 (封装 `ingest()`)
   - `GET /api/v1/topics` — 获取活跃话题列表
   - `POST /api/v1/topics/{id}/trigger` — 手动触发话题结算

3. **记忆管理 API**:
   - `GET /api/v1/memories` — 检索记忆 (支持 query/filter 参数)
   - `GET /api/v1/memories/{id}` — 获取单条记忆详情
   - `DELETE /api/v1/memories/{id}` — 删除记忆

4. **基础设施**:
   - CORS 中间件配置 (允许 Next.js 开发服务器跨域)
   - 简单的 `x-user-id` 头透传 (暂不引入 JWT)
   - 健康检查端点 `GET /health`
   - 请求日志中间件

5. **关键设计决策**:
   - PatchouliSystem 作为 FastAPI 的 `lifespan` 单例初始化
   - Chat 接口使用 SSE 而非 WebSocket (降低前端复杂度，Phase 1 足够)
   - MTP 执行过程中的中间状态通过 SSE event 实时推送

### P2: Next.js 前端核心 (Chat UI)

> 提供可用的图形界面，替代 Streamlit Demo。

1. **项目初始化**:
   - Next.js 15 + App Router
   - TailwindCSS + Shadcn/UI
   - 目录: `web/` (项目根目录下)

2. **Chat 界面核心功能**:
   - 消息流式渲染 (SSE 消费 + 打字机效果)
   - MTP 协议可视化:
     - 隐藏原始 `⟪...⟫` 指令符号
     - 工具调用状态指示器 ("正在搜索记忆...", "正在执行代码...")
     - 折叠/展开工具执行结果
   - 多话题切换 (基于 Topic ID 的侧边栏)

3. **布局结构**:
   - 左侧: 话题列表 (Active Topics)
   - 中间: 聊天主区域
   - 右侧 (可折叠): Debug 面板 — 显示当前 Topic Snapshot、LogicalBlock 结构

### P3: 技术债务清理与系统加固

> 在 Web 集成基本可用后，回头补齐核心引擎的遗留项。

1. **TD-1 修复**: 实现 `relay_controller.py` 中 LLM 驱动的 Relay 摘要
   - 当单话题 Token 溢出时，使用小模型 (如 deepseek-chat) 生成 state_summary
   - 这是长对话场景下的关键能力

2. **TD-4 修复**: 实现 `LibrarianCore.start_gardening()`
   - 定时扫描低活力记忆，触发 GC
   - 集成 APScheduler

3. **E2E 测试恢复**: 全面更新 E2E 测试套件，确保 134 个用例全部可运行

### P4: 高级特性与生产化

1. **WebSocket 升级**: 将 SSE 升级为 WebSocket，支持双向实时通信
2. **用户认证**: 集成 JWT / OAuth2
3. **记忆可视化**: 前端展示记忆节点关联图谱
4. **文件上传**: 集成多模态能力
5. **Docker Compose**: 一键部署 (FastAPI + Next.js + Qdrant + Redis)
6. **DB-based Archiver**: 替代文件归档，提升可扩展性

---

## 三、架构建议

### 3.1 FastAPI ↔ PatchouliSystem 集成模式

```
┌─────────────────────────────────────────────┐
│  Next.js Frontend (web/)                    │
│  ├── SSE Consumer (chat stream)             │
│  ├── REST Client (memories, topics)         │
│  └── MTP Visual Renderer                   │
└──────────────────┬──────────────────────────┘
                   │ HTTP / SSE
┌──────────────────▼──────────────────────────┐
│  FastAPI Server (src/hivememory/server/)     │
│  ├── Lifespan: PatchouliSystem singleton    │
│  ├── POST /chat → system.chat() + SSE      │
│  ├── POST /ingest → system.ingest()         │
│  └── GET /memories → client.retrieve()      │
└──────────────────┬──────────────────────────┘
                   │ In-Process (async)
┌──────────────────▼──────────────────────────┐
│  PatchouliSystem (已有核心)                  │
│  ├── TheEye → Gateway Engine                │
│  ├── Kernel → Worker + Koakuma + Retrieval  │
│  └── Librarian → Perception + Generation    │
└─────────────────────────────────────────────┘
```

### 3.2 SSE 流式响应设计

Chat 接口的 SSE 事件流建议采用以下事件类型：

| Event Type | 数据内容 | 触发时机 |
|:---|:---|:---|
| `token` | `{"content": "..."}` | Worker Agent 生成文本 token |
| `mtp_start` | `{"verb": "READ", "target": "..."}` | MTP 指令被拦截 |
| `mtp_result` | `{"status": "success", "data": "..."}` | MTP 执行完成 |
| `topic_info` | `{"topic_id": "...", "title": "..."}` | 话题路由结果 |
| `done` | `{"usage": {...}}` | 生成完成 |
| `error` | `{"message": "..."}` | 错误发生 |

### 3.3 关键注意事项

1. **异步一致性**: FastAPI 天然支持 async，与 PatchouliSystem 的全异步架构完美契合，无需额外的线程桥接。
2. **单例生命周期**: PatchouliSystem 的初始化较重 (加载模型、连接 DB)，必须在 FastAPI lifespan 中一次性完成。
3. **流式中断**: 当前 `PatchouliSystem.chat()` 返回最终结果，需要改造为 `AsyncGenerator` 以支持 SSE 中间状态推送。这是 P1 阶段最大的改造点。

---

## 四、总结

HiveMemory 核心引擎的实现完成度约 **93%**，系统架构设计成熟，代码质量高。当前存在 3 个已知 Bug (均为低/中严重度) 和 5 项技术债务。

进入 Phase 5 的最大挑战不在于核心能力，而在于：
1. 将同步的 `chat()` 接口改造为支持 SSE 流式输出的 `AsyncGenerator`
2. 前端对 MTP 协议执行过程的可视化渲染

建议严格按照 P0 → P1 → P2 → P3 → P4 的顺序推进，确保每个阶段都有可验证的交付物。
