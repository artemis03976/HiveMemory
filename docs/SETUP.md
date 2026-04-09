# HiveMemory 环境搭建指南

本文档面向当前 `v0.1.0` 测试版，说明如何从零开始搭建 HiveMemory 的本地开发 / 运行环境。

如需快速了解项目整体功能与架构，请先阅读 [README.md](../README.md)。

---

## 1. 前置要求

### 必需工具

- **Python 3.12+**
- **Git**
- **Docker / Docker Compose**
- **Node.js**（用于前端开发界面，建议使用较新的 LTS 版本）

### 推荐硬件

- **内存**：至少 8GB RAM
- **磁盘**：至少 5GB 可用空间（模型缓存、数据库数据、依赖安装）
- **GPU**：可选；如需加速 Embedding / Reranker，可使用 CUDA 环境

> 首次运行时，Embedding 与 Reranker 模型可能需要下载与预热，因此初次启动会明显慢于后续启动。

---

## 2. 克隆仓库

```bash
git clone https://github.com/artemis03976/HiveMemory.git
cd HiveMemory
```

---

## 3. 启动基础设施

当前项目本地依赖以下基础设施：

- **Qdrant**：向量数据库
- **Qdrant Web UI**：可选，仅用于调试可视化

### 启动 Qdrant（与后端应用）

```bash
docker-compose -f docker/docker-compose.yml up -d
```

### 可选：同时启动 Qdrant Web UI

```bash
docker-compose -f docker/docker-compose.yml --profile debug up -d
```

### 端口说明

- HiveMemory Web: `8000`
- Qdrant HTTP: `6333`
- Qdrant gRPC: `6334`
- Qdrant Web UI: `6335`（debug profile）

### 检查容器状态

```bash
docker ps
```

## 4. 创建 Python 虚拟环境

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 5. 安装后端依赖

当前项目已采用 `pyproject.toml` 作为包管理入口，**不要再使用旧的 `requirements.txt` 方式作为主安装路径**。

### 运行项目所需的最小安装

```bash
pip install -e .
```

### 如需运行测试与开发工具

```bash
pip install -e ".[dev]"
```

### 可选：如果模型下载较慢

中国大陆网络环境下可尝试配置 HuggingFace 镜像：

```bash
# Windows PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"

# Linux / macOS
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 6. 配置环境变量

先复制模板文件：

```bash
cp configs/.env.example .env
```

当前配置采用 **环境变量 + YAML** 分层模式：

- `.env`：放 API Key、Qdrant 地址、调试开关等
- `configs/config.yaml`：放业务逻辑参数、检索参数、感知参数、生命周期参数等

### 当前环境变量格式

环境变量统一使用 `HIVEMEMORY__` 前缀和双下划线层级分隔，例如：

```env
HIVEMEMORY__LLM__WORKER__MODEL=deepseek/deepseek-chat
HIVEMEMORY__LLM__WORKER__API_KEY=your_worker_api_key
HIVEMEMORY__LLM__WORKER__API_BASE=https://api.deepseek.com

HIVEMEMORY__LLM__LIBRARIAN__MODEL=deepseek/deepseek-chat
HIVEMEMORY__LLM__LIBRARIAN__API_KEY=your_librarian_api_key
HIVEMEMORY__LLM__LIBRARIAN__API_BASE=https://api.deepseek.com

HIVEMEMORY__QDRANT__HOST=localhost
HIVEMEMORY__QDRANT__PORT=6333

HIVEMEMORY__REDIS__HOST=localhost
HIVEMEMORY__REDIS__PORT=6379
HIVEMEMORY__REDIS__PASSWORD=
```

### 最低建议检查项

至少请确认以下字段：

- `HIVEMEMORY__LLM__WORKER__API_KEY`
- `HIVEMEMORY__LLM__LIBRARIAN__API_KEY`
- `HIVEMEMORY__QDRANT__HOST`
- `HIVEMEMORY__QDRANT__PORT`

### 配置文件加载说明

当前配置系统支持：

- 仓库根目录 `.env`
- `configs/.env`
- `HIVEMEMORY_CONFIG_PATH` 指定的 YAML 配置文件

默认 YAML 配置文件为：

- `configs/config.yaml`

---

## 7. 启动后端服务

推荐使用项目自带的包脚本：

```bash
hivememory-server
```

当前默认后端地址：

- `http://localhost:8769`

### 启动后检查

```bash
curl http://localhost:8769/health
curl http://localhost:8769/health/ready
```

接口说明：

- `/health`：服务是否存活
- `/health/ready`：模型是否已完成预热
  - 若仍在后台预热，返回 `503` 和 `warming_up`

> 服务启动成功不代表模型已经完全 ready；初次启动请优先检查 `/health/ready`。

---

## 8. 启动前端开发界面

```bash
cd frontend
npm install
npm run dev
```

当前前端开发服务器默认地址：

- `http://127.0.0.1:5173`

前端开发代理会将 `/api` 转发到：

- `http://localhost:8769`

如果你只想验证后端，也可以跳过此前端步骤，直接调用 HTTP API。

---

## 9. 运行测试

当前项目测试由 `pytest` 驱动，默认配置见 [pyproject.toml](../pyproject.toml)。

### 运行默认测试集

```bash
pytest
```

默认会跳过以下类型测试：

- `live_llm`
- `e2e`
- `slow`

### 只运行单元 / 集成测试

```bash
pytest -m "unit or integration"
```

### 运行前端检查

```bash
cd frontend
npm run lint
npm run build
```

---

## 10. 常见问题排查

### 问题 1：Qdrant 连接失败

**现象**：服务启动时报 Qdrant 连接错误，或检索功能不可用。

**排查方式**：

```bash
docker ps
docker logs hivememory_qdrant
```

确认以下内容：

- Qdrant 容器已正常运行
- Docker Compose 部署时 `HIVEMEMORY__QDRANT__HOST=qdrant`
- 本地开发时 `HIVEMEMORY__QDRANT__HOST=localhost`
- `HIVEMEMORY__QDRANT__PORT=6333`

---

### 问题 2：模型下载或预热太慢

**现象**：服务已启动，但 `/health/ready` 长时间仍为 `warming_up`。

**排查方式**：

- 首次启动时等待模型下载与预热完成
- 检查网络连接是否影响 HuggingFace 模型下载
- 必要时配置 `HF_ENDPOINT=https://hf-mirror.com`

---

### 问题 3：LLM 调用失败

**现象**：请求时出现 API Key 错误或上游模型调用失败。

**排查方式**：

检查 `.env` 中以下字段：

- `HIVEMEMORY__LLM__WORKER__MODEL`
- `HIVEMEMORY__LLM__WORKER__API_KEY`
- `HIVEMEMORY__LLM__WORKER__API_BASE`
- `HIVEMEMORY__LLM__LIBRARIAN__MODEL`
- `HIVEMEMORY__LLM__LIBRARIAN__API_KEY`
- `HIVEMEMORY__LLM__LIBRARIAN__API_BASE`

若使用 DeepSeek，模型名通常应类似：

```env
HIVEMEMORY__LLM__WORKER__MODEL=deepseek/deepseek-chat
HIVEMEMORY__LLM__LIBRARIAN__MODEL=deepseek/deepseek-chat
```

---

### 问题 4：前端无法连接后端

**现象**：前端页面已打开，但 API 请求失败。

**排查方式**：

确认以下内容：

- 后端已运行在 `http://localhost:8769`
- 前端 dev server 运行在 `http://127.0.0.1:5173`
- `frontend/vite.config.ts` 中 `/api` 代理目标未被改动

---

## 11. 项目结构参考

```text
HiveMemory/
├── configs/                 # 环境变量模板与主配置
├── docker/                  # Docker 一键部署（后端应用 + Qdrant）
├── docs/                    # 项目文档
├── frontend/                # React + Vite 前端开发界面
├── scripts/                 # 调试与辅助脚本
├── src/hivememory/
│   ├── core/                # 核心数据模型
│   ├── engines/             # Gateway / Retrieval / Perception / Generation / Lifecycle
│   ├── infrastructure/      # Storage / LLM / SystemBus / WebSocket
│   ├── patchouli/           # Patchouli 体系、Kernel、MTP、WorkerAgent
│   └── server/              # FastAPI 服务入口与路由
└── tests/                   # 单元、集成、端到端测试
```

---

## 12. 验收检查清单

完成以下项目后，可认为本地环境已基本可用：

- [ ] Docker 服务正常运行（HiveMemory + Qdrant）
- [ ] Python 虚拟环境已激活
- [ ] 后端依赖安装成功（`pip install -e .`）
- [ ] `.env` 中已配置 LLM API Key
- [ ] `hivememory-server` 可正常启动
- [ ] `GET /health` 返回成功
- [ ] `GET /health/ready` 最终返回 ready
- [ ] 前端 `npm run dev` 可正常运行（可选）
- [ ] `pytest` 可正常执行默认测试集（可选）

---

## 13. 下一步

环境搭建完成后，你可以继续：

1. 阅读 [README.md](../README.md) 了解项目整体能力
2. 阅读 [PROJECT.md](PROJECT.md) 了解 Patchouli 架构与设计背景
3. 通过 `/api/v1/chat` 或 `/api/v1/ingest` 接入你的 Agent 工作流
4. 修改 `configs/config.yaml` 调整检索、感知和生命周期参数

---

## 14. 获取帮助

如果遇到问题，建议按以下顺序排查：

1. 查看后端日志输出
2. 检查容器状态与 Docker 日志
3. 检查 `.env` 与 `configs/config.yaml` 是否一致
4. 在 GitHub 提交 Issue：
   - [HiveMemory Issues](https://github.com/artemis03976/HiveMemory/issues)
