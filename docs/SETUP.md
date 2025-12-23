# HiveMemory 环境搭建指南

本文档指导您从零开始搭建 HiveMemory 开发/运行环境。

---

## 📋 前置要求

### 必需工具
- **Python 3.12+** ([下载地址](https://www.python.org/downloads/))
- **Docker Desktop** ([下载地址](https://www.docker.com/products/docker-desktop))
- **Git** (用于克隆项目)

### 硬件要求
- **内存**: 至少 8GB RAM (运行 BGE-M3 Embedding 模型)
- **磁盘**: 5GB 可用空间 (模型文件 + 数据库)
- **GPU** (可选): 如需加速 Embedding, 推荐 NVIDIA GPU + CUDA

---

## 🚀 快速开始 (5分钟)

### Step 1: 克隆项目

```bash
git clone <your-repo-url>
cd HiveMemory
```

### Step 2: 启动 Docker 服务

```bash
# 启动 Qdrant 和 Redis
cd docker
docker-compose up -d

# 验证服务状态
docker ps
```

**预期输出:**
```
CONTAINER ID   IMAGE                    STATUS
xxx            qdrant/qdrant:latest     Up 10 seconds
yyy            redis:7-alpine           Up 10 seconds
```

**访问 Qdrant Dashboard**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

### Step 3: 创建 Python 虚拟环境

```bash
# 返回项目根目录
cd ..

# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 4: 安装依赖

```bash
pip install -r requirements.txt
```

**首次安装说明**:
- 安装时间约 5-10 分钟 (取决于网速)
- `sentence-transformers` 会自动下载 BGE-M3 模型 (~2GB)
- 如果下载慢, 可配置 HuggingFace 镜像:

```bash
# Windows (PowerShell)
$env:HF_ENDPOINT = "https://hf-mirror.com"

# Linux/macOS
export HF_ENDPOINT=https://hf-mirror.com
```

### Step 5: 配置环境变量

```bash
# 复制环境变量模板
cp configs/.env.example .env

# 编辑 .env 文件
# Windows: notepad .env
# Linux/macOS: nano .env 或 vim .env
```

**必须配置的字段** (阶段1测试):

```env
# Librarian Agent 使用的 LLM (帕秋莉)
LIBRARIAN_LLM_MODEL=deepseek/deepseek-chat
LIBRARIAN_LLM_API_KEY=sk-xxxxx  # 替换为您的 DeepSeek API Key
LIBRARIAN_LLM_API_BASE=https://api.deepseek.com

# Embedding 模型 (本地运行, 无需 API Key)
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cpu  # 如有 GPU, 改为 cuda
```

**可选配置** (Worker Agent, 阶段2使用):

```env
WORKER_LLM_MODEL=gpt-4o
WORKER_LLM_API_KEY=sk-xxxxx
```

### Step 6: 运行连接性测试

```bash
python tests/test_connections.py
```

**预期输出**:
```
✓ Qdrant 连接成功!
✓ Redis 连接成功!
✓ Embedding 模型加载成功!
✓ 数据模型验证成功!
⊘ LiteLLM 测试 (跳过或成功)

🎉 所有核心组件测试通过! 系统已就绪。
```

---

## 🧪 运行阶段1测试

### 端到端记忆入库测试

```bash
python scripts/test_ingestion.py
```

**测试内容**:
1. 模拟 3 个对话场景 (代码片段/用户偏好/闲聊)
2. Patchouli 提取结构化记忆
3. 存储到 Qdrant 向量数据库
4. 验证语义检索功能

**预期输出**:
```
📝 场景: 代码片段提取
✓ 记忆原子 xxx-xxx-xxx-xxx
  标题: Python ISO8601 日期解析函数
  类型: CODE_SNIPPET
  标签: #python #datetime #iso8601

测试结果汇总:
  ✓ 通过  代码片段提取
  ✓ 通过  用户偏好设置
  ○ 跳过  闲聊过滤测试

🎉 测试完全成功! Patchouli 工作正常。
```

---

## 🔧 常见问题排查

### 问题1: Qdrant 连接失败

**症状**:
```
✗ Qdrant 连接失败: Connection refused
```

**解决方案**:
```bash
# 检查 Docker 容器是否运行
docker ps

# 如果没有运行, 启动服务
cd docker
docker-compose up -d

# 查看日志
docker logs hivememory_qdrant
```

### 问题2: Embedding 模型下载失败

**症状**:
```
✗ Embedding 模型加载失败: Connection timeout
```

**解决方案**:
```bash
# 方案1: 使用 HuggingFace 镜像 (中国大陆)
export HF_ENDPOINT=https://hf-mirror.com
pip install -r requirements.txt

# 方案2: 手动下载模型 (如果已有模型文件)
mkdir -p ~/.cache/huggingface/hub
# 将模型文件放入上述目录
```

### 问题3: LiteLLM 调用失败

**症状**:
```
✗ LiteLLM 调用失败: Invalid API Key
```

**解决方案**:
```bash
# 检查 .env 文件中的 API Key 是否正确
cat .env | grep LIBRARIAN_LLM_API_KEY

# 如果使用 DeepSeek, 确保格式正确:
# LIBRARIAN_LLM_MODEL=deepseek/deepseek-chat  (注意前缀 deepseek/)
# LIBRARIAN_LLM_API_KEY=sk-xxxxx
```

### 问题4: 记忆提取为空

**症状**:
```
⚠️  所有场景都未提取到记忆
```

**可能原因**:
1. **API Key 未配置**: 检查 `.env` 文件
2. **模型判断无价值**: 查看日志中 `has_value=false`
3. **JSON 解析失败**: 查看日志中的错误信息

**调试方法**:
```bash
# 启用详细日志
# 在脚本开头添加:
logging.basicConfig(level=logging.DEBUG)

# 重新运行测试
python scripts/test_ingestion.py
```

---

## 📁 项目结构说明

```
HiveMemory/
├── src/hivememory/          # 核心代码
│   ├── core/
│   │   ├── models.py        # ✅ 数据模型 (MemoryAtom)
│   │   └── config.py        # ✅ 配置管理
│   ├── agents/
│   │   └── patchouli.py     # ✅ Librarian Agent
│   ├── memory/
│   │   └── storage.py       # ✅ Qdrant 存储层
│   └── utils/
│       └── buffer.py        # ✅ 对话缓冲器
│
├── tests/
│   └── test_connections.py # ✅ 连接性测试
│
├── scripts/
│   └── test_ingestion.py   # ✅ 端到端测试
│
├── docker/
│   └── docker-compose.yml  # ✅ Docker 配置
│
└── configs/
    ├── config.yaml         # ✅ 主配置
    └── .env.example        # ✅ 环境变量模板
```

**图例**:
- ✅ 阶段0/1已实现
- ⏳ 后续阶段开发
- 📝 文档

---

## 🎯 验收检查清单

完成以下检查项后，阶段0和阶段1即为成功搭建:

- [ ] Docker 服务正常运行 (`docker ps` 显示 Qdrant 和 Redis)
- [ ] Python 虚拟环境已激活
- [ ] 依赖全部安装成功 (`pip list | grep langchain`)
- [ ] BGE-M3 模型已下载 (首次运行测试时自动下载)
- [ ] `.env` 文件已配置 API Key
- [ ] `test_connections.py` 全部通过 (或仅 LiteLLM 跳过)
- [ ] `test_ingestion.py` 至少提取到 1 个记忆
- [ ] Qdrant Dashboard 中可见向量数据

---

## 🔜 下一步

完成环境搭建后, 您可以:

1. **阶段 II**: 实现记忆检索与 Context 注入 (详见 [ROADMAP.md](ROADMAP.md))
2. **自定义配置**: 修改 `configs/config.yaml` 调整参数
3. **集成到项目**: 参考 `scripts/test_ingestion.py` 集成到您的应用

---

## 📞 获取帮助

如遇到问题:

1. 查看详细日志: `python xxx.py` 会输出调试信息
2. 检查 Docker 日志: `docker logs hivememory_qdrant`
3. 提交 Issue: [GitHub Issues](https://github.com/yourusername/HiveMemory/issues)

---

**祝您使用愉快！ 🐝**
