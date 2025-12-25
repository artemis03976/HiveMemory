# ChatBot Worker Agent 使用指南

## 📖 简介

ChatBot Worker Agent 是 HiveMemory 的一个简单对话机器人，主要用于测试帕秋莉（Patchouli）的记忆生成与写入功能。

### 核心功能

1. **对话交互** - 使用可配置的 LLM（GPT-4o/DeepSeek 等）与用户对话
2. **会话管理** - 基于 Redis 的会话持久化，支持多会话切换
3. **记忆生成** - 自动将对话推送给帕秋莉，触发记忆提取和存储
4. **Web UI** - 基于 Streamlit 的用户友好界面

---

## 🚀 快速开始

### 1. 环境准备

#### 启动必需服务（Qdrant + Redis）

```bash
# 启动 Qdrant 和 Redis（使用 Docker Compose）
cd docker
docker-compose up -d

# 验证服务状态
docker-compose ps
```

#### 配置环境变量

```bash
# 复制配置示例
cp configs/.env.example .env

# 编辑 .env 文件，填入你的 API Key
nano .env
```

**关键配置项**:

```bash
# Worker LLM（ChatBot 使用）
WORKER_LLM_MODEL=gpt-4o  # 或 deepseek/deepseek-chat
WORKER_LLM_API_KEY=your_api_key_here
WORKER_LLM_API_BASE=https://api.openai.com/v1

# Librarian LLM（帕秋莉使用）
LIBRARIAN_LLM_MODEL=deepseek/deepseek-chat
LIBRARIAN_LLM_API_KEY=your_deepseek_key_here
LIBRARIAN_LLM_API_BASE=https://api.deepseek.com

# Embedding 模型（本地运行）
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cpu  # 或 cuda/mps

# 数据库
QDRANT_HOST=localhost
REDIS_HOST=localhost
REDIS_PASSWORD=hivememory_redis_pass
```

---

### 2. 安装依赖

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 或使用 Poetry
poetry install
```

---

### 3. 运行测试脚本

```bash
# 运行 ChatBot 测试（验证功能）
python tests/test_chatbot.py
```

**预期输出**:

```
╭────────────────────────────────────────────────╮
│ HiveMemory ChatBot 端到端测试                  │
│ 测试 ChatBot 对话、会话管理与记忆生成         │
╰────────────────────────────────────────────────╯

📦 初始化系统组件...
  ✓ 配置加载成功
  ✓ Redis 连接成功 (localhost:6379)
  ✓ Qdrant 连接成功 (localhost:6333)
  ✓ PatchouliAgent 初始化成功
  ✓ SessionManager 初始化成功

🤖 创建 ChatBot Agent...
  ✓ ChatBot 创建成功
  模型: gpt-4o
  温度: 0.7
  最大 Tokens: 2048

💬 测试对话流程
Session ID: test_session_001

轮次 1/5
👤 User: 你好！
🤖 Bot: 你好！有什么可以帮你的吗？

轮次 2/5
👤 User: 我叫张三，是一名软件工程师
🤖 Bot: 很高兴认识你，张三！作为软件工程师...

...

✓ 对话测试完成！成功 5/5 轮

📊 验证会话存储...
  Session ID     test_session_001
  消息数量       10
  存在状态       ✓ 存在

✓ 会话存储验证通过

📚 验证记忆生成...
  找到 3 条相关记忆

╭─ 记忆 1 ──────────────────────────────╮
│ 标题: 张三的职业和编程语言偏好          │
│ 类型: USER_PROFILE                    │
│ 置信度: 0.89                          │
│ 摘要: 张三是一名软件工程师，在北京...   │
╰────────────────────────────────────────╯

✓ 记忆生成验证通过！帕秋莉已成功提取并存储记忆

============================================================

📋 测试结果汇总

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ 测试项               ┃   状态   ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ 对话测试             │ ✓ 通过   │
│ 会话存储             │ ✓ 通过   │
│ 记忆生成             │ ✓ 通过   │
└────────────────────┴──────────┘

╭──────────────────────────────────╮
│ ✅ 所有测试通过！(3/3)            │
╰──────────────────────────────────╯

📝 下一步:
  1. 启动 Streamlit UI: streamlit run examples/chatbot_ui.py
  2. 在 UI 中进行更多对话测试
  3. 查看 Qdrant Dashboard: http://localhost:6333/dashboard
```

---

### 4. 启动 Streamlit Web UI

```bash
# 启动 Web 界面
streamlit run examples/chatbot_ui.py
```

**访问**: 浏览器打开 [http://localhost:8501](http://localhost:8501)

---

## 🎨 Streamlit UI 使用

### 界面布局

```
+----------------------------------------+
| 🤖 HiveMemory ChatBot                  |
+----------------------------------------+
| [侧边栏]                 [聊天区域]     |
| ⚙️ 系统设置              👤 User: 你好  |
|                         🤖 Bot: 你好！  |
| 📋 当前会话                            |
| Session ID: abc-123     [输入框 💬]     |
| User ID: demo_user                    |
| 消息数量: 6                            |
|                                       |
| [🗑️ 清空会话]                          |
| [➕ 新建会话]                          |
|                                       |
| 🤖 LLM 配置                            |
| 模型: gpt-4o                          |
| 温度: 0.7                             |
|                                       |
| 📚 帕秋莉配置                          |
| 触发阈值: 5 条消息                     |
| 空闲触发: 15 分钟                      |
+----------------------------------------+
```

### 核心功能

1. **发送消息** - 在底部输入框输入消息，按 Enter 发送
2. **查看历史** - 自动显示当前会话的所有历史消息
3. **新建会话** - 点击"➕ 新建会话"开始新对话
4. **清空会话** - 点击"🗑️ 清空会话"删除当前会话历史
5. **会话持久化** - 刷新页面后历史消息仍然保留

### 帕秋莉自动触发

在聊天过程中，以下情况会自动触发帕秋莉的记忆生成：

- **消息数量触发**: 每发送 5 条消息（user + assistant）
- **空闲超时触发**: 15 分钟无新消息

你会看到帕秋莉在后台自动：

1. 评估对话价值（Gating）
2. 提取结构化记忆（Extraction）
3. 去重和知识演化（Deduplication）
4. 生成向量并存储到 Qdrant（Persist）

---

## 🔧 技术架构

### 系统流程图

```
用户输入
   ↓
Streamlit UI
   ↓
ChatBotAgent
   ↓
┌─────────────────┬──────────────────┐
│                 │                  │
│  Worker LLM     │  SessionManager  │
│  (生成回复)      │  (Redis 存储)     │
│                 │                  │
└─────────────────┴──────────────────┘
         ↓
  ConversationBuffer
         ↓
  (自动触发: 5条/15分钟)
         ↓
  MemoryOrchestrator (帕秋莉)
         ↓
  [Gating → Extraction → Dedup → Persist]
         ↓
  Qdrant VectorDB ✅
```

### 核心组件

| 组件                    | 职责                                  | 文件路径                                            |
| ----------------------- | ------------------------------------- | --------------------------------------------------- |
| **ChatBotAgent**        | 对话管理、LLM 调用、Buffer 推送       | `src/hivememory/agents/chatbot.py`                  |
| **SessionManager**      | 会话持久化（Redis）                   | `src/hivememory/agents/session_manager.py`          |
| **ConversationBuffer**  | 对话缓冲、自动触发                    | `src/hivememory/generation/buffer.py`               |
| **MemoryOrchestrator**  | 记忆生成流程编排（帕秋莉）            | `src/hivememory/generation/orchestrator.py`         |
| **QdrantMemoryStore**   | 向量存储和检索                        | `src/hivememory/memory/storage.py`                  |
| **Streamlit UI**        | Web 界面                              | `examples/chatbot_ui.py`                            |

---

## 📊 验证记忆生成

### 方法 1: 通过测试脚本

```bash
# test_chatbot.py 会自动验证记忆生成
python tests/test_chatbot.py
```

### 方法 2: 手动查询 Qdrant

```python
from hivememory.memory.storage import QdrantMemoryStore
from hivememory.core.config import get_config

config = get_config()
storage = QdrantMemoryStore(
    host=config.qdrant.host,
    port=config.qdrant.port,
    collection_name=config.qdrant.collection_name,
    embedding_config=config.embedding
)

# 搜索记忆
memories = storage.search_memories(
    query_text="用户的个人信息",
    user_id="test_user",
    top_k=10
)

for mem in memories:
    print(f"标题: {mem.index.title}")
    print(f"摘要: {mem.index.summary}")
    print(f"置信度: {mem.meta.confidence_score}")
    print("-" * 50)
```

### 方法 3: 使用 Qdrant Dashboard

访问 [http://localhost:6333/dashboard](http://localhost:6333/dashboard) 查看集合中的向量数据。

---

## ⚙️ 配置说明

### LLM 模型切换

在 `.env` 文件中修改:

```bash
# 切换为 DeepSeek
WORKER_LLM_MODEL=deepseek/deepseek-chat
WORKER_LLM_API_KEY=sk-xxxxx
WORKER_LLM_API_BASE=https://api.deepseek.com

# 或切换为 Claude
WORKER_LLM_MODEL=anthropic/claude-3-5-sonnet-20241022
WORKER_LLM_API_KEY=sk-ant-xxxxx
WORKER_LLM_API_BASE=https://api.anthropic.com
```

### Buffer 触发条件

在 `configs/config.yaml` 中修改:

```yaml
memory:
  buffer:
    max_messages: 5  # 改为 3 可更频繁触发
    timeout_seconds: 900  # 改为 300（5分钟）
```

### 会话 TTL（过期时间）

在 `.env` 或 `config.yaml` 中:

```bash
SESSION_TTL_DAYS=7  # 7天后自动清理
```

---

## 🧪 测试场景建议

为了更好地观察帕秋莉的记忆生成行为，建议测试以下对话场景：

### 场景 1: 个人信息录入

```
1. "你好，我叫李明"
2. "我在上海工作，职业是产品经理"
3. "我最喜欢的运动是游泳"
4. "我每周都会去健身房三次"
5. "我的生日是1990年5月15日"
```

**预期记忆**:

- `USER_PROFILE`: 李明的个人信息（姓名、职业、城市）
- `USER_PROFILE`: 李明的运动偏好

### 场景 2: 技术讨论

```
1. "我在学习 React"
2. "我发现 useEffect 的依赖数组很容易出错"
3. "今天终于理解了闭包陷阱的问题"
4. "我决定写一篇博客记录这个知识点"
5. "下次遇到类似问题就不会困惑了"
```

**预期记忆**:

- `REFLECTION`: 关于 React useEffect 依赖数组的经验教训
- `WORK_IN_PROGRESS`: 计划写博客

### 场景 3: 项目规划

```
1. "我们团队下周要开始新项目"
2. "项目名叫 SmartBot，是一个 AI 客服系统"
3. "技术栈是 Python + FastAPI + LangChain"
4. "我负责后端 API 开发"
5. "预计两个月完成 MVP"
```

**预期记忆**:

- `WORK_IN_PROGRESS`: SmartBot 项目信息
- `CODE_SNIPPET`: 技术栈选型（可能）

---

## 🐛 常见问题

### Q1: "LLM 调用失败"

**原因**: API Key 配置错误或网络问题

**解决**:

```bash
# 检查 .env 文件
cat .env | grep WORKER_LLM

# 测试 API 连接
curl -H "Authorization: Bearer $WORKER_LLM_API_KEY" \
     https://api.openai.com/v1/models
```

### Q2: "Redis 连接失败"

**原因**: Redis 服务未启动

**解决**:

```bash
# 检查 Docker 服务
docker-compose ps

# 重启 Redis
docker-compose restart redis
```

### Q3: "帕秋莉没有生成记忆"

**原因**: 对话数量不足或对话内容价值过低

**解决**:

1. 确保发送了至少 5 条消息
2. 消息内容包含有价值的信息（不是简单的"你好"）
3. 手动触发 Buffer（在代码中调用 `buffer.flush()`）

### Q4: "Streamlit 启动失败"

**原因**: 依赖未安装

**解决**:

```bash
pip install streamlit
```

---

## 📈 下一步

1. **测试更多场景** - 尝试不同类型的对话，观察帕秋莉的提取行为
2. **查看记忆数据** - 在 Qdrant Dashboard 中查看生成的记忆向量
3. **实现记忆检索** - 将记忆注入到 ChatBot 的上下文中（Stage 2）
4. **集成到应用** - 将 ChatBot 集成到你的实际应用中

---

## 📚 相关文档

- [项目架构说明](../docs/PROJECT.md)
- [开发路线图](../docs/ROADMAP.md)
- [记忆模型设计](../src/hivememory/core/models.py)
- [帕秋莉实现](../src/hivememory/agents/patchouli.py)

---

## 🙋 反馈与支持

如有问题，请查看:

- GitHub Issues: [https://github.com/yourusername/HiveMemory/issues](https://github.com/yourusername/HiveMemory/issues)
- 技术文档: `docs/PROJECT.md`

---

**Happy Testing! 🚀**
