# HiveMemory 待办事项

## Patchouli Stage 1 集成测试相关

### 测试文件创建

- [x] 创建 `tests/test_patchouli_stage1.py` - 主测试文件
- [x] 创建 `tests/fixtures/patchouli_test_data.py` - 测试数据 fixtures

### 测试环境依赖

测试需要 Qdrant 服务器运行才能正常执行。启动 Qdrant：

```bash
# Docker 方式
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# 或本地安装方式
qdrant run
```

---

## API 限制与改进建议

### 1. 工具调用消息传递限制

**问题描述**：`PatchouliAgent.add_message()` 只接受 `role` 和 `content` 参数，无法传递 OpenAI 格式的 `tool_calls` 字段。

**当前限制**：
```python
# 现有 API
patchouli.add_message(role="assistant", content="...", user_id=..., agent_id=..., session_id=...)

# 需要（但不支持）
patchouli.add_message(
    role="assistant",
    content="...",
    tool_calls=[...],  # 不支持
    user_id=...,
    agent_id=...,
    session_id=...
)
```

**影响**：无法直接测试完整的工具调用流程（Thought → Tool Call → Tool Output）

**临时解决方案**：
- 测试中通过 `role="tool"` 传递工具输出
- 工具调用流程的完整测试需要直接使用底层感知层

**建议改进**：
```python
def add_message(
    self,
    role: str,
    content: str,
    user_id: str,
    agent_id: str,
    session_id: str,
    tool_calls: Optional[List[Dict]] = None,  # 新增
    tool_call_id: Optional[str] = None,        # 新增
) -> None:
    """添加消息到感知层（增强版）"""
    # ...
```

### 2. LogicalBlock 内部结构访问

**问题描述**：无法通过 Patchouli API 访问 LogicalBlock 的 `execution_chain`，无法验证 Triplet 是否正确记录。

**影响**：只能通过间接方式（如 token 计数）验证工具调用被处理。

**建议改进**：添加调试接口
```python
def get_current_block(
    self,
    user_id: str,
    agent_id: str,
    session_id: str,
) -> Optional[LogicalBlock]:
    """获取当前正在构建的 Block（用于调试和测试）"""
```

### 3. 语义相似度数值获取

**问题描述**：`SemanticBoundaryAdsorber` 的相似度计算是内部过程，无法获取具体数值。

**影响**：无法精确验证相似度是否在预期范围内（如 0.6-0.8）。

**建议改进**：在 FlushEvent 中添加相似度信息
```python
@dataclass
class FlushEvent:
    # ...
    similarity: Optional[float] = None  # 新增：语义相似度
```

### 4. StreamParser 不支持直接传递 StreamMessage

**问题描述**：StreamParser 只能从原始消息解析，无法直接构造 StreamMessage 并传递。

**影响**：测试时需要通过复杂的 role/content 组装来模拟工具调用。

**建议改进**：添加直接接收 StreamMessage 的接口

---

## 已知 Bug

### Qdrant Sparse Vector 存储问题

**问题**：稀疏向量检索返回 0 结果

**临时方案**：
- 测试中使用 `use_sparse=False`
- 仅使用 dense vector 进行存储和检索

**长期方案**：修复稀疏向量生成和检索逻辑

---

## 其他待办事项

- [ ] 修复 Qdrant sparse vector bug
- [ ] 扩展 PatchouliAgent.add_message() 支持工具调用参数
- [ ] 添加 get_current_block() 调试接口
- [ ] 在 FlushEvent 中添加 similarity 字段
