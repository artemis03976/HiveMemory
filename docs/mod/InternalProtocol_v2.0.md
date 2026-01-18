# HiveMemory v2.0 内部协议（最小闭环草案）

**版本**：v2.0-draft  
**目标**：把 v2.0 架构里“模块之间传什么、失败如何回退、如何观测”的共识写成可落地的最小协议，避免 GlobalGateway / Perception / Retrieval / Renderer 各写一套字段与语义。

---

## 1. 术语与分层

- **Hot Path**：同步阻塞链路（用户当前轮输入 → 网关 → 可选检索 → Worker）。核心指标是延迟与稳定性。
- **Cold Path**：异步非阻塞链路（对话流入缓冲 → 话题切分 → 记忆生成/生命周期）。核心指标是准确率、一致性与幂等。
- **Observation**：网关对“用户输入 + 截断历史”的一次结构化解析结果，供 Hot/Cold 复用。

---

## 2. Observation（网关的统一输出）

### 2.1 JSON 形态（逻辑结构）

```json
{
  "schema_version": "2.0",
  "intent": "RAG|CHAT|TOOL|SYSTEM",
  "content_payload": {
    "rewritten_query": "string",
    "search_keywords": ["string"],
    "target_filters": { "type": "string" }
  },
  "memory_signal": {
    "worth_saving": true,
    "reason": "string"
  }
}
```

### 2.2 字段语义

- `schema_version`：内部协议版本号；下游必须允许未知字段，避免升级时全链路失败。
- `intent`：
  - `SYSTEM`：如 /clear /reset；Hot Path 直接透传给 Worker（或本地处理），不触发检索与冷链路写入。
  - `CHAT`：自洽闲聊，默认不检索；但仍可投喂 Cold Path（由 worth_saving 决定是否入库）。
  - `RAG`：触发检索，`rewritten_query + search_keywords + target_filters` 参与召回/过滤。
  - `TOOL`：工具类动作；是否检索由 Worker 的工具策略决定（默认不检索）。
- `content_payload.rewritten_query`：指代消解后的“可独立理解”的查询（也可被称为 standalone query）；下游不再使用 raw query 做语义锚点。
- `content_payload.search_keywords`：用于 sparse / BM25 / keyword 检索的关键词数组；允许为空数组。
- `content_payload.target_filters`：可选启发式过滤（如 memory_type/source）；允许缺省。
- `memory_signal.worth_saving`：是否值得进入昂贵的生成链路；必须允许 false。
- `memory_signal.reason`：解释性文本，仅用于调试与可观测（不要注入到用户侧回复）。

### 2.3 解析失败与回退（强制）

当网关模块离线，或输出无法解析为 JSON ，或缺关键字段时，下游必须执行保守回退：

- `intent = "CHAT"`
- `rewritten_query = 原 query`
- `search_keywords = []`
- `worth_saving = false`

并记录一次 `gateway_parse_failed=true` 指标/日志字段（见第 6 章）。

---

## 3. LogicalBlock（语义流最小字段）

### 3.1 必选字段（现状保持）

- `user_block.content`：原始用户输入
- `response_block.content`：最终回复（如果是 Agent 执行链，可能延后闭合）

### 3.2 建议新增字段（v2.0 目标）

- `rewritten_query`：来自 GatewayResult.content_payload.rewritten_query
- `anchor_text`：默认取 `rewritten_query`；若缺省则回退到 `user_block.content`
- `turn_id`：可复用会话层 turn 序号，用于去重与幂等（见第 5 章）

---

## 4. Retrieval 协议（最小交互）

### 4.1 输入（RetrievalRequest）

- `query`：优先使用 `rewritten_query`
- `keywords`：来自 `search_keywords`（可为空）
- `filters`：来自 `target_filters`（可缺省）
- `top_k`：可配置

### 4.2 输出（RetrievalResult）

- `items[]`：候选记忆原子（至少包含 id、title、type、tags、updated_at、confidence）
- `rendered_context`：给 Worker 注入的字符串（XML/Markdown），由 Renderer 负责生成

---

## 5. 幂等与一致性（Cold Path 最小约束）

- **投喂语义**：Eye → Core 的投喂建议采用 at-least-once（至少一次）队列语义。
- **去重键**：`conversation_id + turn_id`（或 `block_id`）作为写入幂等键；同一键重复到达时必须不产生重复记忆。
- **Flush 原子性**：一次 flush 产生的“摘要/记忆原子/生命周期更新”要么全部成功，要么可重试恢复，避免半写入。

---

## 6. 可观测性（统一日志/指标字段）

建议在 Hot/Cold 路径统一打点以下字段（日志字段或 metrics labels）：

- `conversation_id`, `turn_id`, `block_id`
- `intent`, `gateway_model`, `schema_version`
- `gateway_parse_failed`（bool）
- `retrieval_triggered`（bool）, `retrieval_empty`（bool）, `retrieval_latency_ms`
- `perception_split`（bool）, `arbiter_triggered`（bool）
- `worth_saving`（bool）, `generation_triggered`（bool）
