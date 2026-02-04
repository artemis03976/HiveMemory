# Lifecycle Module Test Design

## 1. 测试目标与范围
本测试设计旨在验证 **Lifecycle (生命周期引擎)** 对记忆价值的动态评估、强化和存储分级管理能力。
*   **测试范围**：
    *   **Vitality Scoring (生命力评分)**：验证基于置信度、固有价值、时间衰减和访问频率的评分公式计算准确性。
    *   **Reinforcement (强化机制)**：验证 HIT, CITATION, FEEDBACK 等事件对记忆生命力的正向/负向影响。
    *   **Archiving (归档机制)**：验证低分记忆的冷热分离（Vector DB -> File System）及唤醒流程。
*   **不包含**：
    *   Garbage Collector 的具体调度策略（仅验证核心 Action）。
    *   文件系统的底层 IO 性能。

## 2. 测试环境与前置条件
*   **运行环境**：
    *   Python 3.10+
    *   `pytest`
*   **外部依赖**：
    *   **Mock Storage**: 模拟 Qdrant 和本地文件系统，用于验证数据的移动。
    *   **Mock Clock**: 用于模拟时间流逝（Time Travel）。

## 3. 测试用例设计

### 3.1 评分逻辑测试 (Scoring)

| ID | 名称 | 测试步骤 | 测试数据 | 预期结果 | 优先级 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| LIF-SCR-001 | 基础分计算 | 1. 创建不同类型的记忆。<br>2. 计算初始分数。 | Type: `CODE` (High Value)<br>Type: `CHAT` (Low Value) | Score(CODE) > Score(CHAT) | P1 |
| LIF-SCR-002 | 时间衰减测试 | 1. 创建记忆 M1。<br>2. 模拟时间流逝 T 天。<br>3. 重新计算分数。 | T = 30 Days<br>Decay Factor = 0.95/day | Score(Now) < Score(Initial) * 0.95^30 | P0 |
| LIF-SCR-003 | 访问加成上限 | 1. 模拟对 M1 进行无限次访问。<br>2. 验证加成是否封顶。 | Access Count = 10000 | Boost Value <= Max Cap (e.g. 50) | P2 |

### 3.2 强化事件测试 (Reinforcement)

| ID | 名称 | 测试步骤 | 测试数据 | 预期结果 | 优先级 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| LIF-RNF-001 | 检索命中 (HIT) | 1. 触发 `HIT` 事件。<br>2. 验证 Access Count 和 Vitality。 | Event: `HIT` | Vitality += 5<br>Access Count += 1 | P0 |
| LIF-RNF-002 | 引用强化 (CITATION) | 1. 触发 `CITATION` 事件。<br>2. 验证 Updated_at 和 Decay 重置。 | Event: `CITATION` | Updated_at = Now<br>Vitality += 20 (大幅回升) | P0 |
| LIF-RNF-003 | 用户反馈 (FEEDBACK) | 1. 触发 `FEEDBACK_NEGATIVE`。<br>2. 验证分数和置信度惩罚。 | Event: `NEGATIVE` | Vitality -= 50<br>Confidence *= 0.5 | P1 |

### 3.3 归档与唤醒测试 (Archiving)

| ID | 名称 | 测试步骤 | 测试数据 | 预期结果 | 优先级 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| LIF-ARC-001 | 归档触发 | 1. 构造低分记忆 M1 (Score < Threshold)。<br>2. 调用 `archive(M1)`。<br>3. 检查存储状态。 | M1.vitality = 5.0<br>Threshold = 10.0 | Qdrant: M1 Deleted<br>FileSystem: M1.json.gz Created | P0 |
| LIF-ARC-002 | 唤醒流程 | 1. 对已归档记忆 M1 调用 `resurrect(M1)`。<br>2. 检查存储状态。 | M1 in FileSystem | Qdrant: M1 Inserted<br>FileSystem: M1 Deleted | P1 |
| LIF-ARC-003 | 归档幂等性 | 1. 对已归档记忆再次调用 `archive`。<br>2. 验证系统稳定性。 | M1 Status: Archived | No Error, No Duplicate File | P2 |

## 4. 关键验证点
1.  **分数的单调性**：在无外部事件干扰下，分数应随时间严格单调递减。
2.  **数据完整性**：归档-唤醒循环后，MemoryAtom 的所有字段（特别是 Vector 和 Payload）应保持不变，无数据丢失。
3.  **并发安全**：虽然本层测试主要关注逻辑，但需确认 `access_count` 的更新在多线程下是否原子（若适用）。

## 5. 通过/失败标准
*   **P0 用例通过率**：100%
*   **计算精度**：评分计算误差 < 0.1。

## 6. 风险与假设
*   **假设**：文件系统的读写权限在测试环境中是可用的。
*   **风险**：如果 Qdrant 的 Delete 操作是异步的（Eventual Consistency），测试中立即查询可能会失败。需在测试代码中增加短暂 Sleep 或 Retry 机制。
