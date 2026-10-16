---
title: 测试套件冗余收敛后续
status: archived
archived_at: 2026-10-16
implemented_by: docs/test-design-economy（本归档提交）
superseded_by: docs/governance/testing/test-design-standards.md
owner: project
scope: test-suite-ownership-and-redundancy-follow-ups
code_paths:
  - tests/
related_docs:
  - docs/governance/testing/test-design-standards.md
  - docs/governance/baselines/test-suite-redundancy-2026-10-16.md
last_reviewed: 2026-10-16
---

# 测试套件冗余收敛后续

本 Todo 已于 2026-10-16 完成并归档。现行规则见[测试设计规范](../../governance/testing/test-design-standards.md)第 4 节，各轮结果记录在该规范第 11 节第 6 项。

## 问题与证据

[测试设计规范](../../governance/testing/test-design-standards.md)第 4 节要求：

- 每条不变量只在归属层拥有权威测试；
- 参数化维度对应不同分支；
- 不写类型层测试与墓碑测试；
- 测试文件不以计划命名。

2026-10-16 按该节完成了第一轮收敛。快速集由 3035 个用例降至 2863 个，收集总数由 3085 个降至 2911 个，参数化展开由 624 个降至 486 个，行与分支覆盖没有新增缺失。第一轮处理了以下几类：

- 展开最多的访问矩阵；
- 身份入口守卫文件；
- 旧 Profile 权限键的下游测试；
- 必填参数测试；
- 以计划或阶段命名的文件；
- 感知与轨迹摘要的平行重复。

同日对遗留事项做了后续追查，证据冻结在[调查基线](../../governance/baselines/test-suite-redundancy-2026-10-16.md)中，快照为 `8f97f130`。随后按基线逐项核对并执行修正。

## 第二轮执行结果（2026-10-16）

整体数量变化：

| 指标 | 执行前 | 执行后 |
|:---|---:|---:|
| 完整收集 | 2909 | 2792 |
| 快速集 | 2863 | 2748（2746 passed、2 skipped） |
| 参数化展开 | 486 | 382 |
| 展开不少于 3 个的函数 | 83 个、368 例 | 49 个、234 例 |

与执行前的全量快速集相比，覆盖率没有新增缺失的行或分支。

原登记的 5 项均已处理：

1. **感知测试主类型**：unit 与 integration 两份感知测试合并为 [感知集成测试](../../../tests/integration/patchouli/test_perception.py)，共用同一工厂。summary-only 列表例下移到 ShortTermMemoryStore 的 owner；感知关闭例补充了“不依赖 apply journal 的重放”断言。
2. **同类文件对**：按基线完成了以下迁移：
   - 交互提交的两个真实链路用例迁入[重试集成测试](../../../tests/integration/patchouli/test_workspace_interaction_retry.py)；
   - CALL coordinator 的 3 个错误用例并回 [unit 文件](../../../tests/unit/alice/orchestration/sub_agent/test_call_coordinator.py)，并写明 Coordinator 与 RunSession 属于同一内聚单元；
   - 生命周期归档组的完整性与存储位置断言并入[生命周期集成测试](../../../tests/integration/test_lifecycle.py)，由测试 helper 自身实现“幂等”的用例删除；
   - QueueAgentRunOutput 的 backpressure 例移到它的 [owner](../../../tests/unit/alice/runtime/test_streaming.py)。
3. **`_e2e` 命名的集成文件**：改名为 `tests/integration/mtp/test_write_chain.py` 与 `test_update_chain.py`，类名去掉 E2E。同时删除未注入任何被测对象的 librarian 替身及其布置。
4. **目录包结构**：9 个目录补充 `__init__.py`。补包前后完整收集的 nodeid 完全一致。
5. **参数化矩阵**：38 个候选全部按基线的承接关系处理。

此外：

- 第 3.1 节中 MTP 包下的 18 个生成引擎用例迁入[生成引擎 owner](../../../tests/unit/engines/generation/test_engine.py)；
- 第 3.2 节中的 5 个缺参测试与 2 个签名快照已退役。

### 与基线不同的处理

基线已冻结，以下差异只在此记录：

- **`test_process_terminal_status_preserves_dispatched_intents`**：保留的是 `new_topic=True` 一侧，而基线对照保留的是 `False`。只有 Gateway 目标与 prepare 返回的话题不同，才能区分意图话题是取自 prepare 结果，还是误用了 Gateway 目标。
- **`test_unusable_custom_profile_returns_preparation_error_without_fallback`**：保留的是 private，而基线建议保留 invalid。理由有两点：
  - missing 与 private 在 Patchouli 的同一 `atom is None` 分支汇合，invalid 已由 Patchouli owner 测试覆盖；
  - private 另外经过真实 adapter 的别名冷读可见性校验。
- **Mode C 仅有 instruction 的 fallback**：合并结果与原内容相同时，`_apply_update` 会降级为 TOUCH 并丢弃 changelog，经 `process()` 无法观察。因此 owner 测试改为断言降级为 TOUCH，没有迁入 changelog 断言。
- **基线的 3 个待核定项**已作出结论：
  - **热态操作配对（4 → 2）**：READ 与 RUN 共用读取缓存，任一入口若在授权前查缓存，只有热态才能暴露，所以两个方向都保留。WRITE 与 UPDATE 在授权前不读取任何共享状态，由冷态逐请求矩阵覆盖。
  - **不可读 redirect（3 → 1）**：先在 AliasResolver 单测补充“backing 返回受限原子时，交付前仍按 policy 复查”的权威用例。变异检查确认，去掉该复查后只有新用例失败。集成层随后保留 policy 拒绝作为代表。
  - **附件 hash 字段矩阵（5 → 0）**：该矩阵只比较改动后字典的 canonical 字节，没有经过生产的 hash 计算。把生产改为只对正文求 hash 后，矩阵 5 例全部通过，冻结样本测试失败。删除该矩阵。
- **长期存储 adapter 读取压缩归档的分支**：原本只由被删除的归档组覆盖。现在扩展 adapter owner 的压缩用例，改为写入后再读回。

## 生命周期评分组与强化组（2026-10-16）

第二轮之后剩余的事项是：生命周期 component 测试中的评分组（3 例）与强化组（5 例），与 owner 测试逐条对照。对照后两组全部属于平行维护，删除后该文件不再含测试，因此连同只被它使用的 `tests/fixtures/lifecycle_test_data.py` 一起删除，并去掉 `pyproject.toml` 中为它设置的 E402 例外。

评分组：

- “代码类记忆衰减慢于进行中工作”、“极高访问数截断到 100”分别与 [`test_vitality.py`](../../../tests/unit/engines/lifecycle/test_vitality.py) 的类型差异用例、截断用例重复；
- 唯一的独有断言是 FACT 的精确衰减值，它固定了 λ_eff = λ·(2 - I) 的调制。该断言并入 owner 的时间衰减用例，并改用固定时钟。原 owner 用例的“100 天后低于一半”无法区分有无类型调制。

强化组：

- HIT、CITATION、正负反馈的计数、加成、衰减基准与置信度断言均已由 [`test_reinforcement.py`](../../../tests/unit/engines/lifecycle/test_reinforcement.py) 覆盖；
- 独有的 CITATION、负反馈、正反馈三种 patch 精确字段集并入 owner 对应用例；
- owner 中完全被负反馈惩罚用例包含的置信度用例删除；
- 正反馈用例原本断言的是 mock 计算器的返回值，改为断言加成累加进 B 项与提交字段。

对照时还发现一个缺口：强化流程应先累加本次事件加成与访问计数，再交给计算器重算，但没有测试守护这个顺序。component 组虽然接了真实计算器，结果却被截断到 100，掩盖了差异。把重算挪到事件处理之前后，生命周期与 Patchouli 相关的 388 个测试全部通过。为此在强化 owner 中补充一条真实计算器用例，按 30 天衰减的 FACT 断言 HIT 后的精确分数；同样的变异下只有该用例失败。

本轮快速集由 2746 passed 降至 2738 passed（另 2 skipped），覆盖率与处理前相比没有新增缺失。

## 未纳入本 Todo 的遗留

[测试设计规范](../../governance/testing/test-design-standards.md)第 13 节登记的 Patchouli application 薄转发测试是独立遗留项，同日另行处理，结果记录在该节。
