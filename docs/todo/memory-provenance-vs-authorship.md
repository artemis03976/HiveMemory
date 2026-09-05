---
title: Memory Provenance vs Authorship
status: todo
owner: patchouli
scope: separate-multi-agent-provenance-from-single-valued-authorship
code_paths:
  - src/hivememory/core/models/memory.py
  - src/hivememory/engines/generation/engine.py
  - src/hivememory/engines/retrieval/filter_adapter.py
  - src/hivememory/engines/retrieval/models.py
  - src/hivememory/patchouli/services/topic_working_set.py
  - src/hivememory/core/models/topic.py
related_docs:
  - docs/archive/plans/perception-topic-buffer-boundary-refactor.md
  - docs/patchouli/perception.md
  - docs/architecture/workspace.md
  - docs/architecture/decisions/0002-unique-identities-and-minimal-concurrency.md
last_reviewed: 2026-09-04
---

# 记忆溯源与责任主体的分离

## 事项定位

`MetaData.source_agent_id` 是单值字段，但它同时被当作三种语义使用：**溯源**（哪些 Agent 的
工作产出了这条内容）、**责任主体**（审计链上的单一 owner）、以及在 v1 兼容层里的**授权**。
Workspace 概念落地后，"topic 属于 Workspace，Agent 只是进入 Workspace 的工作者"，用户可以
在一个话题内随时切换 Agent，于是**一个话题天然包含多个 Agent 的贡献**，单值字段无法表达。

历史维护路径（IDLE_TIMEOUT / SHUTDOWN 扫描）曾通过 `build_maintenance_scope`
（已随 `TopicBufferService` 删除）从"最后一个 block 的 identity"猜出一个 Agent 作为归属，
无 block 时回落到 Pydantic 默认值 `omni_doll`。这既表达不了多 Agent 事实，也会把"未知"
写成一个真实存在的 Agent 名字。边界重构后维护路径直接复用 `TopicWorkingSet`
（[topic_working_set.py](../../src/hivememory/patchouli/services/topic_working_set.py)）
在 touch 时冻结的执行作用域，但"多 Agent 溯源"的语义缺口仍在，本 todo 继续有效。

本事项**不是安全缺陷**（见下"授权已经分离"），而是溯源建模不准 + 一处默认值伪造。

## 授权已经分离，这决定了本事项的性质

v2 记忆的读取授权只看 `access_policy`，不看 `source_agent_id`：

- [`memory_visible_to_actor`](../../src/hivememory/engines/retrieval/policy.py#L21-L33)
  只读 `memory.meta.access_policy`；
- [`_read_policy_filter`](../../src/hivememory/engines/retrieval/filter_adapter.py#L151-L195)
  的 v2 分支匹配 `meta.access_policy.*`；`meta.source_agent_id` 只出现在
  [legacy 分支](../../src/hivememory/engines/retrieval/filter_adapter.py#L209-L224)
  （`schema_version` is_empty）；
- 只有 v1 记忆通过
  [`_adapt_v1_policy`](../../src/hivememory/engines/retrieval/memory_codec.py#L134-L155)
  把 `source_agent_id` 反推成 `target_agent_id`——这个历史耦合**不得重新引入**；
- 结算记忆的策略在
  [engine.py:474](../../src/hivememory/engines/generation/engine.py#L469-L477)
  硬编码为 `MemoryAccessPolicy.public()`。

因此把溯源改成多值**不可能开出权限口子**：权限住在另一个字段里。这也是本事项可以安全推进、
且不需要 `schema_version` 升级的前提。

## 单值字段里挤了两个仍需保留的语义

| 语义 | 需要的基数 | 现状 | 结论 |
|:---|:---|:---|:---|
| 溯源：哪些 Agent 的工作产出了内容 | **多值** | 挤在 `source_agent_id`，只能取一个 | 需要新字段 |
| 责任主体：审计链单一 owner | **必须单值** | 同一个 `source_agent_id` | 保留，但需要正确取值 |
| 授权：谁能读 | 单值策略 | 已由 `access_policy` 独立承载 | 不动 |

责任主体不可取消：
[`MemoryVersionArtifact.owner_agent_id`](../../src/hivememory/engines/artifacts/memory.py#L54-L65)
与 [`MemoryCreationArtifact.owner_agent_id`](../../src/hivememory/engines/artifacts/memory.py#L69-L79)
都是单值，且构成审计链本身。"全部记下来"只能补充它，不能替代它。

## 父子 Agent 模式不在本事项范围

父子（CALL / 子 Agent）模式下"主体工作者仍是父 Agent"**已经是结构事实，无需实现**：

- 一次顶层 run 只产出一个 `InteractionPayload`（`finalize_agent_run`）；
- [`_build_block`](../../src/hivememory/engines/perception/semantic_flow_perception_layer.py#L305-L337)
  使用提交时的 `identity_scope.actor_identity`；
- 子 Agent 在父 frame 内执行、以 IPC return 并回父的 working history，不单独走摄入。

所以子 Agent 身份到不了 `LogicalBlock`。**唯一产生多身份 block 的路径是用户在话题内主动
切换 Agent**（Workspace 访客模型）。本事项的实际范围仅此一种。

## 裁定：被动结算的记忆没有 Agent 作者

顺着"Agent 只是 Workspace 访客"往下推，不应在多个 Agent 中挑一个，而应承认**被动结算不是
任何 Agent 的行为**：

- Mode B/C（WRITE/UPDATE）是 Agent 显式署名的创作，`source_agent_id` 就是它，天然正确；
- Mode A 结算是**系统**决定把话题蒸馏成记忆，没有任何 Agent 提出请求。

系统内部已经这样分类了——
[`MemoryGenerationSource.SETTLE.creation_artifact_intent`](../../src/hivememory/patchouli/control/memory_generation/models.py#L43-L53)
返回 `"SYSTEM"`。也就是说 **artifact 层已经把结算产物标记为系统来源，而
`source_agent_id` 却同时声称它出自某个具体 Agent**，两者当前互相矛盾。本裁定只是让后者与前者
一致。

据此：

1. settle 路径的 `source_agent_id` 使用一个**保留系统 id**，不是首个 / 末个 / 贡献最多的
   Agent。`MetaData.source_agent_id` 是 `min_length=1` 必填
   （[memory.py:122](../../src/hivememory/core/models/memory.py#L122)），所以必须是显式保留
   值，不能留空；
2. 多 Agent 事实由新增多值字段承载；
3. 不引入"贡献度最大"之类的 tie-break 启发式——它既不稳定（换人收尾即翻转），也无法从内容
   自证。

这样多 Agent 话题从"难题"变成"非问题"。

## 建议的多值溯源字段

```python
# MetaData 新增
contributing_agent_ids: tuple[str, ...] = ()   # 由 block identity 去重得到，保持出现顺序
```

**有现成消费者，不是预留设计。** `QueryFilters.source_agent_id`
（[retrieval/models.py:21](../../src/hivememory/engines/retrieval/models.py#L21)，
由 MTP `agent:` 过滤 token 填充，见
[parser.py:251-252](../../src/hivememory/core/mtp/parser.py#L249-L252)）在
[filter_adapter.py:93-99](../../src/hivememory/engines/retrieval/filter_adapter.py#L93-L99)
匹配 `meta.source_agent_id`。当前查"来自 coder_doll 的记忆"会漏掉 coder_doll 做了绝大部分
工作、但由别人收尾的话题。改为匹配溯源集合才符合用户预期。

**成本比预期低。** Qdrant 的 `MatchValue` 对数组类型 payload 字段按"任一元素匹配"语义工作，
因此 `FieldCondition` 的写法不变；且
[`create_collection`](../../src/hivememory/infrastructure/storage/vector_store.py#L107-L140)
目前**没有创建任何 payload index**，所以本改动不需要新增索引工作，也不改变现有索引策略。

**无需 schema 升级。** 该字段是纯增量的，默认空 tuple 即可解码历史记录，`schema_version`
保持 2，不需要数据迁移。单 Agent 话题下它等于 `(source_agent_id,)`，无信息损失。

## 不是缺陷的部分

`user_id` 维度安全，不需要处理：

- [`IdentityScope._require_same_owner`](../../src/hivememory/core/models/identity.py#L112-L121)
  强制 `actor.user_id == workspace.owner_user_id`；
- 所有 block 都经由通过该校验的 scope 写入；
- adapter 拒绝同一 `topic_id` 跨 Workspace 写入
  （[short_term.py:83-90](../../src/hivememory/patchouli/memory_library/adapters/short_term.py#L83-L90)）。

所以 fallback 里的 `user_id=owner_user_id` 恒等于真值，被伪造的**只有 `agent_id` 与
`team_id`**，且 `IdentityScope` 构造不会在此处抛 `OwnerMismatchError`。

## 需要处理的具体缺陷

### 1. 维护路径用默认值伪造 `agent_id` / `team_id`

`ActorIdentity(user_id=...)` 未传 `agent_id` / `team_id`，落到
[constants.py](../../src/hivememory/core/constants.py#L12-L16) 的
`DEFAULT_AGENT_ID = "omni_doll"` 与 `DEFAULT_TEAM_ID = None`。

该分支**不是理论路径**：`TopicData.has_content` 为
`bool(self.blocks) or bool(self.state_summary.strip())`
（[topic.py:138-146](../../src/hivememory/core/models/topic.py#L138-L146)），
即 `blocks == () and state_summary != ""` 是合法驻留态——Page Folding 折叠掉全部 block 后
正是这个形态。这类话题结算时 `agent_id` 无条件变成 `omni_doll`。

采纳本裁定后此缺陷自然消解：归属不再依赖"最后一个 block 还在不在"，`team_id` 也不再需要
重建（系统来源本就没有 team）。这与
[fail-loud 偏好](../../../.claude/projects/c--Users-29305-Projects-HiveMemory/memory/fail-loud-no-silent-model-fallback.md)
一致：缺失的身份事实不回落常量，而是显式标记为系统来源。

### 2. `current_agent_id` 是死字段，应当删除

[`TopicData.current_agent_id`](../../src/hivememory/core/models/topic.py#L110) 默认
`"default"`，已删除的 e2e flush 测试注释曾明确记录"v4 架构下话题不记录 `current_agent_id`（恒为 default）"。渲染路径用的是
`context.identity.agent_id`（[assembler.py:54](../../src/hivememory/prompts/assembler.py#L54)），
不读该字段。

采纳"结算无作者"后它没有承载对象，应**删除**而非复活。

## 影响范围

- `core/models/memory.py`：`MetaData` 新增 `contributing_agent_ids`；
- `engines/generation/engine.py`：settle 路径写入系统 `source_agent_id` +
  溯源集合（需要能区分 Mode A 与 Mode B/C）；
- `engines/retrieval/filter_adapter.py`：`filters.source_agent_id` 改匹配溯源集合；
- `core/constants.py`：新增保留系统 actor id 常量；
- 维护路径的执行作用域由 `TopicWorkingSet` 在 touch 时冻结（`build_maintenance_scope`
  已随 `TopicBufferService` 删除，不照搬旧语义）；
- `core/models/topic.py`：删除 `current_agent_id`；
- 已写入的历史 `MemoryAtom`：**不做数据迁移**。已落库的 `source_agent_id` 无法可靠反推真值，
  回填只会制造第二种猜测。修复只保证此后写入正确。

## 明确非目标

- 不改变 `access_policy` 语义，不让溯源字段参与授权判断；
- 不恢复 v1 那种 `source_agent_id → target_agent_id` 的耦合；
- 不引入"贡献度"权重或任何单值 tie-break 启发式；
- 不升级 `schema_version`，不迁移或回填历史 `MemoryAtom.meta`；
- 不改变 `IdentityScope` 的字段集，也不向其加入 topic 维度；
- 不修改 Page Folding 的折叠策略来"保住" block 身份（裁定后不再需要）；
- 不阻塞[感知层边界重构](../archive/plans/perception-topic-buffer-boundary-refactor.md)（已实施完成）。

## 完成条件

- [ ] `MetaData` 具备多值溯源字段，settle 记忆携带话题内全部贡献 Agent（去重、保持顺序）；
- [ ] settle 记忆的 `source_agent_id` 为保留系统 id，与 artifact 层已有的 `SYSTEM`
      intent 一致；
- [ ] Mode B/C 记忆的 `source_agent_id` 仍为发起 WRITE/UPDATE 的真实 Agent，不受本改动影响；
- [ ] 维护路径不再通过 Pydantic 字段默认值产生 `agent_id` / `team_id`；
- [ ] `blocks == () and state_summary != ""` 的折叠态话题结算时不产生 `omni_doll` 归属；
- [ ] `filters.source_agent_id`（含 MTP `agent:` token）匹配溯源集合，能检出"参与过但未收尾"
      的话题记忆，且有覆盖该场景的测试；
- [ ] 存在测试断言"维护路径产出的 `source_agent_id` 不等于 `DEFAULT_AGENT_ID`"；
- [ ] 溯源字段不参与任何授权判断，有测试覆盖"改变溯源不改变可见性"；
- [ ] 历史 v1/v2 记录在新字段缺省下仍能正常解码，`schema_version` 未升级；
- [ ] `TopicData.current_agent_id` 已删除；
- [ ] `docs/patchouli/perception.md` 记录被动结算的系统归属语义，与实现一致。

## 相关事项

- [感知层与短期记忆边界重构](../archive/plans/perception-topic-buffer-boundary-refactor.md)（已完成）：
  `build_maintenance_scope` 已随 TopicBufferService 删除，但不修复其语义；
- [Topic shutdown 逐 Topic 失败隔离](./topic-shutdown-per-topic-failure-isolation.md)：
  同样作用于 `flush_all_for_shutdown`，可能触及相同代码；
- [ADR-0002：全局唯一身份与按需并发保护](../architecture/decisions/0002-unique-identities-and-minimal-concurrency.md)。