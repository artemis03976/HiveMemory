---
title: 测试冗余后续追查基线（2026-10-16）
status: baseline
owner: project
scope: remaining-test-matrices-and-ownership-review
snapshot_at: 2026-10-16
source_commit: 8f97f130aa3865008d5d1d409d821df8e3f92806
code_paths:
  - tests/
  - src/hivememory/
related_docs:
  - docs/governance/testing/test-design-standards.md
  - docs/archive/todo/test-suite-redundancy-follow-ups.md
last_reviewed: 2026-10-16
---

# 测试冗余后续追查基线

本文冻结 `source_commit` 上对[冗余收敛后续 Todo](../../archive/todo/test-suite-redundancy-follow-ups.md)的调查证据。
最新行为仍以代码、测试和当前设计为准；待处理项由 Todo 跟踪。本轮只调查和整理文档，没有删除、迁移或改写测试。

## 1. 范围、数量与判断边界

完整收集为 **2909 个用例**：unit 2296、integration 567、e2e 46。沿用当前默认 marker 表达式选择 2863 个；
再显式排除 `real_infra` 后为 2861 个。Todo 中第一轮的 2911 是当时的历史结果，不覆盖为本次数字。

按参数化展开前的 nodeid 分组，展开不少于 3 个的 **83 个函数、368 个用例已逐项核对**：

- **42 个函数保留**：不同入口接线、字段绑定、状态/错误映射和不同协议输入形态有独立缺陷价值；
- **38 个函数有收敛或迁合候选**：下表的建议保留数合计比现有少 80 个；这包含补断言后才能删除的矩阵，
  不是已实施成果，也不是全套可删量承诺；
- **3 个函数待核定**：热态操作配对、redirect 冷读失败、附件 hash 字段矩阵，不计入上述 80 个。

另检查同名文件对和真实构造组件集合相近的候选，共记录 11 组边界结论。文件对筛查不代表逐条审过整套测试。

## 2. 测试边界与归属

| 测试位置 | 实际边界与证据 | 后续处置 |
|:---|:---|:---|
| `unit/patchouli/services/test_perception.py`（26 例）与 `integration/patchouli/test_perception_flush_chain.py`（13 例） | 两份工厂都使用真实 Familiar、Engine、Store、WorkingSet、Journal，仅 relay/bus 为替身 | 统一到 `integration/patchouli` 的一个感知文件；保留独立场景并共用工厂。`test_summary_only_topic_stays_in_non_empty_active_list` 实际只测 Store 筛选，可下移 Store owner。 |
| `unit/patchouli/control/test_interaction_submission.py` 与 `integration/patchouli/test_workspace_interaction_retry.py` | 前者 `test_ambiguous_failure_after_add_block_does_not_duplicate_block`、`test_retry_resumes_pending_compact_without_duplicating_block` 已组装真实 Queue、Handler、Familiar、Engine、Store、WorkingSet、Journal | 两例迁入集成文件；写入后、compact 中、调用感知前的失败时间点不同，必须保留。前者 `test_disabled_perception_does_not_require_apply_journal_entry` 直接调用 Familiar，可并入感知 disabled owner，并补一次重放断言。 |
| `unit/alice/orchestration/sub_agent/test_call_coordinator.py` 与 `integration/alice/orchestration/test_call_coordinator_error_paths.py` | 两份工厂同为真实 Coordinator、RunSession，其余协作者均替身；integration 没有增加真实边界 | 建议将后者 3 个错误用例并入 unit 文件，说明 Coordinator 与自持 RunSession 属于同一内聚单元；错误种类和 harvest 故障仍保留。 |
| `unit/engines/lifecycle/test_lifecycle_component.py::TestArchiving` 与 `integration/test_lifecycle.py` | 归档组三例使用真实 MemoryLibrary、Store、文件 adapter；integration 已有 archive/revive 链 | 把两例的内容、标签及存储位置断言并入 `test_memory_library_archive_and_revive_moves_between_stores`。`test_lif_arc_003_archive_idempotency` 的二次调用被测试自写 `_LegacyArchiverFixture.archive` 提前返回，未验证生产幂等，单独删除。评分/强化组未逐项排查。 |
| Alice `application/test_agent_run_service.py` 两个主类型 | unit 替换整个 RunExecutor；integration 使用真实 executor、session、frame factory、stream 管线 | 保留实际边界差异。 |
| Alice `orchestration/test_run_executor.py` 两个主类型 | unit 用记录型 output/null sink；integration 用真实 AgentRunStream、QueueAgentRunOutput 和消费方关闭 | 保留。单独的 QueueAgentRunOutput backpressure 例可移到它的 owner。 |
| Alice `test_system.py` 两个主类型 | 一份验证 health 投影，一份验证外部 GlobalBus 路由生命周期 | 保留不同被测行为。 |
| Patchouli `test_memory_change_events.py` 两个位置 | unit 为 Store、替身存储和记录型 publisher；integration 增加真实 adapter、publisher、local/global buses、bridge | 保留逐入口单测与异常隔离、内联等待、卸载重挂载集成场景。 |
| `unit/patchouli/services/test_memory_generation.py` 与 `integration/patchouli/test_memory_generation_pipeline.py` | unit 的 engine/library/artifact 协作者为 mock；integration 有真实 Coordinator、Controller、Bus，部分场景还有真实文件仓库 | 保留不同边界。 |
| `unit/patchouli/services/test_lifecycle.py` 与 `integration/test_lifecycle.py` | 前者为 Familiar 与 mock Engine 的结果/错误映射；后者为真实 Engine、GC、Reinforcement、Library、Store | 保留。 |
| `unit/server/routers/test_agents.py` 与 `integration/workspace/test_profile_injection_script.py` | 共享部分 root API 组件，但真实入口分别是 HTTP 和脚本认证/生命周期 | 不按共同构造类判定重复。 |

## 3. 参数矩阵以外的具体候选

### 3.1 生成引擎测试仍放在 MTP 包下

WRITE 单测（`tests/unit/agent_runtime/mtp/test_write_chain.py`）的 7 例和
UPDATE 单测（`tests/unit/agent_runtime/mtp/test_update_chain.py`）的 11 例实际都直接测试 `MemoryGenerationEngine`。
它们应由[生成引擎测试](../../../tests/unit/engines/generation/test_engine.py)承接，不能仅因文件名不同继续平行维护。

| 旧测试 | 既有 owner 测试 / 需要迁入的断言 |
|:---|:---|
| `tests/unit/agent_runtime/mtp/test_write_chain.py` / `test_mode_b_calls_extractor_with_write_metadata` | [tests/unit/engines/generation/test_engine.py](../../../tests/unit/engines/generation/test_engine.py) / `test_routes_to_mode_b`；将write_content/write_reason精确断言补入现有ModeB路由例 |
| `tests/unit/agent_runtime/mtp/test_write_chain.py` / `test_mode_a_no_write_metadata` | [tests/unit/engines/generation/test_engine.py](../../../tests/unit/engines/generation/test_engine.py) / `test_mode_a_extract_success`；将metadata没有mode断言补入现有ModeA成功例 |
| `tests/unit/agent_runtime/mtp/test_write_chain.py` / `test_fallback_when_extractor_returns_none` | [tests/unit/engines/generation/test_engine.py](../../../tests/unit/engines/generation/test_engine.py) / `test_mode_b_fallback_on_extract_failure`；owner已断言fallback输出实际content，比原仅len+notNone更强 |
| `tests/unit/agent_runtime/mtp/test_write_chain.py` / `test_fallback_draft_content` | [tests/unit/engines/generation/test_engine.py](../../../tests/unit/engines/generation/test_engine.py) / `test_build_fallback_draft`；owner已有title/content/confidence/tags/has_value；原额外memory_type=FACT合入 |
| `tests/unit/agent_runtime/mtp/test_write_chain.py` / `test_fallback_draft_no_title` | [tests/unit/engines/generation/test_engine.py](../../../tests/unit/engines/generation/test_engine.py) / `test_build_fallback_draft_no_title`；原>50字符串截断样例替换owner当前短字符串，保留截断断言 |
| `tests/unit/agent_runtime/mtp/test_write_chain.py` / `test_request_param_mode_a` | [tests/unit/engines/generation/test_engine.py](../../../tests/unit/engines/generation/test_engine.py) / `test_mode_a_extract_returns_none`；同样context非空+extractNone返回[] |
| `tests/unit/agent_runtime/mtp/test_write_chain.py` / `test_empty_request_returns_empty` | [tests/unit/engines/generation/test_engine.py](../../../tests/unit/engines/generation/test_engine.py) / `test_empty_messages_no_focus_returns_empty`；同GenerationRequest()早返回，owner还有不调用extract断言 |
| `tests/unit/agent_runtime/mtp/test_update_chain.py` / `test_mode_c_calls_merge_not_extract` | [tests/unit/engines/generation/test_engine.py](../../../tests/unit/engines/generation/test_engine.py) / `test_routes_to_mode_c`；合入extract未调用、metadata.mode/update instruction断言 |
| `tests/unit/agent_runtime/mtp/test_update_chain.py` / `test_mode_c_returns_updated_memory` | [tests/unit/engines/generation/test_engine.py](../../../tests/unit/engines/generation/test_engine.py) / `test_mode_c_merge_success`；合入id与alias不变断言 |
| `tests/unit/agent_runtime/mtp/test_update_chain.py` / `test_fallback_when_merge_returns_none` | [tests/unit/engines/generation/test_engine.py](../../../tests/unit/engines/generation/test_engine.py) / `test_mode_c_fallback_on_merge_failure`；合入旧内容保留断言 |
| `tests/unit/agent_runtime/mtp/test_update_chain.py` / `test_fallback_content_append` | [tests/unit/engines/generation/test_engine.py](../../../tests/unit/engines/generation/test_engine.py) / `test_mode_c_fallback_on_merge_failure`；process现有now参数可传FIXED_NOW，合入日期2026-09-01与Fallback changelog，避免新私有方法用例 |
| `tests/unit/agent_runtime/mtp/test_update_chain.py` / `test_fallback_instruction_only` | [tests/unit/engines/generation/test_engine.py](../../../tests/unit/engines/generation/test_engine.py) / `test_mode_c_fallback_no_content`；合入Fallback与instruction changelog断言 |
| `tests/unit/agent_runtime/mtp/test_update_chain.py` / `test_mode_c_no_existing_memory_returns_empty` | [tests/unit/engines/generation/test_engine.py](../../../tests/unit/engines/generation/test_engine.py) / `test_mode_c_no_existing_memory`；触发和断言完全一致 |
| `tests/unit/agent_runtime/mtp/test_update_chain.py` / `test_version_not_touched_by_engine` | [tests/unit/engines/generation/test_engine.py](../../../tests/unit/engines/generation/test_engine.py) / `test_apply_update_merges_content_and_captures_before_snapshot`；owner已有version/updated_at，补decay_anchor不变 |
| `tests/unit/agent_runtime/mtp/test_update_chain.py` / `test_content_updated` | [tests/unit/engines/generation/test_engine.py](../../../tests/unit/engines/generation/test_engine.py) / `test_apply_update_merges_content_and_captures_before_snapshot`；owner已有内容更新 |
| `tests/unit/agent_runtime/mtp/test_update_chain.py` / `test_outcome_records_changelog_and_before_snapshot` | [tests/unit/engines/generation/test_engine.py](../../../tests/unit/engines/generation/test_engine.py) / `test_apply_update_merges_content_and_captures_before_snapshot`；owner已有深拷贝before旧content/version与changelog |
| `tests/unit/agent_runtime/mtp/test_update_chain.py` / `test_confidence_untouched_by_engine` | [tests/unit/engines/generation/test_engine.py](../../../tests/unit/engines/generation/test_engine.py) / `test_apply_update_merges_content_and_captures_before_snapshot`；owner已有confidence=0.42保持 |
| `tests/unit/agent_runtime/mtp/test_update_chain.py` / `test_multiple_updates_snapshot_each_round` | [tests/unit/engines/generation/test_engine.py](../../../tests/unit/engines/generation/test_engine.py)；首个 before 保留初始正文，第二个 before 保留 `v2 content`，两个 `meta.version` 均保持 1；后续修订不得污染前轮快照，引擎不推进版本。现有 owner 单轮例未覆盖这一隔离性。 |

独有的 metadata、fallback 日期/changelog、长标题截断、decay anchor 等断言必须先迁入，不能直接丢弃。
18 例中仅多轮 before-snapshot 需要保留一个迁入 owner 的独立场景，名义净减 17 个；
owner 自身 `test_empty_messages_no_focus_returns_empty` 与 `test_mode_a_empty_messages` 还可合并，前者断言更强。
本组尚未执行删除后的覆盖对照，不能将 17 或 18 个记为已验证删减。

### 3.2 第一轮之后仍有类型层测试

以下五例都通过省略 Python 签名中的必填参数触发 `TypeError`，未进入生产函数体，按规范 §4.4 可退役。
真正的资源归属、准入和拒绝前无副作用测试继续保留。

- [filter adapter](../../../tests/unit/engines/retrieval/test_filter_adapter.py)：`test_filter_converter_requires_actor_for_visibility_query`；
- [ArtifactStore](../../../tests/unit/engines/artifacts/test_artifact_store.py)：`test_artifact_reads_require_workspace_ownership`；
- [意图提交](../../../tests/unit/patchouli/application/test_submission_services.py)：`test_submit_memory_intent_requires_identity_scope`；
- [Profile 读取](../../../tests/unit/patchouli/services/test_retrieval.py)：`test_custom_profile_requires_owner_and_actor_arguments`；
- [finalize](../../../tests/unit/patchouli/test_service_gateway_decision.py)：`test_finalize_requires_explicit_stage_scope`。

[公共路由测试](../../../tests/unit/system/contracts/test_public_routes.py)中
`test_patchouli_chat_route_signatures` 与 `test_alice_run_routes_receive_cpu_input_manifest` 还直接冻结类型注解、参数集合及默认值。
签名与类型由静态门禁保护；若涉及 CPU 身份隔离，复用 `tests/unit/architecture/` 的安全结构守卫，勿另建运行时签名快照。

### 3.3 名称和包目录

`integration/mtp/test_koakuma_write_e2e.py` 与 `test_koakuma_update_e2e.py` 使用真实 parser、handler、workspace 操作入口、授权与读取视图，
存储边界为替身，属于 integration。建议改为 `test_write_chain.py` / `test_update_chain.py` 并修改类名中的 E2E。
快照提交中的名称引用只出现在对应 Todo 与历史归档说明；改名时更新活动入口，历史说明不批量改写。
WRITE fixture 中 `mock_librarian` 未注入任何被测对象，`sample_memory`/`identity` 随之形成无效布置，可在同批去掉。

以下 9 个直接含测试文件的目录尚无 `__init__.py`；当前收集成功，记录的是后续通用文件名的冲突风险，不是当前失败：

```text
tests/integration/system
tests/integration/system/application
tests/unit/alice/application
tests/unit/alice/orchestration
tests/unit/alice/orchestration/sub_agent
tests/unit/core
tests/unit/gateway/commands
tests/unit/patchouli/memory_library/adapters
tests/unit/prompts
```

统一补包应单独做完整收集验证，不计为测试数量减少。

## 4. 全部 83 个参数函数的逐项结论

“候选”表示后续收敛建议；建议数为 0 的行需要先合入列明的承接测试。
“待核定”维持原数量，不计入可删量。生产入口列保留首个可定位入口；完整归属与承接条件写在判断中。

### Workspace、System、Server 与 Components

| 测试文件 / 函数 | 现有 → 建议 | 结论、生产判断与承接条件 |
|:---|:---:|:---|
| [integration/system/services/attachments/test_docx_parser.py](../../../tests/integration/system/services/attachments/test_docx_parser.py) / `test_unsupported_structures_fail_whole_document` | 6 → 6 | **保留**：ins/del虽共用revision文案，却是两个独立XML标签映射键；altChunk/sdt另有映射键，AlternateContent与嵌套表分别有检查。 承接：本函数即DOCX parser与真实python-docx容器协作的归属层矩阵；不是枚举同一错误码。。 入口：[src/hivememory/infrastructure/attachments/docx_parser.py:75](../../../src/hivememory/infrastructure/attachments/docx_parser.py)。 |
| [integration/system/test_memory_task_access_api.py](../../../tests/integration/system/test_memory_task_access_api.py) / `test_http_foreign_task_is_hidden_without_cancelling_queue` | 4 → 2 | **候选**：保留observe与cancel两个HTTP入口，各取同owner其他workspace一例；other-owner与other-workspace在资源归属全值比较处汇合。 承接：tests/unit/patchouli/application/test_memory_task_access.py:91; :121; :136。HTTP仅保留两个入口各自404映射/无取消副作用。未运行HTTP。。 入口：[src/hivememory/patchouli/application/memory_task_management_service.py:75](../../../src/hivememory/patchouli/application/memory_task_management_service.py)。 |
| [integration/workspace/capability/test_assets.py](../../../tests/integration/workspace/capability/test_assets.py) / `test_upload_rejects_invalid_or_path_like_names_with_no_side_effect` | 3 → 1 | **候选**：保留空白归一化后为空的一例，并将异常收紧到InvalidAttachmentNameError。空串被空白样例支配；../etc/passwd清理后只是无批准扩展名，重复通用格式拒绝。 承接：同文件test_upload_sanitizes_separators_and_control_characters:201与test_upload_rejects_unapproved_format_before_store_write:253承接另外两种行为。未运行上传。。 入口：[src/hivememory/workspace/assets/upload.py:45](../../../src/hivememory/workspace/assets/upload.py)。 |
| [integration/workspace/test_operation_requests.py](../../../tests/integration/workspace/test_operation_requests.py) / `test_request_without_required_operation_is_denied_before_side_effects` | 6 → 6 | **保留**：六种请求映射到六个actor方法/operation，各方法各自调用授权，能分别漏接或误接。 承接：本函数是请求入口到actor授权接线的权威矩阵；authorizer共享白名单判断不替代逐入口接线。。 入口：[src/hivememory/workspace/capability/operations.py:35](../../../src/hivememory/workspace/capability/operations.py)。 |
| [integration/workspace/test_operation_requests.py](../../../tests/integration/workspace/test_operation_requests.py) / `test_paired_operations_remain_independent_after_an_authorized_request` | 4 → 4 | **待核定**：四个拒绝映射已由冷态矩阵覆盖；先成功操作后形成缓存/意图状态可能有附加价值，但当前actor授权发生在这些状态访问前。须由owner明确要守护哪类热态回归，再决定4→2或1，不纳入高置信删减。 承接：同文件test_request_without_required_operation_is_denied_before_side_effects:416与六个only-operation成功例；目前保留。。 入口：[src/hivememory/workspace/capability/actor/memory.py:72](../../../src/hivememory/workspace/capability/actor/memory.py)。 |
| [integration/workspace/test_process_intents.py](../../../tests/integration/workspace/test_process_intents.py) / `test_process_terminal_status_preserves_dispatched_intents` | 6 → 3 | **候选**：移除new_topic轴，保留三个终态；测试的prepare/finalize/cleanup均为替身，进程未检查is_new_topic，只转发topic_id。 承接：同函数status三例保留终态与写入不回滚；真实新话题清理由Patchouli owner承担。。 已做局部筛选覆盖对照。 入口：[src/hivememory/workspace/process/allocation.py:185](../../../src/hivememory/workspace/process/allocation.py)。 |
| [integration/workspace/test_process_threads.py](../../../tests/integration/workspace/test_process_threads.py) / `test_derivation_rejects_invalid_caller_or_phase_without_registration` | 7 → 5 | **候选**：保留child/missing/finalize/revoked/stopped；other-process与ended均落在main context对象身份不匹配检查，已由child覆盖同一判断。 承接：同文件test_child_snapshots_follow_exact_callee_record_and_all_threads_are_revoked:60另保统一撤销；每个独立布尔guard仍有代表。。 已做局部筛选覆盖对照。 入口：[src/hivememory/workspace/process/threads.py:43](../../../src/hivememory/workspace/process/threads.py)。 |
| [integration/workspace/test_reference_citations.py](../../../tests/integration/workspace/test_reference_citations.py) / `test_unreadable_redirect_hides_target_and_does_not_record_citation` | 3 → 3 | **待核定**：missing/private/other_workspace实际触发AliasResolver.read的缺失、冷读policy、冷读归属三个不同判断；引用不发生在同一守卫，但下层fake会提前过滤private，直接删为一例可能削弱真实冷读边界。先迁归AliasResolver并补齐未过滤backing的归属测试，再收敛宽集成。 承接：tests/unit/workspace/resolution/test_alias_resolver.py:366保护redirect脱敏，但其_FakeBacking:103-106预先做policy过滤；:210保护冷读越域。暂不删除。。 入口：[src/hivememory/workspace/resolution/alias.py:160](../../../src/hivememory/workspace/resolution/alias.py)。 |
| [integration/workspace/test_root_api_authorization.py](../../../tests/integration/workspace/test_root_api_authorization.py) / `test_each_root_method_rejects_non_system_before_owner_or_asset_effects` | 15 → 15 | **保留**：每个root入口独立调用authorize_root，逐方法是不同接线；第一轮已去掉空/全operations笛卡尔积。 承接：同文件矩阵是归属层唯一逐方法拒绝矩阵。共享root规则在tests/unit/workspace/test_access.py:548之后单测；不能替代漏guard检测。。 入口：[src/hivememory/workspace/capability/root/memory.py](../../../src/hivememory/workspace/capability/root/memory.py)。 |
| [integration/workspace/test_root_api_authorization.py](../../../tests/integration/workspace/test_root_api_authorization.py) / `test_each_routed_root_method_reaches_owner_as_system_without_actor_operations` | 14 → 14 | **保留**：14条不同管理方法→公开route映射；误路由/错误system scope是独立接线缺陷。 承接：本函数归属root方法route接线矩阵；与拒绝矩阵保护成功与拒绝两侧不同契约。。 入口：[src/hivememory/workspace/capability/root/memory.py](../../../src/hivememory/workspace/capability/root/memory.py)。 |
| [integration/workspace/test_session_completion.py](../../../tests/integration/workspace/test_session_completion.py) / `test_output_with_each_outcome_is_submitted_before_terminal` | 3 → 3 | **保留**：有输出的三种结果进入不同终态输出类型，且会话/提交必须保留对应outcome；属于runner的结果映射矩阵。 承接：本函数为真实会话与进程协作的outcome权威矩阵；可承接其他上层对三种status的重复。。 入口：[src/hivememory/workspace/process/runner.py:298](../../../src/hivememory/workspace/process/runner.py)。 |
| [integration/workspace/test_session_completion.py](../../../tests/integration/workspace/test_session_completion.py) / `test_empty_terminal_does_not_create_turn_and_done_omits_text` | 3 → 1 | **候选**：无输出时跳过封口/会话追加在status分派前统一判断。保留cancelled一例，不能随意换completed：取消done无topic_id还覆盖outputs.py:170的false分支。 承接：同文件有输出三结局:127与unit/process/test_gateway_chat_flow.py:462保各终态映射；本函数只守共用无输出语义。。 已做局部筛选覆盖对照。 入口：[src/hivememory/workspace/process/runner.py:294](../../../src/hivememory/workspace/process/runner.py)。 |
| [integration/workspace/test_thread_context.py](../../../tests/integration/workspace/test_thread_context.py) / `test_derivation_rejects_nonprocess_or_revoked_source` | 3 → 2 | **候选**：保留revoked与request；request/thread都是有效grant但run_type不是TASK_PROCESS，走相同比较。revoked走grant None短路，独立保留。 承接：本函数保留两个实际判断分支；线程类型使用边界另有同文件test_thread_queries_reject_other_run_types:113。。 已做局部筛选覆盖对照。 入口：[src/hivememory/workspace/authentication.py:138](../../../src/hivememory/workspace/authentication.py)。 |
| [integration/workspace/test_thread_context.py](../../../tests/integration/workspace/test_thread_context.py) / `test_child_derivation_rejects_unadmitted_or_system_actor` | 3 → 3 | **保留**：missing与disabled分别测试record None和not record.enabled；system为子线程派生的额外保留身份拒绝。 承接：本函数为callee准入的最小协作层，不能把system错误分支合入一般未准入。。 入口：[src/hivememory/workspace/authentication.py:165](../../../src/hivememory/workspace/authentication.py)。 |
| [integration/workspace/test_thread_context.py](../../../tests/integration/workspace/test_thread_context.py) / `test_child_derivation_requires_valid_process_context` | 3 → 2 | **候选**：保留revoked_process_context与request_context；thread_context与request_context在同一run_type比较处汇合。 承接：主线程与子线程是两份独立实现guard，因此两函数各保两例，不整函数删除。。 已做局部筛选覆盖对照。 入口：[src/hivememory/workspace/authentication.py:162](../../../src/hivememory/workspace/authentication.py)。 |
| [integration/workspace/test_thread_context_process.py](../../../tests/integration/workspace/test_thread_context_process.py) / `test_cpu_end_revokes_thread_before_terminal_and_keeps_stage_authorization` | 5 → 2 | **候选**：保留completed和exception。completed/failed/cancelled都在结果状态分派前的同一finally撤权；exception/no-result都走execution_result None的提前撤权。 承接：unit/process/test_cpu_port.py:379保无输出failed/cancelled，:450保无终态protocol error，:483保异常；本函数仅负责线程撤销与进程阶段权限共存。。 已做局部筛选覆盖对照。 入口：[src/hivememory/workspace/process/runner.py:277](../../../src/hivememory/workspace/process/runner.py)。 |
| [unit/components/bus/test_route_arguments.py](../../../tests/unit/components/bus/test_route_arguments.py) / `test_annotations_without_runtime_class_are_not_type_checked` | 5 → 5 | **保留**：Any显式排除；Protocol为runtime class但_is_protocol；Callable为特定origin；Literal为非class origin；TypeVar本身不是class。五行覆盖不同短路路径。 承接：本函数处于类型检查器归属层，运行时检查真实存在，不是测试typing自身。。 入口：[src/hivememory/components/bus/arguments.py:102](../../../src/hivememory/components/bus/arguments.py)。 |
| [unit/components/work_queue/test_contracts.py](../../../tests/unit/components/work_queue/test_contracts.py) / `test_queue_policy_rejects_invalid_limits` | 5 → 5 | **保留**：五个字段有五条显式数值guard，任一删除都只使对应case失败。 承接：本函数处于QueuePolicy归属层。。 入口：[src/hivememory/components/work_queue/policies.py:23](../../../src/hivememory/components/work_queue/policies.py)。 |
| [unit/components/work_queue/test_payloads.py](../../../tests/unit/components/work_queue/test_payloads.py) / `test_json_encoding_rejects_values_without_stable_projection` | 3 → 3 | **保留**：不可投影对象、无序set、非有限数是三种不同JSON输入形态；NaN还依赖allow_nan=False。按规范4.3数据样例例外保留。 承接：本函数拥有canonical JSON接收边界；不是任意值复写共享context guard。。 入口：[src/hivememory/components/work_queue/payloads.py:43](../../../src/hivememory/components/work_queue/payloads.py)。 |
| [unit/components/work_queue/test_runtime.py](../../../tests/unit/components/work_queue/test_runtime.py) / `test_runtime_events_use_authoritative_transition_records_without_follow_up_get` | 4 → 4 | **保留**：成功/retry/fail/dead-letter各自调用不同store迁移API并分别传record给事件；四个调用点能独立错传。 承接：本函数为runtime迁移记录→事件接线权威矩阵。。 入口：[src/hivememory/components/work_queue/runtime.py:278](../../../src/hivememory/components/work_queue/runtime.py)。 |
| [unit/system/services/attachments/test_formats.py](../../../tests/unit/system/services/attachments/test_formats.py) / `test_unsupported_or_missing_extension_is_rejected` | 4 → 2 | **候选**：保留archive.pdf（未知扩展名）与docx（无点但像批准后缀）。doc.rtf重复未知扩展，noext重复无点分支。 承接：同文件legacy .doc特殊提示:81独立保留。。 已做局部筛选覆盖对照。 入口：[src/hivememory/infrastructure/attachments/formats.py:86](../../../src/hivememory/infrastructure/attachments/formats.py)。 |
| [unit/system/services/attachments/test_parser_dispatch.py](../../../tests/unit/system/services/attachments/test_parser_dispatch.py) / `test_approved_media_types_dispatch_to_dedicated_parsers` | 3 → 3 | **保留**：plain_text/markdown/docx三个显式if，前二者虽同parser类但content_format不同。 承接：本函数即分派表归属层矩阵。。 入口：[src/hivememory/infrastructure/attachments/parser.py:61](../../../src/hivememory/infrastructure/attachments/parser.py)。 |
| [unit/system/services/attachments/test_result_model.py](../../../tests/unit/system/services/attachments/test_result_model.py) / `test_changing_any_hashed_field_is_detectable` | 5 → 5 | **待核定**：当前只比较canonical bytes与固定原始样本，未验证变更后的content_hash；字段经过通用json.dumps而无逐字段选择分支。与同文件固定canonical/hash样本重叠，但协议是否需要字段级变更敏感性应由owner决定；不直接计为可删。 承接：同文件test_canonical_bytes_and_hash_match_independently_frozen_sample:56已保护完整序列化与hash。若保留矩阵，应使断言真正检测生产hash遗漏字段。。 入口：[src/hivememory/infrastructure/attachments/models.py:44](../../../src/hivememory/infrastructure/attachments/models.py)。 |
| [unit/system/test_patchouli_config.py](../../../tests/unit/system/test_patchouli_config.py) / `test_retriever_level_reranker_is_rejected_with_migration_hint` | 3 → 1 | **候选**：hybrid/dense/sparse在MemoryRetrievalConfig的before-validator先检查旧reranker字段并抛错，尚未进入retriever discriminator。 承接：本函数保留hybrid代表即可继续保护旧配置键迁移提示。。 已做局部筛选覆盖对照。 入口：[src/hivememory/config/patchouli.py:277](../../../src/hivememory/config/patchouli.py)。 |
| [unit/system/test_patchouli_config.py](../../../tests/unit/system/test_patchouli_config.py) / `test_removed_threshold_and_truncation_keys_fail_explicitly` | 5 → 4 | **候选**：移除hybrid-sparse：与standalone sparse同一SparseRetrieverConfig validator。保留sparse/hybrid/rrf/adaptive，因为它们四个模型分别注册各自validator，漏接是独立缺陷。 承接：本函数四例为不同配置模型接线矩阵，不能因共用_reject_removed_keys而全部只留一例。。 已做局部筛选覆盖对照。 入口：[src/hivememory/config/patchouli.py:129](../../../src/hivememory/config/patchouli.py)。 |
| [unit/workspace/capability/actor/test_memory.py](../../../tests/unit/workspace/capability/actor/test_memory.py) / `TestActorMemoryService::test_reference_status_outside_formal_delivery_does_not_record_citation` | 5 → 1 | **候选**：五种非atom/redirect状态均在同一kind排除检查处continue；保留pending且携带atom的反例即可，防止只按atom存在性记录引用。 承接：integration/workspace/test_reference_citations.py:233保真实pending链路不引用；各状态投影由AliasResolver owner测试保护。。 已做局部筛选覆盖对照。 入口：[src/hivememory/workspace/capability/actor/memory.py:269](../../../src/hivememory/workspace/capability/actor/memory.py)。 |
| [unit/workspace/process/test_command_terminal.py](../../../tests/unit/workspace/process/test_command_terminal.py) / `test_failed_parse_result_is_rejected_with_parse_error_code` | 3 → 3 | **保留**：三个解析状态对应三个稳定公共error_code输出，属于输入输出映射样例；预期错误码均独立写明。 承接：本函数为命令终态转换归属层；不在更宽流程重复失败状态矩阵。。 入口：[src/hivememory/workspace/process/command_terminal.py:39](../../../src/hivememory/workspace/process/command_terminal.py)。 |
| [unit/workspace/process/test_cpu_allocation.py](../../../tests/unit/workspace/process/test_cpu_allocation.py) / `test_manifest_operations_follow_access_record_precedence` | 3 → 1 | **候选**：CPUAllocator只调用authorizer并原样放入manifest，不实现记录优先级。保留empty-exact-record代表，最能发现空名单被用户级许可覆盖或传输丢失。 承接：unit/workspace/test_registry.py:111; :154; :165已有精确、fallback及空白名单禁用精确记录不fallback的权威用例。。 已做局部筛选覆盖对照。 入口：[src/hivememory/workspace/process/allocation.py:140](../../../src/hivememory/workspace/process/allocation.py)。 |
| [unit/workspace/process/test_gateway_chat_flow.py](../../../tests/unit/workspace/process/test_gateway_chat_flow.py) / `test_streaming_done_omits_sealing_only_result_fields` | 3 → 3 | **保留**：三个status成为RunCompleted/RunCancelled/RunFailed三个输出类型，各自从独立case调用_done_result_fields，属于真实投影接线矩阵。 承接：字段排除helper共享，但每个case可独立错误地绕过helper；目前无更低层直接stream_events矩阵承接，保留。。 入口：[src/hivememory/workspace/process/outputs.py:154](../../../src/hivememory/workspace/process/outputs.py)。 |
| [unit/workspace/resolution/test_alias_resolver.py](../../../tests/unit/workspace/resolution/test_alias_resolver.py) / `test_update_intent_on_unreadable_base_is_not_found_in_every_state` | 4 → 1 | **候选**：不可读UPDATE基础在status分支前统一返回not_found；保留materializing代表。各visible状态映射已有一般ReferenceResolution用例，不需乘UPDATE拒绝条件。 承接：同文件test_reference_results_cover_terminal_states_and_canonical_atom:400、test_settled_reference_redirect_authorizes_and_redacts_nested_target:366保状态映射；test_update_intent_on_readable_base_stays_visible_to_other_actors:509保可读UPDATE。。 已做局部筛选覆盖对照。 入口：[src/hivememory/workspace/resolution/alias.py:97](../../../src/hivememory/workspace/resolution/alias.py)。 |
| [unit/workspace/resolution/test_profile_resolver.py](../../../tests/unit/workspace/resolution/test_profile_resolver.py) / `test_builtin_profile_is_resolved_by_backing_each_time_and_not_cached` | 4 → 2 | **候选**：保留None和default覆盖空alias不查缓存/非空alias先查缓存。空串与None同归一化，omni_doll与default同非空分支；builtin映射本身由fake实现，不属于该resolver。 承接：tests/unit/patchouli/services/test_retrieval.py:147真实owner已覆盖None/空白/default/omni_doll别名归一化与fallback。。 已做局部筛选覆盖对照。 入口：[src/hivememory/workspace/resolution/profile.py:54](../../../src/hivememory/workspace/resolution/profile.py)。 |
| [unit/workspace/test_access.py](../../../tests/unit/workspace/test_access.py) / `test_context_refuses_copy_and_serialization` | 3 → 3 | **保留**：copy/deepcopy/pickle分别触发三个显式拒绝dunder，非语言天然保证。 承接：本函数为密封context不可复制契约归属层。。 入口：[src/hivememory/core/access.py:184](../../../src/hivememory/core/access.py)。 |
| [unit/workspace/test_access.py](../../../tests/unit/workspace/test_access.py) / `test_authorize_stage_rechecks_current_admission` | 4 → 2 | **候选**：method两入口与missing/disabled两种shared _redeem失败做4组合没有交互分支。保留authorize_stage×missing、get_stage_operations×disabled对角两例，同时覆盖两个入口与两个短路检查。 承接：同文件test_authorize_operation_rechecks_admission_against_the_access_registry:331保共享准入复查；对角两例仍覆盖缺失与禁用。。 已做局部筛选覆盖对照。 入口：[src/hivememory/workspace/authorization.py:139](../../../src/hivememory/workspace/authorization.py)。 |
| [unit/workspace/test_access.py](../../../tests/unit/workspace/test_access.py) / `test_authorize_stage_rejects_target_outside_resident_workspace` | 4 → 2 | **候选**：两种target在完整WorkspaceIdentity不相等检查处汇合，owner差异不是后续target_owner_mismatch分支。每方法保留一个跨workspace目标。 承接：同文件test_authorize_operation_rejects_foreign_owner_target:282已说明owner为WorkspaceIdentity坐标；分别保两方法接线。。 已做局部筛选覆盖对照。 入口：[src/hivememory/workspace/authorization.py:140](../../../src/hivememory/workspace/authorization.py)。 |
| [unit/workspace/test_sessions.py](../../../tests/unit/workspace/test_sessions.py) / `test_foreign_workspace_cannot_access_session` | 5 → 5 | **保留**：五个按session_id入口各自调用_get，漏接会导致某个入口越域，按规范4.1属于归属层入口矩阵。 承接：共享归属检查_get:62在同owner最小服务单元内；不新增更宽逐方法矩阵。。 入口：[src/hivememory/workspace/sessions/service.py:100](../../../src/hivememory/workspace/sessions/service.py)。 |
| [unit/workspace/test_sessions.py](../../../tests/unit/workspace/test_sessions.py) / `test_closed_service_rejects_reads_and_writes` | 7 → 7 | **保留**：create/list直接_require_open，其余方法各自_get→require_open；每方法需独立接线，close还清空容器，漏guard可能转为成功或错误NotFound。 承接：本函数为ConversationSessionService生命周期入口权威矩阵。。 入口：[src/hivememory/workspace/sessions/service.py:58](../../../src/hivememory/workspace/sessions/service.py)。 |

### Core、Engines、基础设施、提示词与 i18n

| 测试文件 / 函数 | 现有 → 建议 | 结论、生产判断与承接条件 |
|:---|:---:|:---|
| [unit/core/models/test_identity.py](../../../tests/unit/core/models/test_identity.py) / `test_system_actor_for_workspace_only_reads_public_policy` | 3 → 3 | **保留**：PUBLIC、PRIVATE、TEAM 三种 policy 分支；TEAM 的 system/no-team 输入额外触发短路，与普通 team-a/team-b 对照不等价。构造 helper 和普通读取谓词的组合保持 system 无特权。 承接：tests/unit/core/models/test_identity.py::test_system_actor_for_workspace_only_reads_public_policy；tests/unit/core/test_memory_access.py::test_read_policy_is_applied_within_owning_workspace。 入口：[src/hivememory/core/models/identity.py:85](../../../src/hivememory/core/models/identity.py)。 |
| [unit/core/models/test_memory.py](../../../tests/unit/core/models/test_memory.py) / `test_access_policy_rejects_invalid_target_combinations` | 5 → 5 | **保留**：五行覆盖 PUBLIC 禁止 target、PRIVATE 缺 agent/多 team、TEAM 多 agent/缺 team 的不同逻辑操作数。 承接：tests/unit/core/models/test_memory.py::test_access_policy_rejects_invalid_target_combinations。 入口：[src/hivememory/core/models/memory.py:67](../../../src/hivememory/core/models/memory.py)。 |
| [unit/core/models/test_memory.py](../../../tests/unit/core/models/test_memory.py) / `test_assignment_rejects_invalid_value_and_keeps_previous` | 7 → 7 | **保留**：七行对应三个模型的七个字段映射、字段上限/UTC/非空约束，且验证 validate_assignment 接线和失败后旧值；不能只保留一种模型。 承接：tests/unit/core/models/test_memory.py::test_assignment_rejects_invalid_value_and_keeps_previous。 入口：[src/hivememory/core/models/memory.py:107](../../../src/hivememory/core/models/memory.py)。 |
| [unit/core/models/test_topic.py](../../../tests/unit/core/models/test_topic.py) / `TestTopicDataContentEmptiness::test_blank_summary_does_not_count_as_content` | 3 → 1 | **候选**：删 empty 与 spaces，保留 whitespace（换行+制表符+空格）。empty 与现有 no-block/no-summary 用例完全重叠；另两行都只把 strip 后空串送入同一内容判断。 承接：tests/unit/core/models/test_topic.py::TestTopicDataContentEmptiness::test_blank_summary_does_not_count_as_content；tests/unit/core/models/test_topic.py::TestTopicDataContentEmptiness::test_no_blocks_no_summary_is_empty。 入口：[src/hivememory/core/models/topic.py:124](../../../src/hivememory/core/models/topic.py)。 |
| [unit/core/models/test_workspace.py](../../../tests/unit/core/models/test_workspace.py) / `test_workspace_identity_rejects_blank_coordinate` | 3 → 3 | **保留**：三个字段是 validator 的独立绑定条目；任一字段从 decorator 中误删，都只能被相应一行捕获，属于归属层逐入口接线矩阵。 承接：tests/unit/core/models/test_workspace.py::test_workspace_identity_rejects_blank_coordinate。 入口：[src/hivememory/core/models/identity.py:71](../../../src/hivememory/core/models/identity.py)。 |
| [unit/core/models/test_workspace.py](../../../tests/unit/core/models/test_workspace.py) / `test_workspace_errors_expose_stable_machine_codes` | 8 → 8 | **保留**：八个异常各自声明不同的外部机器码映射；期望是协议字面值，未从被测常量回读。保留稳定错误语义的 owner 表。 承接：tests/unit/core/models/test_workspace.py::test_workspace_errors_expose_stable_machine_codes。 入口：[src/hivememory/core/errors.py:25](../../../src/hivememory/core/errors.py)。 |
| [unit/core/mtp/test_formatter_xml_escaping.py](../../../tests/unit/core/mtp/test_formatter_xml_escaping.py) / `test_content_with_reserved_characters_round_trips_as_text` | 4 → 3 | **候选**：预编码实体行与独立 preencoded 用例输入、解析相同；先把无子节点和含 CDATA 的断言并入独立例，再删除矩阵中的相同行。其余三行保留普通保留字符、CDATA terminator、注入式嵌套 markup 三种协议输入形态。 承接：`test_preencoded_entities_are_not_escaped_again`，保留原始 wire 与不双转义断言。 入口：[src/hivememory/core/mtp/formatter.py:34](../../../src/hivememory/core/mtp/formatter.py)。 |
| [unit/core/test_access_binding.py](../../../tests/unit/core/test_access_binding.py) / `test_run_binding_rejects_inconsistent_thread_identifier` | 4 → 3 | **候选**：删 request/main，保留 task_process/main 作为非线程运行的代表。两者都走同一 elif thread_id is not None，run_type 不再分支。execution_thread None 和空白分别覆盖类型检查与 strip 非空检查。 承接：tests/unit/core/test_access_binding.py::test_run_binding_rejects_inconsistent_thread_identifier。 入口：[src/hivememory/core/access.py:74](../../../src/hivememory/core/access.py)。 |
| [unit/core/test_memory_access.py](../../../tests/unit/core/test_memory_access.py) / `test_read_policy_is_applied_within_owning_workspace` | 5 → 5 | **保留**：PUBLIC 恒可读、PRIVATE 目标匹配/不匹配、TEAM 目标匹配/不匹配，均为独立授权条件结果。 承接：tests/unit/core/test_memory_access.py::test_read_policy_is_applied_within_owning_workspace。 入口：[src/hivememory/core/memory_access.py:27](../../../src/hivememory/core/memory_access.py)。 |
| [unit/engines/gateway/test_topic_router.py](../../../tests/unit/engines/gateway/test_topic_router.py) / `test_topic_router_converts_invalid_output_to_recoverable_error` | 4 → 4 | **保留**：四行分别为不可见 topic、NEW_TOPIC 缺元数据、JSON语法错误、非object结果；前两行不同领域守卫，后两行是不同解析失败形态。 承接：tests/unit/engines/gateway/test_topic_router.py::test_topic_router_converts_invalid_output_to_recoverable_error。 入口：[src/hivememory/engines/gateway/topic_router.py:56](../../../src/hivememory/engines/gateway/topic_router.py)。 |
| [unit/engines/generation/test_alias.py](../../../tests/unit/engines/generation/test_alias.py) / `TestBuildCandidate::test_prefix_follows_memory_type` | 7 → 7 | **保留**：七个 MemoryType 到七个 alias prefix 的独立映射条目；不是同一共享失败检查重复。 承接：tests/unit/engines/generation/test_alias.py::TestBuildCandidate::test_prefix_follows_memory_type。 入口：[src/hivememory/engines/generation/alias.py:28](../../../src/hivememory/engines/generation/alias.py)。 |
| [unit/engines/retrieval/test_memory_codec.py](../../../tests/unit/engines/retrieval/test_memory_codec.py) / `test_non_current_schema_version_is_rejected` | 3 → 2 | **候选**：删整数7，保留整数2与字符串2。2/7同类型同结构同一 equality guard；整数/字符串两种实际持久化版本输入形态保留。 承接：tests/unit/engines/retrieval/test_memory_codec.py::test_non_current_schema_version_is_rejected。 入口：[src/hivememory/engines/retrieval/memory_codec.py:33](../../../src/hivememory/engines/retrieval/memory_codec.py)。 |
| [unit/i18n/test_types.py](../../../tests/unit/i18n/test_types.py) / `TestNormalizeLanguage::test_known_aliases` | 9 → 9 | **保留**：九行逐个验证生产 aliases 表的独立键；符合纯数据表式输入输出例外。 承接：tests/unit/i18n/test_types.py::TestNormalizeLanguage::test_known_aliases。 入口：[src/hivememory/i18n/types.py:18](../../../src/hivememory/i18n/types.py)。 |
| [unit/i18n/test_types.py](../../../tests/unit/i18n/test_types.py) / `TestNormalizeLanguage::test_case_insensitive_zh` | 3 → 1 | **候选**：保留 ZH 全大写形态，删 Zh/CHINESE；大小写归一化仅一处 lower，另一函数保留 English 混合大小写形态，所有 alias 键由 known_aliases 单独守护。 承接：tests/unit/i18n/test_types.py::TestNormalizeLanguage::test_case_insensitive_zh；tests/unit/i18n/test_types.py::TestNormalizeLanguage::test_known_aliases；tests/unit/i18n/test_types.py::TestNormalizeLanguage::test_case_insensitive_en。 入口：[src/hivememory/i18n/types.py:37](../../../src/hivememory/i18n/types.py)。 |
| [unit/i18n/test_types.py](../../../tests/unit/i18n/test_types.py) / `TestNormalizeLanguage::test_case_insensitive_en` | 3 → 1 | **候选**：保留 English 混合大小写形态，删 EN/En；与保留的 ZH 合计覆盖全大写/混合大小写，不交叉枚举相同 lower 路径的每个语言/别名。 承接：tests/unit/i18n/test_types.py::TestNormalizeLanguage::test_case_insensitive_en；tests/unit/i18n/test_types.py::TestNormalizeLanguage::test_known_aliases；tests/unit/i18n/test_types.py::TestNormalizeLanguage::test_case_insensitive_zh。 入口：[src/hivememory/i18n/types.py:37](../../../src/hivememory/i18n/types.py)。 |
| [unit/i18n/test_types.py](../../../tests/unit/i18n/test_types.py) / `TestNormalizeLanguage::test_unknown_returns_none` | 5 → 3 | **候选**：删 ja/de；与 fr 同为不支持的两字母语言码形态。保留 fr/unknown/123，分别是不支持语言码、未知名称、数值字符串。 承接：tests/unit/i18n/test_types.py::TestNormalizeLanguage::test_unknown_returns_none。 入口：[src/hivememory/i18n/types.py:40](../../../src/hivememory/i18n/types.py)。 |
| [unit/infrastructure/test_storage_unit.py](../../../tests/unit/infrastructure/test_storage_unit.py) / `TestQdrantMemoryStore::test_write_failure_raises_storage_write_error` | 3 → 3 | **保留**：重算向量 upsert、保留向量 set_payload、局部 patch set_payload 为不同 await 位置/try 边界；每一条都要证明客户端失败从相应入口归类为 StorageWriteError。 承接：tests/unit/infrastructure/test_storage_unit.py::TestQdrantMemoryStore::test_write_failure_raises_storage_write_error。 入口：[src/hivememory/infrastructure/storage/vector_store.py:169](../../../src/hivememory/infrastructure/storage/vector_store.py)。 |
| [unit/infrastructure/work_queue/test_in_memory_store.py](../../../tests/unit/infrastructure/work_queue/test_in_memory_store.py) / `TestInMemoryWorkStoreContract::test_successful_transition_returns_committed_record` | 4 → 4 | **保留**：四个公共 store 方法映射不同终态或 retry 状态/时间/错误；三种 terminal 虽共享 helper，目标状态映射独立。 承接：tests/unit/infrastructure/work_queue/test_in_memory_store.py::TestInMemoryWorkStoreContract::test_successful_transition_returns_committed_record。 入口：[src/hivememory/infrastructure/work_queue/in_memory_store.py:120](../../../src/hivememory/infrastructure/work_queue/in_memory_store.py)。 |
| [unit/infrastructure/work_queue/test_in_memory_store.py](../../../tests/unit/infrastructure/work_queue/test_in_memory_store.py) / `TestInMemoryWorkStoreContract::test_illegal_transition_is_rejected` | 4 → 4 | **保留**：不同公共入口/目标 WorkState 映射应分别接上 transition guard；属于最低层逐入口矩阵，不是分派前统一检查的冗余。 承接：tests/unit/infrastructure/work_queue/test_in_memory_store.py::TestInMemoryWorkStoreContract::test_illegal_transition_is_rejected。 入口：[src/hivememory/infrastructure/work_queue/in_memory_store.py:120](../../../src/hivememory/infrastructure/work_queue/in_memory_store.py)。 |
| [unit/prompts/test_mtp_permissions.py](../../../tests/unit/prompts/test_mtp_permissions.py) / `test_write_update_instruction_does_not_imply_read_permission` | 4 → 4 | **保留**：verb×language 的四行各读取不同文案映射条目；任一条 WRITE/UPDATE 中英文文案都可能独立误教 READ，因此四种输出都有价值。 承接：tests/unit/prompts/test_mtp_permissions.py::test_write_update_instruction_does_not_imply_read_permission。 入口：[src/hivememory/prompts/mtp.py:135](../../../src/hivememory/prompts/mtp.py)。 |
| [unit/prompts/test_mtp_permissions.py](../../../tests/unit/prompts/test_mtp_permissions.py) / `test_demo_requires_every_verb_used_by_the_example` | 8 → 4 | **候选**：保留四种 permitted_verbs，仅取 en；三种缺失成员分别守护 _DEMO_VERBS 的三个要求。language 不参与 subset gate，正/反双语言模板在已有 read_search_run/run_only 测试覆盖。 承接：tests/unit/prompts/test_mtp_permissions.py::test_demo_requires_every_verb_used_by_the_example；tests/unit/prompts/test_mtp_permissions.py::test_read_search_run_prompt_omits_other_verbs_from_all_teaching；tests/unit/prompts/test_mtp_permissions.py::test_run_only_prompt_omits_resource_teaching_and_preserves_all_system_tools。 入口：[src/hivememory/prompts/mtp.py:53](../../../src/hivememory/prompts/mtp.py)。 |

### Alice、MTP 与会话 E2E

| 测试文件 / 函数 | 现有 → 建议 | 结论、生产判断与承接条件 |
|:---|:---:|:---|
| [e2e/system/test_chat_session.py](../../../tests/e2e/system/test_chat_session.py) / `test_text_only_cpu_output_is_structured_before_patchouli_submission` | 3 → 1 | **候选**：三终态的提交和会话映射已由进程归属层矩阵保护，只有正文时补事件由seal_interaction单测保护；HTTP到真实Patchouli保留completed代表，不再重跑终态矩阵。 承接：tests/integration/workspace/test_session_completion.py::test_output_with_each_outcome_is_submitted_before_terminal；tests/unit/workspace/process/test_seal_interaction.py::test_seal_interaction_without_turn_events_has_empty_mtp_traces。 入口：[src/hivememory/workspace/process/runner.py:294](../../../src/hivememory/workspace/process/runner.py)。 |
| [integration/alice/application/test_agent_run_service.py](../../../tests/integration/alice/application/test_agent_run_service.py) / `test_registered_labels_drive_runtime_events_and_profile_controls_stream_display` | 3 → 2 | **候选**：保留显式display_agent和None回退；空字符串与None均在同一个or表达式回退注册标签，没有第二个身份或事件分支。 承接：keep display_agent and None。 入口：[src/hivememory/alice/application/agent_run_service.py:385](../../../src/hivememory/alice/application/agent_run_service.py)。 |
| [integration/alice/orchestration/test_call_profile_requests.py](../../../tests/integration/alice/orchestration/test_call_profile_requests.py) / `test_builtin_call_uses_workspace_profile_read_and_keeps_builtin_configuration` | 3 → 1 | **候选**：空alias/default/omni_doll的真实归一化归Patchouli，owner已有五种输入全表；CALL链保留omni_doll证明经过真实backing且不缓存。 承接：tests/unit/patchouli/services/test_retrieval.py::TestRetrievalFamiliarAgentProfiles::test_unspecified_or_builtin_profile_uses_explicit_fallback。 入口：[src/hivememory/patchouli/services/retrieval.py:202](../../../src/hivememory/patchouli/services/retrieval.py)。 |
| [integration/alice/orchestration/test_call_profile_requests.py](../../../tests/integration/alice/orchestration/test_call_profile_requests.py) / `test_call_without_call_permission_returns_permission_error_without_dispatch` | 4 → 1 | **候选**：四个alias在ActorAgentService第44行授权前汇合，拒绝发生在Profile解析之前。保留内置omni_doll，防止内置目标旁路CALL授权。 承接：tests/unit/workspace/capability/actor/test_agent_profiles.py::test_call_target_without_operation_denied_before_backing；tests/integration/workspace/test_operation_requests.py::test_request_without_required_operation_is_denied_before_side_effects。 入口：[src/hivememory/workspace/capability/actor/agent_profiles.py:44](../../../src/hivememory/workspace/capability/actor/agent_profiles.py)。 |
| [integration/alice/orchestration/test_call_profile_requests.py](../../../tests/integration/alice/orchestration/test_call_profile_requests.py) / `test_unusable_custom_profile_returns_preparation_error_without_fallback` | 3 → 1 | **候选**：缺失/私有/损坏的完整策略矩阵归Profile读取owner与资源可见性；错误类型翻译归CallContextProvider。上层保留损坏配置代表，确认真实拒绝被回填且不创建子帧。 承接：tests/unit/patchouli/services/test_retrieval.py::TestRetrievalFamiliarAgentProfiles；tests/unit/alice/orchestration/sub_agent/test_call_context_provider.py::test_profile_failure_maps_to_mtp_error_before_resolving_context_refs；tests/unit/core/test_memory_access.py::test_read_policy_is_applied_within_owning_workspace。 入口：[src/hivememory/patchouli/services/retrieval.py:206](../../../src/hivememory/patchouli/services/retrieval.py)。 |
| [integration/alice/orchestration/test_call_profile_requests.py](../../../tests/integration/alice/orchestration/test_call_profile_requests.py) / `test_unadmitted_call_target_returns_structured_error_without_creating_child` | 3 → 1 | **候选**：missing/disabled/system三种准入拒绝由认证归属层分别保护；Alice这里只需证明AdmissionDenied映射为结构化错误，且真实进程不登记子线程。保留missing。 承接：tests/integration/workspace/test_thread_context.py::test_child_derivation_rejects_unadmitted_or_system_actor。 入口：[src/hivememory/alice/orchestration/sub_agent/call_context_provider.py:67](../../../src/hivememory/alice/orchestration/sub_agent/call_context_provider.py)。 |
| [integration/alice/orchestration/test_call_profile_requests.py](../../../tests/integration/alice/orchestration/test_call_profile_requests.py) / `test_failed_child_preserves_dispatched_intents` | 3 → 1 | **候选**：FAILED/CANCELLED/BUDGET_EXHAUSTED只改变CALL响应，不改变意图登记；响应映射有独立owner测试。保留FAILED链路及CALL返回后子线程仍可提交的断言，同时吸收unit同类四终态矩阵。 承接：tests/unit/alice/orchestration/sub_agent/test_call_response.py；tests/integration/workspace/test_process_intents.py::test_process_terminal_status_preserves_dispatched_intents。 入口：[src/hivememory/alice/orchestration/sub_agent/call_coordinator.py:230](../../../src/hivememory/alice/orchestration/sub_agent/call_coordinator.py)。 |
| [integration/mtp/test_search_source_agent.py](../../../tests/integration/mtp/test_search_source_agent.py) / `test_search_agent_filter_matches_source_or_contributor_and_excludes_other_sources` | 4 → 2 | **候选**：保留source-only和system-contributor，分别证明真实Qdrant对OR两分支的匹配；other-source-contributor与system-contributor同分支，source-with-contributors被source-only及contributor-only覆盖。 承接：tests/unit/engines/retrieval/test_filter_adapter.py::test_source_agent_filter_matches_contributors_and_source_branches；keep source-only and system-contributor。 入口：[src/hivememory/engines/retrieval/filter_adapter.py:126](../../../src/hivememory/engines/retrieval/filter_adapter.py)。 |
| [integration/mtp/test_search_source_agent.py](../../../tests/integration/mtp/test_search_source_agent.py) / `test_search_agent_filter_intersects_callers_resource_visibility` | 4 → 4 | **保留**：当前是Qdrant内存适配器真正执行PRIVATE/TEAM目标相等与不等筛选的四种语义，core的Python谓词和converter结构断言不能替代。后续可把全矩阵下移到最小适配器集成层，MTP保留一例；本轮不直接计入可删量。 承接：tests/unit/core/test_memory_access.py::test_read_policy_is_applied_within_owning_workspace；tests/unit/engines/retrieval/test_filter_adapter.py::test_v2_actor_policy_targets_are_distinct_from_provenance。 入口：[src/hivememory/engines/retrieval/filter_adapter.py:159](../../../src/hivememory/engines/retrieval/filter_adapter.py)。 |
| [unit/alice/application/test_operation_policy.py](../../../tests/unit/alice/application/test_operation_policy.py) / `test_each_actor_operation_maps_to_its_verb_and_system_run` | 6 → 6 | **保留**：六项operation到动词的独立映射条目，属于翻译owner全矩阵。 入口：[src/hivememory/alice/application/operation_policy.py:10](../../../src/hivememory/alice/application/operation_policy.py)。 |
| [unit/alice/orchestration/sub_agent/test_call_context_provider.py](../../../tests/unit/alice/orchestration/sub_agent/test_call_context_provider.py) / `test_profile_failure_maps_to_mtp_error_before_resolving_context_refs` | 5 → 5 | **保留**：OperationDenied/AdmissionDenied/ResourceUnavailable/MTPError/其他异常对应五个独立except及message key映射。 入口：[src/hivememory/alice/orchestration/sub_agent/call_context_provider.py:61](../../../src/hivememory/alice/orchestration/sub_agent/call_context_provider.py)。 |
| [unit/alice/orchestration/sub_agent/test_call_coordinator.py](../../../tests/unit/alice/orchestration/sub_agent/test_call_coordinator.py) / `test_callee_terminal_status_preserves_its_acknowledged_intents` | 4 → 0 | **候选**：与CALL集成中的子帧意图保留重复；本测试引入真实Workspace harness但coordinator协作者全替身，不是意图状态owner。合入test_failed_child_preserves_dispatched_intents并保留其更强子线程寿命断言，删除四例。 承接：tests/integration/alice/orchestration/test_call_profile_requests.py::test_failed_child_preserves_dispatched_intents；tests/integration/workspace/test_process_intents.py::test_process_terminal_status_preserves_dispatched_intents。 入口：[src/hivememory/alice/orchestration/sub_agent/call_coordinator.py:230](../../../src/hivememory/alice/orchestration/sub_agent/call_coordinator.py)。 |
| [unit/alice/orchestration/test_run_executor.py](../../../tests/unit/alice/orchestration/test_run_executor.py) / `test_run_executor_finalizes_each_terminal_status_exactly_once` | 4 → 4 | **保留**：四种终态是显式match接受集合的四个成员，漏任意成员会进入unsupported分支；保留逐成员归属层覆盖。测试只检查terminal_result，函数名中的exactly_once应改为与断言一致。 入口：[src/hivememory/alice/orchestration/run_executor.py:142](../../../src/hivememory/alice/orchestration/run_executor.py)。 |

### Patchouli

| 测试文件 / 函数 | 现有 → 建议 | 结论、生产判断与承接条件 |
|:---|:---:|:---|
| [integration/patchouli/test_settlement_identity.py](../../../tests/integration/patchouli/test_settlement_identity.py) / `test_all_settlement_triggers_create_system_initiated_tasks` | 4 → 0 | **候选**：四种触发在PerceptionFamiliar._settle_topic先汇合，system actor仅由Coordinator.submit_settlement统一组装；完整idle落库测试已验证同链system身份。加强coordinator现有spec测试的from_actor断言后可删除四例。 承接：tests/unit/patchouli/control/test_memory_generation_coordinator.py::TestMemoryGenerationCoordinator::test_submit_settlement_builds_settle_spec (line 184; add from_actor assertion)；tests/integration/patchouli/test_settlement_identity.py::test_idle_settlement_persists_public_memory_without_merging_similar_private (line 255)。 入口：[src/hivememory/patchouli/services/perception.py:182](../../../src/hivememory/patchouli/services/perception.py)。 |
| [integration/patchouli/test_settlement_identity.py](../../../tests/integration/patchouli/test_settlement_identity.py) / `test_settle_skips_restricted_dedup_but_write_keeps_actor_visibility` | 4 → 2 | **候选**：active_write区分Engine两种调用模式，应保留；PRIVATE/TEAM分支属于core memory_is_readable且那里已有全矩阵，本集成层各模式取PRIVATE一个代表即可。 承接：tests/unit/core/test_memory_access.py::test_read_policy_is_applied_within_owning_workspace (line 94)；keep settle-PRIVATE and write-PRIVATE。 入口：[src/hivememory/engines/generation/engine.py](../../../src/hivememory/engines/generation/engine.py)。 |
| [unit/patchouli/application/test_memory_task_access.py](../../../tests/unit/patchouli/application/test_memory_task_access.py) / `test_non_scope_identity_rejected_before_reaching_task_control` | 3 → 3 | **保留**：list/get/cancel三个入口分别显式调用require_identity_scope，逐入口防漏接判断属于归属层矩阵；不是共享分派前检查。 入口：[src/hivememory/patchouli/application/memory_task_management_service.py:33](../../../src/hivememory/patchouli/application/memory_task_management_service.py)。 |
| [unit/patchouli/application/test_submission_services.py](../../../tests/unit/patchouli/application/test_submission_services.py) / `test_memory_intent_requires_registered_identifiers` | 3 → 3 | **保留**：topic_id/intent_id/pending_alias是显式校验字段列表的三个独立条目，删除任意名字只对应一例失败；公共dataclass确有自写运行时守卫。可后续迁到contracts归属但无需减case。 入口：[src/hivememory/patchouli/contracts/memory_intent.py:46](../../../src/hivememory/patchouli/contracts/memory_intent.py)。 |
| [unit/patchouli/control/test_memory_generation_events.py](../../../tests/unit/patchouli/control/test_memory_generation_events.py) / `test_terminal_event_maps_domain_status` | 3 → 3 | **保留**：三个终态分别映射三个RuntimeEventType并分配severity，是不同映射条目。 入口：[src/hivememory/patchouli/control/memory_generation/events.py:13](../../../src/hivememory/patchouli/control/memory_generation/events.py)。 |
| [unit/patchouli/control/test_memory_generation_models.py](../../../tests/unit/patchouli/control/test_memory_generation_models.py) / `test_memory_generation_task_maps_work_state_to_domain_status` | 7 → 7 | **保留**：七个WorkState属于显式映射字典七条目；即使多个输出相同，也能单独改坏映射条目。 入口：[src/hivememory/patchouli/control/memory_generation/models.py:32](../../../src/hivememory/patchouli/control/memory_generation/models.py)。 |
| [unit/patchouli/memory_library/adapters/test_mid_term.py](../../../tests/unit/patchouli/memory_library/adapters/test_mid_term.py) / `test_patch_payload_rejects_invalid_values_without_write` | 3 → 2 | **候选**：adapter有access_policy赋值和lifecycle setattr两条路径，保留非法PRIVATE缺target与confidence越界即可。garbage类型和PRIVATE域约束都进入同一policy赋值分支，域约束全矩阵已有core模型测试。 承接：tests/unit/core/models/test_memory.py::test_access_policy_rejects_invalid_target_combinations (line 80)；tests/unit/core/models/test_memory.py::test_access_policy_assignment_revalidates_invariants (line 154)；keep patch1 and patch2。 入口：[src/hivememory/patchouli/memory_library/adapters/mid_term.py:88](../../../src/hivememory/patchouli/memory_library/adapters/mid_term.py)。 |
| [unit/patchouli/memory_library/test_memory_change_events.py](../../../tests/unit/patchouli/memory_library/test_memory_change_events.py) / `test_each_mutation_publishes_one_resource_coordinate` | 4 → 4 | **保留**：四个Store公开提交入口各有自己的finally与通知载荷组装，防漏通知/错operation需要逐入口覆盖。 入口：[src/hivememory/patchouli/memory_library/stores.py:156](../../../src/hivememory/patchouli/memory_library/stores.py)。 |
| [unit/patchouli/memory_library/test_memory_change_events.py](../../../tests/unit/patchouli/memory_library/test_memory_change_events.py) / `test_primary_failure_still_invalidates_and_preserves_storage_error` | 4 → 4 | **保留**：每个mutation入口独立try/finally，primary失败不能绕开各自通知；不是一次共享检查的四值重复。 入口：[src/hivememory/patchouli/memory_library/stores.py:163](../../../src/hivememory/patchouli/memory_library/stores.py)。 |
| [unit/patchouli/memory_library/test_memory_change_events.py](../../../tests/unit/patchouli/memory_library/test_memory_change_events.py) / `test_secondary_partial_failure_keeps_primary_commit_and_invalidates` | 4 → 4 | **保留**：四个入口有独立secondary遍历及finally，结果分别包含upsert/patch/delete状态，部分提交失败分支独立。 入口：[src/hivememory/patchouli/memory_library/stores.py:166](../../../src/hivememory/patchouli/memory_library/stores.py)。 |
| [unit/patchouli/runtime/test_core.py](../../../tests/unit/patchouli/runtime/test_core.py) / `TestRuntimeRerankerAssembly::test_enabled_reranker_is_built_for_any_retriever_type` | 3 → 1 | **候选**：初始化只读reranker.enabled及其配置；retriever type不参与此判断，且测试把_build_engines整体patch掉。dense/sparse/hybrid执行完全同一路径，保留dense代表。 承接：keep dense; keep separate test_disabled_reranker_is_not_built。 入口：[src/hivememory/patchouli/runtime/core.py:332](../../../src/hivememory/patchouli/runtime/core.py)。 |
| [unit/patchouli/services/test_external_edit_validation.py](../../../tests/unit/patchouli/services/test_external_edit_validation.py) / `test_invalid_edit_is_rejected_without_commit` | 3 → 3 | **保留**：title/summary/alias为外部编辑映射不同字段，分别验证字段传递和领域错误映射，并确保拒绝发生在commit前；不是同一种值笛卡尔积。 入口：[src/hivememory/patchouli/services/memory_generation.py:243](../../../src/hivememory/patchouli/services/memory_generation.py)。 |
| [unit/patchouli/services/test_retrieval.py](../../../tests/unit/patchouli/services/test_retrieval.py) / `TestRetrievalFamiliarAgentProfiles::test_unspecified_or_builtin_profile_uses_explicit_fallback` | 5 → 5 | **保留**：None/空串/空白是不同归一化输入形态；default/omni_doll为两个内置别名条目，属于规范§4.3明确允许的别名归一化输入表。 入口：[src/hivememory/patchouli/services/retrieval.py:202](../../../src/hivememory/patchouli/services/retrieval.py)。 |

## 5. 实际验证与限制

- 完整收集：`PYTHONPATH=/tmp:src LITELLM_LOCAL_MODEL_COST_MAP=True .venv/bin/python -m pytest -p hivememory_followup_collection --collect-only -o addopts='' -q`，2909 collected；临时插件只记录 nodeid、marker、位置，不修改收集结果。
- Core/Engines/i18n/提示词等 21 个矩阵：99 passed in 2.75s。
- Alice/MTP 的 12 个矩阵（不含 HTTP E2E）：46 passed in 0.20s。
- Workspace/线程/配置等定向基线：112 passed in 2.58s。
- 上述五个缺参测试及两项签名快照：7 passed in 2.45s；通过只证明这些现有测试可运行，不表示值得保留。
- Patchouli 13 个矩阵的合并运行触发 50 秒时限：只有 42 个成功进度标记，没有 pytest 最终总结；最后的 Profile 别名 5 例单跑 5 passed in 2.06s；外部编辑 3 例未完成验证，不记为全通过。

Core/Engines 采样使用 `-q --no-cov -m 'not live_llm and not real_infra and not slow'`；
Alice/MTP 与类型层采样另使用 `-o addopts=''`，并排除 `e2e/live_llm/real_infra/slow`。
Workspace 初始基线使用 `-q -o addopts=''`，明确选择纯代码测试文件，未包含 HTTP、上传或真实资源。未运行依赖真实 LLM 或独立 Qdrant 服务的测试；
MTP SEARCH 使用本地 Qdrant 内存模式和确定性 embedding。

### 局部删减选择对照

15 个矩阵从 63 个选择项减为 31 个，并在两次运行中保留相同承接测试：

| 对照 | 原选择 | 保留选择 |
|:---|---:|---:|
| pytest 结果 | 91 passed，7.09s | 59 passed，6.85s |
| 源码执行行 | 7714 | 7714 |
| 源码执行分支 | 377 | 377 |

逐文件比较 `executed_lines` 与 `executed_branches`，无丢失；这只是局部运行，不是完整快速集覆盖率。
`test_empty_terminal_does_not_create_turn_and_done_omits_text` 必须保留 `cancelled` 代表；随意换成 completed 会失去取消输出无 topic_id 分支。
本轮不根据这组耗时推算整套 CI 节省。

两次实际运行都使用 `--cov=hivememory --cov-branch --cov-report=json:... --cov-report=`；
各自 `COVERAGE_FILE`、JSON 和日志都在 `/tmp`。初次导入触发 LiteLLM 价格表自动下载并因 DNS 失败回退本地，
后续明确设置 `LITELLM_LOCAL_MODEL_COST_MAP=True`；未发生 LLM 推理调用。

筛选对照保留的参数 ID：

| 参数函数（表中同名行） | 保留 ID |
|:---|:---|
| `test_process_terminal_status_preserves_dispatched_intents` | `False-completed`、`False-cancelled`、`False-failed` |
| `test_derivation_rejects_invalid_caller_or_phase_without_registration` | `child`、`missing`、`finalize`、`revoked`、`stopped` |
| `test_empty_terminal_does_not_create_turn_and_done_omits_text` | `cancelled` |
| `test_derivation_rejects_nonprocess_or_revoked_source` | `revoked`、`request` |
| `test_child_derivation_requires_valid_process_context` | `revoked_process_context`、`request_context` |
| `test_cpu_end_revokes_thread_before_terminal_and_keeps_stage_authorization` | `completed`、`exception` |
| `test_unsupported_or_missing_extension_is_rejected` | `archive.pdf`、`docx` |
| `test_retriever_level_reranker_is_rejected_with_migration_hint` | `hybrid` |
| `test_removed_threshold_and_truncation_keys_fail_explicitly` | `sparse`、`hybrid`、`rrf`、`adaptive` |
| `test_reference_status_outside_formal_delivery_does_not_record_citation` | `pending` |
| `test_manifest_operations_follow_access_record_precedence` | `empty-exact-record` |
| `test_update_intent_on_unreadable_base_is_not_found_in_every_state` | `materializing` |
| `test_builtin_profile_is_resolved_by_backing_each_time_and_not_cached` | `None`、`default` |
| `test_authorize_stage_rechecks_current_admission` | `authorize_stage-missing-actor`、`get_stage_operations-disabled-actor` |
| `test_authorize_stage_rejects_target_outside_resident_workspace` | `authorize_stage-workspace`、`get_stage_operations-workspace` |

两次运行共同包含以下承接选择器：

```text
tests/unit/workspace/test_registry.py
tests/unit/workspace/resolution/test_alias_resolver.py::test_reference_results_cover_terminal_states_and_canonical_atom
tests/unit/workspace/resolution/test_alias_resolver.py::test_settled_reference_redirect_authorizes_and_redacts_nested_target
tests/unit/patchouli/services/test_retrieval.py::TestRetrievalFamiliarAgentProfiles::test_unspecified_or_builtin_profile_uses_explicit_fallback
tests/unit/workspace/process/test_cpu_port.py::test_cpu_self_reported_terminal_ends_process_without_finalize
tests/unit/workspace/process/test_cpu_port.py::test_cpu_stream_without_terminal_result_fails_process_with_error_event
tests/unit/workspace/process/test_cpu_port.py::test_cpu_exception_fails_process
tests/unit/workspace/process/test_gateway_chat_flow.py::test_streaming_done_omits_sealing_only_result_fields
```

原选择可从表中这 15 个参数函数的全部 nodeid 重建，保留选择使用上表 ID，两次均加同一承接选择器。
删除落地后仍须重新验证受影响文件与配置的完整快速集，比较行、分支缺失，并审查独立失败语义；覆盖集合相同不能代替行为证据。
HTTP/上传候选、其余矩阵和生成引擎迁合尚未做删除后的覆盖对照，不标记完成。
