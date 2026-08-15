---
title: 测试体系问题清单（Testing Suite Audit）
status: in-progress
owner: all
scope: tests-suite-quality-audit
related_docs:
  - pyproject.toml
  - .github/workflows/ci.yml
last_reviewed: 2026-08-14
---

# 测试体系问题清单

## 概述

对 `tests/` 下全部 204 个测试文件（约 2009 个测试用例）进行逐文件审查，覆盖
`unit` / `integration` / `e2e` 三个层级，并交叉核对了覆盖率基线（CI 门禁 `--cov-fail-under=85`，
实测 88.99%）与低覆盖模块。

**核心结论**：测试数量充足但有效性分布极不均匀。约一半文件是扎实的行为断言，可作项目参照标准；
另一半存在"测试数量多但无法发现回归"的问题——恒真断言、同义反复、只验证 mock 调用细节、
层级错位（mock 冒充集成 / 假 e2e）、被 marker 掩盖的死测试、真实 LLM 条件下的软失败模式等。

## 全局问题类别统计

| 类别 | 计数 | 说明 |
|---|---|---|
| TAUTOLOGY（恒真/同义反复断言） | 26 | `assert x == x`、构造后回读刚赋的值、mock 返回值镜像 |
| WEAK_ASSERT（过于宽松/条件守卫/无断言） | 30 | 只断言 `is not None` / `len>0`，或断言包在 `if ...:` 内条件不满足即通过，甚至完全没有 assert |
| MOCK_VANITY（只验证 mock 调用细节） | 19 | 断言"mock 被调了几次、传了什么参数"，mock 返回值使断言永远成立 |
| LOCAL_LOGIC（断言写死逻辑） | 10 | 测试内重新实现一遍生产公式/序列化/映射再断言 |
| DEAD_CODE（死代码/死 fixture/陈旧测试） | 26 | 未使用的变量、重复赋值、被 skip 的陈旧测试、整套未使用的 fixtures 目录 |
| IMPL_DETAIL（断言内部实现细节） | 17 | 断言私有属性/私有方法/内部字典结构，重构即挂 |
| STRUCTURE（层级错位/误标 marker/sys.path hack） | 13 | integration 用 mock 冒充、e2e 目录下纯单元测试、docstring 复制残留 |
| FLAKY（真实时钟/固定 sleep/全局状态泄漏） | 18 | `time.sleep(x)` 等待、`>=` 时间断言、真实 LLM 非确定性、无 try/finally 的全局 context |
| NAMING（命名/docstring 与断言不符） | 9 | 测试名说 A 断言 B、docstring 与行为相反 |
| COVERAGE_GAP（关键分支完全无测试） | 5 | gateway 命令解析、litellm 服务、server 入口、fusion timeline、vitality event 分量 |

## 一、按目录的详细问题清单

### 1. `tests/unit/agent_runtime/`（29 文件）

#### 1.1 最严重：live 类测试整体失修（3 个文件 + 1 个辅助模块）

`mtp/test_scenario_live.py`、`mtp/syscalls/test_live_runtime.py`、`mtp/syscalls/test_live_file_io.py`、
`mtp/syscalls/live_support.py`

- 【STRUCTURE/死测试】全部使用已废弃的 `KoakumaRuntime(retrieval_familiar=..., librarian_core=..., storage=...)`
  构造签名；当前生产构造函数已改为 bus 架构 `(bus, config, *, alias_resolver)`。这些测试因 `live_llm`
  marker 被默认排除，从未运行，**任何一个都无法运行成功**（必抛 `TypeError`）。
  - `test_scenario_live.py` L147-152 / L217-222 / L282-287 / L335-340 / L387-392 共 5 处
  - `test_live_runtime.py` L35-37 / L358-363
  - `test_live_file_io.py` L46-53
  - `live_support.py` L169-175
- 【WEAK_ASSERT/无断言】`test_scenario_live.py` 的 `test_read_after_search_uses_alias`（L250-263）、
  `test_greeting_no_mtp`（L462-477）整个函数没有任何 assert。
- 【WEAK_ASSERT】条件守卫模式：`if mock_retrieval.retrieve.called: assert ...`，LLM 未触发该路径时测试
  静默通过（`test_scenario_live.py` L176-194、L444-460；`test_live_runtime.py` 7 处；
  `test_live_file_io.py` 3 处）。
- 【DEAD_CODE】`test_live_runtime.py` L327-384：文件在 "Test 8" 注释处结束，其后 5 个 fixture 无任何测试
  引用，且使用未导入的 `tempfile`/`shutil`。
- 【DEAD_CODE】`test_live_runtime.py` L100 使用 `MTPParser()` 但从未 import（一旦以 `-m live_llm` 运行抛 `NameError`）。
- 【WEAK_ASSERT】`test_live_runtime.py` L138-140 `assert "UTC" in backfill or "20" in backfill`——
  `"20"` 子串几乎匹配任意文本。
- 【WEAK_ASSERT】`test_clock_result_backfilled` 等对时钟输出的断言过宽。

**建议**：三个 live 文件要么全部改用 `make_koakuma_runtime(bus, config)` 工厂并去掉条件断言，
要么从 unit 目录移除并归入专门的 live/e2e 目录；删除 `test_live_runtime.py` 后半段死代码并补 `MTPParser`
导入。

#### 1.2 MTP 模型层同义反复（恒真）

- `mtp/test_write_chain.py` L123-144、L152-173：构造 `WriteFocus`/`GenerationRequest` 后断言字段等于刚赋的值
  （Pydantic dataclass 赋值回读，恒真）。
- `mtp/test_update_chain.py` L126-165、L182-197、L205-230、L167-174：同样的构造-回读模式。
- `test_execution_frame.py` L15：断言 `frame.harvested_aliases == []`（默认值回读）。

**建议**：删除模型构造类测试，仅保留有派生逻辑的（如 `is_write`/`is_update` 互斥）。

#### 1.3 断言私有实现/内部格式

- `mtp/test_call_handler.py` L40-47 / L68-72 / L86-90：直接调用私有方法 `koakuma._handle_call(...)`
  绕过公共入口。
- `test_runtime.py` L55-57：断言两层私有属性 `runtime._loop_executor._mtp_executor is mtp_executor`。
- `execution/test_loop_turn_events.py` L302：写死 `action_id == "action_1_0"` 内部编号格式。
- `test_execution_frame.py` L17-20 / L34：`assert not hasattr(frame.runtime_scope, "depth")`——断言"未来不应有某属性"，
  实现新增任一无害属性即挂。

#### 1.4 mock 调用细节

- `mtp/test_executor.py` 全文：薄委托包装测试，只断言 `intercept_and_execute.assert_awaited_once_with(...)`。
- `mtp/test_update_chain.py` L268-273：断言 merge 的中间 metadata 参数细节。

#### 1.5 其他

- 【WEAK_ASSERT】`mtp/syscalls/test_registry.py` L31-33：测试名"custom repl timeout"但只断言
  `"sys_python_repl" in registry`，参数完全未验证。
- 【WEAK_ASSERT】`mtp/test_run_chain.py` L127-132：`len(...) == 10 or "20" in ...` 双分支让精确断言失去约束力。
- 【WEAK_ASSERT】`mtp/test_write_chain.py` L142-144、`test_update_chain.py` L159-165 / L193-197：
  `pytest.raises(Exception)` 捕获一切异常，无法区分错误类型。
- 【WEAK_ASSERT/FLAKY】`test_update_chain.py` L435-442：`before = datetime.now(); assert updated_at >= before`
  真实时钟 + `>=` 边界。
- 【FLAKY】`mtp/syscalls/test_mtp_integration.py` L106-112：在项目工作目录创建 `.test_tmp` 真实目录而非
  `tmp_path`。
- 【DEAD_CODE/维护风险】`mtp/conftest.py` 与 `mtp/syscalls/conftest.py` 90% 重复，且 syscalls 版本缺少
  citation 路由注册，两套 bus 行为不一致。
- 【COVERAGE_GAP】`test_runtime.py` 缺少 `run_frame` 成功路径测试（只有异常映射与 finalize 系列）。

### 2. `tests/unit/engines/`（31 文件）

#### 2.1 恒真/无操作断言

- `retrieval/test_engine.py` L92：`assert result.latency_ms >= 0` 恒真（真实时钟非负）。
- `retrieval/test_fusion.py` L259：`assert ...confidence_score > 0.9` 断言的是构造输入对象的固有属性。
- `retrieval/test_retriever.py` L145：`assert results.results[0].score > 0` 恒真。
- `lifecycle/test_engine.py` L252：`assert results == mock_history`——mock 返回值镜像。

#### 2.2 测试测试自身的辅助函数（同义反复最严重）

- `retrieval/test_agent_menu_rendering.py` L61-73、L140-147：测试 `self._separate_agent_profiles`
  测试类自定义辅助方法，不调用任何生产代码，断言永远成立。生产分离逻辑在 `compiler.py` L209-220。

#### 2.3 断言写死逻辑（LOCAL_LOGIC）

- `lifecycle/test_vitality.py` L80-102（`test_access_boost`）：测试内复刻 `coef * log(1+n)` 公式再断言一致，
  公式改错时测试副本必须同步改，等于无保护；L104-126 完整复制生产三段公式，且 L126 断言的是**测试自己算出的
  中间变量**（恒真、无意义）。
- `artifacts/test_artifact_store.py` L82-95：测试内复刻 `json.dumps(..., separators=(",", ":"))` + sha256。

#### 2.4 时间衰减测试无区分力（重点）

- `retrieval/test_retriever.py` L88-115：注释里推导了期望分数（`0.85*0.9=0.765` 等），但唯一断言是
  `results.results[0].memory.index.title == "M1"`。已核实生产代码**不排序、保持存储返回顺序**，mock 返回顺序
  本就是 `[M1, M2]`——即使衰减逻辑整体被删除测试依然通过。

#### 2.5 mock 细节/内部实现

- `retrieval/test_retriever.py` L66-86：断言传给 mock 的 qdrant `Filter` 对象内部结构
  （`filter.must[].key`），qdrant 内部结构一变即挂。
- `retrieval/test_reranker.py` L147-155：直接调用私有方法 `_normalize_score`。
- `generation/test_deduplicator.py` L65-89：直接测私有方法 `_calculate_text_similarity` 并手算 Jaccard。
- `lifecycle/test_vitality.py` L31-38：断言生产常量字典的具体值。

#### 2.6 过于宽松断言

- `memory_compiler/test_compiler.py` L274-277：`assert "截断" in artifact.text or len(artifact.text) < 500`——
  OR 兜底恒真（payload 仅 60 字符，截断逻辑失效也能通过）；同文件英文版 L279-288 才是正确写法。
- `retrieval/test_fusion.py` L389-402：只断言 `len == 1`，未验证"回退到 concept 权重"（同文件 L322-337 有精确值先例）。
- `retrieval/test_fusion.py` L347-361：只断言 `score < 0.9`。

#### 2.7 其他

- 【DEAD_CODE】`generation/test_engine.py` 多处重复 `self.mock_storage.upsert = AsyncMock()`（setup 已赋值）。
- 【NAMING】`generation/test_engine.py` L474-485：docstring 写"DISCARD 决策返回空"，断言却是 `len(result) == 1`
  （与生产一致：返回 atom 为 None 的占位结果），描述误导。
- 【COVERAGE_GAP】`retrieval/test_fusion.py` timeline 模式权重无直接测试；`lifecycle/test_vitality.py`
  event_vitality_boost（B 项）分量无直接断言。
- 【WEAK_ASSERT】`generation/test_generation_transcript_builder.py` L342：`assert "摘要" in transcript or ...`——
  `state_summary` 恒非空时后一条件恒真。

### 3. `tests/unit/patchouli/`（29 文件）

#### 3.1 薄 service 封装测试（MOCK_VANITY 集中地，7 个文件）

`application/` 下 `test_model_readiness_service.py`（全文）、`test_memory_task_management_service.py`（全文）、
`test_memory_management_service.py`（L40/129/151 三处）、`test_topic_management_service.py`（转发类部分）、
`test_agent_profile_management_service.py`（L71）、`services/test_lifecycle.py`（大部分）、
`test_patchouli_system.py`（全文）：

- 共同模式：`bus.request = AsyncMock(return_value=X)` → `assert result is X`（mock 返回值恒真）；
  或只断言 `assert_awaited_once_with(route, ...)` 验证"参数透传给 bus"，无任何可观察行为验证。
- **影响**：即使 service 业务处理被清空，测试依然全绿；无法发现业务回归。
- **例外（质量好，需保留）**：`test_memory_management_service.py` L44-100（list/vitality/exclude_types 真实行为）、
  `test_agent_profile_management_service.py` L40-48（profile type 强制）、`test_lifecycle.py` L122-133
  （UUID 归一化）、`test_topic_management_service.py` L52-96（owner 校验）。

**建议**：这类转发测试收敛为参数化数据驱动测试，或并入真实 bus 的集成测试。

#### 3.2 恒真断言

- `control/test_memory_generation_queue.py` L123-127：`assert payload_bytes == codecs.encode(kind, version, work)`
  —— 对同一输入重复编码并断言相等，确定性函数必然相等。
- `control/test_memory_generation_models.py` L92-97：构造 spec 后回读字段。
- `memory_library/test_buffer.py` L46-56：构造后逐字段回读（L26-44 默认值断言同理可合并）。

#### 3.3 死代码 + 重复

- `test_active_interaction_submission.py` L458-487：注册的 `PENDING_ATOM_FAILED` 订阅者永远不会被调用
  （生产 `service.py` L358-369 失败路径只打 warning 不发布该事件），且与 L424-455 相邻测试完全重复。

#### 3.4 断言与语义无关（重点）

- `services/test_retrieval.py` L316-323 `test_retrieve_min_confidence_zero_ignored`：测试名声称"min_confidence=0
  被忽略"，但断言是 `query.filters.min_confidence == 0`——生产默认值就是 0.0，无论生产是否正确"忽略"断言都通过。

#### 3.5 内部实现细节

- `runtime/test_bus.py` L37 / L109 / L127：直接断言 `bus._handlers` / `bus._subscribers` 私有字典结构。

#### 3.6 FLAKY

- `services/test_perception.py` L373 / L390 / L406：3 处真实 `time.sleep(1.1)` 等待 idle timeout。
- `test_active_interaction_submission.py` L267-270：固定 20 次轮询等待队列接纳（轻微）。

#### 3.7 其他

- 【WEAK_ASSERT】`memory_library/test_buffer.py` L21-24：测试名"is_uuid"但只断言 `is not None` + `len > 0`。
- 【WEAK_ASSERT】`memory_library/test_memory_library.py` L125-131：`last_accessed_at >= initial` 应改为 `>`。
- 【LOCAL_LOGIC】`memory_library/test_buffer.py` L186-189、`control/test_memory_generation_models.py` L23-27：
  复制枚举定义常量断言。
- 【COVERAGE_GAP】finalize 成功路径返回的 materialize 任务列表从未被断言（轻微）。

### 4. `tests/unit/system/` + `tests/unit/server/`（45 文件）

#### 4.1 恒真断言

- `test_chat_run_control_contract.py` L135：`assert not isinstance(task.exception() if task.done() and not task.cancelled() else None, _ChatRunCancelled)`——
  条件表达式恒返回 `None`，断言恒真；真实意图（owner 取消不翻译成 `_ChatRunCancelled`）被上方
  `run.outcome is ChatRunOutcome.RUNNING` 覆盖。
- `server/models/test_models.py` 整文件：构造 Pydantic 对象后回读字段（`req.user_id == "u1"`）恒真；
  保留 `test_defaults`、`test_from_atom`、HealthResponse 版本断言即可。
- `contracts/test_public_routes.py` L246-262：16 条路由字符串字面量自断言（兼 LOCAL_LOGIC）。

#### 4.2 docstring 复制残留（5 个文件）

- `system/application/` 下 `test_memory_service.py`、`test_readiness_service.py`、`test_agent_service.py`、
  `test_topic_service.py`、`test_api_services.py` 第 1 行 docstring 全部为
  "ChatApplicationService / PassiveIngressService 委托测试"，与实际测试对象完全不符。

#### 4.3 死代码（application 目录复制残留）

- 上述 5 个文件中的 `_make_analysis_result`、`_make_memory_atom` helper 从未被调用；
  `test_readiness_service.py` 有约 4 倍于有效代码的死代码（含未用 fixture `passive_config` 与 14 个未用 import）。
- `test_lifecycle.py` L231-232：`patchouli.runtime = runtime` 重复赋值。
- `test_cancel_hardening.py` L154-156：循环体内 import。

#### 4.4 FLAKY（固定 sleep 时序）

- `runtime/scheduler/test_async_scheduler.py` 多处（L79-159、L222-274）：`await asyncio.sleep(0.2/0.25/0.4)`
  后断言 `call_count >= 1`，CI 负载高时可能不足一次；`test_non_reentrant_skip` 断言 `<= 2` 慢机器上可能更多。
- `runtime/work_queue/test_runtime.py` L578-596：`sleep(0.02)` 后断言终态。

#### 4.5 内部实现细节

- `test_hivememory_system.py` L286-287：`assert system.config is system._config`。
- `services/passive/test_passive_ordering_and_submission.py` L125 / L282：断言/访问多层私有
  （`ingressor._serial_gate.active_key_count`、`_memory_context._bus`）。
- `runtime/test_runtime_events.py` L39 / L69 / L73：直接操作订阅者私有属性（`sub._initial_events` 等）。
- `services/passive/test_passive_submission_queue.py` L266-291：`__new__` 绕过构造 + 注入 mock 私有字段。

#### 4.6 其他

- 【WEAK_ASSERT】`routers/test_chat.py` L148-149（token 事件只断言 `>= 1`，不验证内容与顺序）、
  L184-185（docstring 声称验证完整事件序列，实际只查两个事件名存在）；
  `routers/test_ingest.py` L66（幂等键只验证 truthy）；`contracts/test_contracts.py` L56-59
  （`pytest.raises(Exception)` 过宽）。
- 【COVERAGE_GAP】`server/__main__.py` 覆盖率 0%，无任何测试；`gateway/commands/parser.py` 覆盖率 31%，
  命令解析（`--key=value`、required 缺失、类型校验、未知命令）完全无单测。

### 5. `tests/unit/core/` + `prompts/` + `i18n/` + `utils/` + `infrastructure/` + `gateway/`（40 文件）

#### 5.1 命名/docstring 与行为相反（重点，会误导对协议的理解）

- `core/mtp/test_trace_reducer.py`：docstring L8 宣称"WRITE / UPDATE / CALL → 过滤"，但测试
  `test_filtered_tool_kinds`（L118-123）、`test_dict_write_filtered`（L150-154）实际断言它们**被保留**
  （与生产一致）。类名、测试名、docstring 三者全部说反。
- `core/mtp/test_call_parser.py`、`test_filter_type_map.py`、`test_call_response_formatting.py`：
  三个文件头部 docstring 一字不差（宣称覆盖 7 项内容），实际各自只测其中一项，其余全是复制残留；
  顶部各有 15 个未使用 import（DEAD_CODE）。

#### 5.2 恒真/宽松断言

- `core/test_agent_profile_fallback.py` L15-16：`OMNI_DOLL_PROFILE.allowed_mtp_verbs == OMNI_DOLL_ALLOWED_MTP_VERBS`
  —— 生产就是用该常量直接构造，恒真；L13-14（与全集比较）才是有效约束。
- `i18n/test_types.py` L51-54：`normalize_language(input_val) is not None`——"大小写不敏感"核心行为
  （归一化结果正确性）完全未验证。

#### 5.3 内部实现细节

- `core/mtp/test_filter_type_map.py` L54 / L60：直接断言私有模块变量 `_FILTER_TYPE_MAP`。

#### 5.4 COVERAGE_GAP（与低覆盖率直接对应）

- `gateway/commands/parser.py`（31%）+ `handlers.py`（42%）：`tokenize_command`、`parse_command_args`、
  `validate_command_args`、`_matches_type`、`handle_help`/`handle_commands`/`handle_status` 零直接测试；
  `dispatcher.py` 的 rejected/未知命令分支也未测。
- `infrastructure/llm/litellm_service.py`（34%）：9 个可测入口只覆盖 `acomplete_json` 的 2 个场景；
  重试耗尽、`complete_with_tools`/`acomplete_with_tools`、`get_gateway_llm_service(None)` 抛 ValueError 等全未测。

#### 5.5 重复测试

- `core/mtp/test_call_response_formatting.py` L52-105 与 `core/mtp/test_parser.py` L373-429 场景几乎完全重复。

#### 5.6 FLAKY

- `infrastructure/test_trace_context.py` L20-34：修改进程级 contextvars 后无 `try/finally`，断言失败会污染后续测试。
- `gateway/test_phase3c_workflow.py` L300-324、`test_phase3f_request_control.py` L69-102：
  真实 sleep（50ms）配合 1ms/5ms 超时，重负载 CI 偶发失败（轻微）。

#### 5.7 冗余

- `prompts/test_system_prompt.py` L13 / L29 / L44：`assert prompt is not None` 冗余（后续 `in` 断言已隐含）。

### 6. `tests/integration/` + `tests/e2e/`（16 文件）

#### 6.1 integration 层级错位（最严重的结构性问题）

- **全部 5 个 integration 测试文件均使用 `unittest.mock.Mock()` 或内存 fake，没有一个连接真实基础设施**，
  实际是"组件协作测试"；文档自称"不测试：与外部存储（Qdrant）的交互"。目录名造成能力承诺失真。
- `tests/integration/fixtures/` 整目录（`mock_storage.py`、`mock_llm.py`、`mock_embedding.py`）为死代码——
  全局检索证实无任何测试导入使用，实际用的是 `Mock()`。
- 【STRUCTURE/sys.path hack】`test_retrieval.py` L16-18：`project_root = Path(__file__).parent.parent` 指向
  `tests/`，插入的是不存在的 `tests/src`；`test_generation.py` L17-19 同理（路径正确但与根 conftest 重复）。

#### 6.2 integration 内的恒真/写死逻辑测试

- `integration/test_retrieval.py` L347-362 `TestScoreNormalization`：测试内自己实现归一化
  （`normalized_sparse = min(sparse_score / 10.0, 1.0)`）再断言落 `[0,1]`——断言由构造恒真，整个类未调用任何生产代码。
- `integration/test_retrieval.py` L235-254 `test_multiple_filters`：构造 `RetrievalQuery` 后回读字段恒真；
  死代码 `mock_storage.get_memories_by_filter` 从未被调用。
- `integration/test_retrieval.py` L76-114：`test_fusion_strategy_combines_results` 唯一断言 `len(fused.results) > 0`；
  L195-233 `test_query_with_filters` 注释写"Verify filters passed"却无对应断言，过滤条件是否传递完全未验证。
- `integration/test_perception.py` L98-102：`assert perception is not None` 恒真；
  L157-238 两个测试被 `@pytest.mark.skip`（"Callback mechanism removed in new architecture"），陈旧死代码。
- `integration/test_debug_messages.py` 全文：无任何实质断言（`assert result is not None`），本质是手动调试脚本。
- `integration/test_generation.py` 整个 `TestMemoryGenerationEngineLogic` 类是单元测试放集成目录（自述 "Unit Level"）。

#### 6.3 e2e 目录两极分化

**真 e2e**（真实 LLM/Qdrant）：`system/test_flush_triggers_e2e.py`（质量最高）、
`pipeline/test_kernel_loop_e2e.py`、`pipeline/test_active_mode_e2e.py`、`pipeline/test_passive_mode_e2e.py`、
`pipeline/test_sub_agent_call_e2e.py`、`component/test_generation_e2e.py`、`component/test_retrieval_e2e.py`、
`component/test_perception_e2e.py`。

**假 e2e**（纯 mock 单元测试挂 e2e 标记）：
- `component/test_lifecycle_e2e.py`：自述"使用 Mock Storage 模拟 Qdrant、使用 Mock Clock"，未用 `e2e_system`。
- `component/test_koakuma_e2e.py`：全部依赖 MagicMock，残留 `# PLACEHOLDER_KOAKUMA_TESTS` 占位注释。
- `pipeline/test_agent_permission_e2e.py`：无任何 pytestmark，全 mock（patch 掉 PatchouliRuntime 三个私有方法），
  实际在 CI 里作为普通单元测试运行，且与 `unit/agent_runtime/mtp/test_permissions.py` 高度重叠。

#### 6.4 e2e 真测试中的"软失败"模式（重点）

真实 LLM 场景下大量测试用 `if ...: assert ... else: logger.warning(...)`——LLM 行为偏离预期时打 warning 照常通过，
**测试不可能失败**：

- `pipeline/test_active_mode_e2e.py` L366-400：测试名声称"WRITE 被执行、Qdrant 中存在记忆"，整个测试没有任何 assert；
  L408-481 UPDATE 测试唯一断言包在 `if "UPDATE" in commands:` 内，"9090" 内容检查只打日志；
  L274-359 多话题路由 `topic_count >= 2` 只打日志。
- `pipeline/test_passive_mode_e2e.py` L452-513 / L516-584：worth_saving 断言包在 `if worth_saving is False:` 内；
  L589-693 测试名"Session 隔离"但泄漏只打 warning，else 分支反而把泄漏当合理行为接受。
- `component/test_perception_e2e.py` L465-491：只断言 `isinstance(overflow_flushes, list)` 恒真；
  L493-520 整个测试没有 assert。
- `component/test_retrieval_e2e.py` L477-516 / L804-859：计算了 `success` 但从不 assert，注释明说"允许软失败"；
  L739-763 `success = True  # 只要不抛异常就算成功`。
- `component/test_generation_e2e.py` L474-508：content/tags/memory_type 检查失败只打 warning 不参与断言；
  L906-910 提取为空时 `pytest.skip`，schema 合规从未被验证。
- `component/test_koakuma_e2e.py` L378-382：`assert execution_time_ms >= 0` 恒真。
- `component/test_lifecycle_e2e.py` L892-911：第二次归档异常被 try/except 吞掉后 `no_error` 只用于打印不参与断言。

#### 6.5 e2e 内的写死逻辑（LOCAL_LOGIC）

- `component/test_lifecycle_e2e.py` L392-410：测试内重写生产衰减公式
  （`lambda_eff = 0.01 * (2.0 - fact_intrinsic)`）再断言比值接近它。
- `component/test_generation_e2e.py` L1016-1039：测试内重新实现生产 `_draft_to_memory` 转换逻辑。

#### 6.6 基础设施问题

- 【E2E/CI】`e2e/conftest.py` e2e_system 构建真实系统，但 **CI 不运行 e2e**；`pyproject.toml` addopts 同样排除，
  本地 `pytest tests/e2e`（不带 -m）除 agent_permission 外全部静默跳过——"跑过但全被跳过"极易误认为通过。
- 【FLAKY】固定 sleep 密集：`pipeline/test_active_mode_e2e.py`、`test_passive_mode_e2e.py`、
  `system/test_flush_triggers_e2e.py` 大量 `time.sleep(FLUSH_SETTLE_SECONDS)` / `sleep(3)`；
  conftest 已有轮询工具（`wait_for_memory_persistence`）却未统一使用。
- 【副作用】`pipeline/test_sub_agent_call_e2e.py` L58-105：向真实共享存储 upsert `coder_doll` profile 且不清理，
  且会删除他人同 alias 记忆。
- 【WEAK_ASSERT】`system/test_flush_triggers_e2e.py` L512-515：`assert topic_b in active_after or topic_c in active_after`
  OR 断言，未验证"B 仍在、仅 A 被驱逐"的精确语义。

## 二、行动计划（按优先级）

### P0 - 立即可做（低风险、高收益，删/修不可能失败的测试）

1. 修复恒真断言与无断言测试：`test_score_normalization`、`test_multiple_filters`（integration/test_retrieval.py）、
   `test_control_flow_refactor_baseline.py` L84 `profile_resolver is profile_resolver`、
   `test_chat_run_control_contract.py` L135、`test_agent_profile_fallback.py` L15-16、
   `test_memory_generation_queue.py` L123-127、`execution_frame` `not hasattr` 系列。
2. 修复"条件守卫"软断言（e2e 与 live 测试）：把 `if ...: assert ... else: warning` 改为无条件断言。
3. 删除/重写 integration 目录的 mock 冒充：删除 `fixtures/` 死目录、`test_debug_messages.py`、
   两个被 skip 的陈旧测试、`test_perception.py` 中恒真 `is not None` 断言。
4. 删除测试内死代码：各文件的未用变量、重复赋值、未用 import、`test_live_runtime.py` L327-384。

### P1 - 短期（结构调整与归类）

5. 重新归类：
   - `tests/integration/` → 降级为组件测试（真实 `MemoryLibrary` + 真实文件存储，参照 test_lifecycle.py 做法），
     或明确并入 unit；真正需要"集成"语义的测试改用真实 Qdrant。
   - `e2e/component/` 下 `test_lifecycle_e2e.py`、`test_koakuma_e2e.py`、`test_agent_permission_e2e.py`
     → 去除 e2e 标记移入 `tests/unit/`。
   - `unit/agent_runtime/` 下 3 个 live 测试文件 → 修复构造签名或移入 e2e 目录。
6. 修复 `application/` 5 个文件的 docstring 与死代码；修复 `core/mtp/` 三个文件的 docstring 与未用 import；
   修正 `test_trace_reducer.py` 命名（"WRITE/UPDATE/CALL → 保留"）。

### P2 - 中期（补真实缺口的测试）

7. 补 COVERAGE_GAP：`gateway/commands/parser.py` + `handlers.py`、`infrastructure/llm/litellm_service.py`、
   `server/__main__.py` 的入口逻辑。
8. 修复 e2e：让 CI 增加可手动触发的 e2e job（docker compose + 真实依赖服务）；
   把固定 sleep 统一替换为 conftest 的轮询工具；消除 `sub_agent_call_e2e.py` 的共享存储副作用；
   将 `test_flush_triggers_e2e.py` 的 OR 断言收紧。
9. 收敛薄 service 转发测试（patchouli application/）为参数化契约测试或并入集成。

### P3 - 长期（机制建设）

10. 为 FLAKY 类测试注入可控时钟（perception idle、vitality、scheduler），消除真实 sleep。
11. 引入 mutation testing 校验断言有效性；把"断言质量"纳入 code review 标准。
12. 建立"新建测试必须通过行为断言而非 mock 细节断言"的测试规范，并写入手册。

## 三、可作参照标准的高质量测试文件

以下文件断言具体、行为导向、错误分支覆盖完整，可作为本项目测试的样板：

- `tests/unit/agent_runtime/pending_atom/test_runtime.py`（状态机迁移/快照不变量/eviction）
- `tests/unit/agent_runtime/mtp/syscalls/test_file_io.py`、`test_repl.py`、`test_web_search.py`、`test_prompt_teaching.py`
- `tests/unit/engines/lifecycle/test_garbage_collector.py`、`test_archiver.py`、`test_reinforcement.py`
- `tests/unit/engines/perception/test_trigger_manager.py`、`test_page_folding.py`
- `tests/unit/engines/gateway/test_interceptors.py`、`test_query_understanding.py`、`test_topic_router.py`
- `tests/unit/patchouli/control/test_interaction_submission.py`、`test_pending_atom_settler.py`
- `tests/unit/patchouli/control/test_memory_generation_coordinator.py`
- `tests/unit/system/runtime/work_queue/test_runtime.py`（除 L578-596 一处时序）
- `tests/unit/system/services/passive/` 系列契约测试
- `tests/unit/core/mtp/test_parser.py`、`tests/unit/core/mtp/test_formatter_xml_escaping.py`
- `tests/unit/utils/test_json_parser.py`、`test_token_estimator.py`、`test_time_formatter.py`（硬编码期望值，无 LOCAL_LOGIC）
- `tests/unit/infrastructure/test_rate_limiter.py`（monkeypatch 固定时钟的典范）
- `tests/e2e/system/test_flush_triggers_e2e.py`（真实端到端断言的典范）

## 四、完成条件

- P0 全部完成：测试套件中不再存在恒真/无断言/恒绿测试；
- P1 完成：integration 与 e2e 目录的层级语义真实，无 mock 冒充；
- P2 完成：CI 运行所有本应运行的层级，低覆盖关键模块补测到位；
- P3 部分完成：FLAKY 类测试可控化，测试规范文档落地。
