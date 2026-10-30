---
title: Model Inference Off The Event Loop
status: archived
archived_at: 2026-10-27
implemented_by: feat/model-memory-footprint（本归档提交）
superseded_by: docs/patchouli/README.md
owner: patchouli
scope: embedding-and-rerank-inference-and-lazy-load-blocking-the-event-loop
code_paths:
  - src/hivememory/infrastructure/inference_worker.py
  - src/hivememory/infrastructure/storage/vector_store.py
  - src/hivememory/infrastructure/embedding/
  - src/hivememory/infrastructure/rerank/
  - src/hivememory/engines/retrieval/engine.py
  - src/hivememory/engines/retrieval/reranker.py
  - src/hivememory/engines/generation/engine.py
  - src/hivememory/patchouli/runtime/core.py
related_docs:
  - docs/patchouli/README.md
  - docs/patchouli/retrieval.md
  - docs/todo/reranker-int8-and-onnx-memory.md
  - docs/archive/plans/v0.7.1-model-idle-unload.md
last_reviewed: 2026-10-27
---

# 模型推理移出事件循环

> 已完成并归档（2026-10-27）。当前事实见 [Patchouli](../../patchouli/README.md) 第 4 节；实施结果见文末实施记录。

## 问题

嵌入与重排的 ONNX 推理在 async 函数里同步执行，推理期间事件循环停住：同一进程内的 HTTP 请求、SSE 输出、Import Bus 提交、work queue 与维护任务都要等它结束。模型在首次使用时懒加载，加载同样发生在调用推理的那一次同步调用里。

Owner 决定（2026-10-27）：本项先于[重排模型 int8 与 ONNX 内存池](../../todo/reranker-int8-and-onnx-memory.md)和[推理模型的空闲卸载与按需加载](../plans/v0.7.1-model-idle-unload.md)完成。空闲卸载之后模型会在请求路径上重新加载，推理与加载留在事件循环上时，每次重新加载都会让整个服务停顿。

## 证据

- `VectorStore.search_memories` 构造 dense 查询时同步调用 `embedding_service.encode`；`upsert_memory` 经 `_encode_vectors` 同步编码。检索、记忆生成的去重候选检索（`engines/generation/engine.py` 经 `mid_term.search`）与记忆写入都经过这两处。
- `RetrievalEngine.retrieve` 同步调用 `reranker.rerank`，`CrossEncoderReranker.rerank` 再同步调用 `compute_score`。
- `SingletonModelService.model`（嵌入与重排各有一份基类）在首次访问时同步加载模型；只有启动时的 `PatchouliRuntime.warmup_models` 经 `asyncio.to_thread` 加载。
- 2026-10-27 开发机实测（16 核，fastembed 0.8.0，onnxruntime 1.30.0，模型文件已在页缓存中）：

  | 操作 | 耗时 |
  |:---|:---|
  | 嵌入一条短查询 | 约 0.04 秒 |
  | 重排 20 条候选（fp32） | 0.8～1.2 秒 |
  | 嵌入一条约 4000 token 的文本 | 约 6 秒 |
  | 嵌入一条约 8000 token 的文本（bge-m3 截断上限 8192） | 约 25 秒 |
  | 加载 bge-m3 / 重排模型 | 1.0～1.4 秒 / 1.5～1.8 秒 |

  核数更少的机器更慢。
- 未核对：sparse 查询与写入以 `Document(model="qdrant/bm25")` 交给 qdrant-client，BM25 是否在客户端同步计算。

## 影响

- 每次检索至少让事件循环停住一次重排的时长；同时进行的其他会话、流式输出与导入请求在这段时间内都没有进展。
- 长输入会让整个服务停住同样长的时间，存储写入的 dense 文本有长度上限（标题与摘要），查询文本的上限未找到。
- 空闲卸载计划的前提：卸载后的重新加载必须不阻塞事件循环。

## 完成条件

- 嵌入推理、重排推理与模型加载都不在事件循环上执行，调用方仍以 async 等待结果；
- 每个模型的推理并发有明确上限并说明其内存影响：同一模型并行推理会叠加激活内存；
- 取消正确传播：调用方取消时 `CancelledError` 照常向上；已经开始的推理在线程中跑完后结果被丢弃，不再产生后续写入；
- 服务关闭时不留下未结束的推理任务，生成队列 drain 期间的编码仍能完成；
- 归属层测试以会阻塞的假模型验证推理期间事件循环仍能调度其他协程，并覆盖取消与关闭，不以固定 sleep 等待状态；
- 核对 [Patchouli 检索](../../patchouli/retrieval.md)与 [Patchouli](../../patchouli/README.md) 中有关模型调用与预热的描述，与实现不符的随之更新。

## 相关文档

- [重排模型 int8 与 ONNX 内存池](../../todo/reranker-int8-and-onnx-memory.md)：同批登记的推理内存事项；
- [推理模型的空闲卸载与按需加载](../plans/v0.7.1-model-idle-unload.md)：以本项完成为进入条件；
- [Patchouli 检索](../../patchouli/retrieval.md)：召回与重排阶段。

## 实施记录（2026-10-27）

**实现**

- 新增 `InferenceWorker`（`infrastructure/inference_worker.py`）：每个模型服务一个单线程执行器，调用在调用方上下文的副本中执行；排队中的调用被取消后不再执行，已经开始的调用跑完后结果被丢弃；`close()` 取消排队调用、等待在途调用结束，之后的调用重新创建线程（模型服务是进程级单例，会被之后启动的系统继续使用）。
- 嵌入服务的 `encode` 与重排服务的 `compute_score` 改为 async：稠密编码与打分（含首次使用时的懒加载）在推理线程中执行；只请求 sparse 原文时不经过线程。`BaseReranker.rerank` 改为 async，`RetrievalEngine` 与 `QdrantMemoryStore` 随之等待。
- 推理并发上限为每个模型一个线程：同一模型并行推理会叠加激活内存，ONNX Runtime 的单次推理本身已使用多个算子线程，并行带来的吞吐收益有限。嵌入与重排各有线程，互不排队。
- `PatchouliSystem.stop()` 在生成 lane 停止之后调用 `PatchouliRuntime.stop_inference_workers()`，释放两个推理线程；drain 期间的生成仍能完成编码。
- 启动预热仍经 `asyncio.to_thread` 调用同步的 `warmup()`，与推理线程上的懒加载由模型服务的加载锁互斥。

**核对结论**

- sparse 查询与写入的 `Document(model="qdrant/bm25")`：qdrant-client 1.19.1 的 `ModelEmbedder` 原样返回该对象，BM25 由 Qdrant 服务端计算，不在客户端事件循环上执行，无需改动。
- 用真实模型通过服务实测事件循环停顿（每 10 ms 一次 tick 的最大延迟）：稳态嵌入（约 1440 字）与重排 20 条期间最大停顿 4～7 ms，两者并发时 7 ms。改动前推理在协程中同步执行，事件循环在整个推理期间停住，同样输入的推理耗时为 0.5 秒与 1.15 秒。
- **已知限制**：模型加载期间事件循环仍会停顿。实测在另一线程中创建 onnxruntime 会话时，bge-m3 加载 0.84 秒、最大停顿 452 ms，fp32 重排模型加载 1.67 秒、最大停顿 835 ms（关闭图优化时 784 ms）；分词器加载 0.48 秒、停顿 132 ms。原因是 onnxruntime 创建会话、解析模型时持有 GIL，换线程无法消除。当前只在启动预热或首次使用时发生；空闲卸载之后每次重新加载都会出现，已作为风险记入[空闲卸载计划](../plans/v0.7.1-model-idle-unload.md)。

**验证**

- 新增 `tests/unit/infrastructure/test_inference_worker.py`：事件循环在调用阻塞期间继续调度、同一 worker 逐个执行、取消在途调用的传播、取消排队调用不再执行、关闭等待在途调用且之后可再次使用。改为在事件循环上执行、两个线程、关闭不等待、关闭不重建线程、排队调用被 shield 五种改坏方式均使对应测试失败。
- `tests/unit/infrastructure/test_fastembed_services.py` 改为 async，并各补充一例：首次编码与首次重排的加载和推理都不在事件循环线程上执行。
- 其余测试随接口改为 async：重排器与检索引擎测试、存储单元测试的嵌入 mock、6 个集成测试的确定性嵌入 fake、Patchouli 生命周期测试补充推理线程的释放。
- `pytest tests/unit tests/integration -m "not live_llm and not e2e and not slow" -n auto`：3049 passed；`pytest tests/e2e -m "e2e and not live_llm"`：22 passed；`tests/e2e/component/test_retrieval_e2e.py`（真实模型与本地 Qdrant）：12 passed；`test_qdrant_real_infra_local_mutation.py`（本地 Qdrant）：2 passed。live LLM 测试未运行。
