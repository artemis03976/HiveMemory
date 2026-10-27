---
title: Model Inference Off The Event Loop
status: todo
owner: patchouli
scope: embedding-and-rerank-inference-and-lazy-load-blocking-the-event-loop
code_paths:
  - src/hivememory/infrastructure/storage/vector_store.py
  - src/hivememory/infrastructure/embedding/
  - src/hivememory/infrastructure/rerank/
  - src/hivememory/engines/retrieval/engine.py
  - src/hivememory/engines/retrieval/reranker.py
  - src/hivememory/engines/generation/engine.py
  - src/hivememory/patchouli/runtime/core.py
related_docs:
  - docs/patchouli/retrieval.md
  - docs/todo/reranker-int8-and-onnx-memory.md
  - docs/plans/v0.7.1-model-idle-unload.md
last_reviewed: 2026-10-27
---

# 模型推理移出事件循环

## 问题

嵌入与重排的 ONNX 推理在 async 函数里同步执行，推理期间事件循环停住：同一进程内的 HTTP 请求、SSE 输出、Import Bus 提交、work queue 与维护任务都要等它结束。模型在首次使用时懒加载，加载同样发生在调用推理的那一次同步调用里。

Owner 决定（2026-10-27）：本项先于[重排模型 int8 与 ONNX 内存池](./reranker-int8-and-onnx-memory.md)和[推理模型的空闲卸载与按需加载](../plans/v0.7.1-model-idle-unload.md)完成。空闲卸载之后模型会在请求路径上重新加载，推理与加载留在事件循环上时，每次重新加载都会让整个服务停顿。

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
- 核对 [Patchouli 检索](../patchouli/retrieval.md)与 [Patchouli](../patchouli/README.md) 中有关模型调用与预热的描述，与实现不符的随之更新。

## 相关文档

- [重排模型 int8 与 ONNX 内存池](./reranker-int8-and-onnx-memory.md)：同批登记的推理内存事项；
- [推理模型的空闲卸载与按需加载](../plans/v0.7.1-model-idle-unload.md)：以本项完成为进入条件；
- [Patchouli 检索](../patchouli/retrieval.md)：召回与重排阶段。
