---
title: 推理模型重新加载时的事件循环停顿
status: todo
owner: patchouli
scope: event-loop-stall-during-inference-model-load-and-unload
code_paths:
  - src/hivememory/infrastructure/model_service.py
  - src/hivememory/infrastructure/inference_worker.py
  - src/hivememory/infrastructure/embedding/bge_m3.py
  - src/hivememory/infrastructure/rerank/fast_embed_reranker.py
  - src/hivememory/patchouli/runtime/core.py
  - src/hivememory/workspace/process/runner.py
related_docs:
  - docs/patchouli/README.md
  - docs/archive/plans/v0.7.1-model-idle-unload.md
  - docs/archive/todo/inference-off-event-loop.md
last_reviewed: 2026-10-30
---

# 推理模型重新加载时的事件循环停顿

## 问题与证据

模型的加载与卸载都在推理线程中执行，但 onnxruntime 创建推理会话时持有 GIL，卸载时的 `gc.collect()` 同样持有 GIL，这段时间里事件循环几乎得不到运行：同一进程内的其他请求、流式输出、导入与维护任务都停住，结束后再一起继续，请求本身不失败。空闲卸载（[Patchouli](../patchouli/README.md#41-模型驻留与空闲卸载)）让这种停顿从只在启动时出现，变为每个空闲周期之后都可能出现一次。

2026-10-29 空闲卸载计划实测验收（[实施记录](../archive/plans/v0.7.1-model-idle-unload.md#12-实施记录)；16 核开发机，以每 10 ms 一次 `GET /health` 的响应时间估计）：

| 场景 | 其他请求的最长等待 |
|:---|---:|
| 卸载后的 chat：预取让两个模型同时加载 | 3.3～3.7 秒 |
| 两个模型先后加载（MCP 检索触发） | 1.1～2.1 秒 |
| 只重新加载嵌入模型（记忆生成触发） | 0.94 秒 |
| 空闲卸载 | 约 0.45 秒 |
| 对照：模型已加载时的 chat | 9 毫秒 |

两个模型同时加载时，一个加载线程释放 GIL 后多半被另一个加载线程取得，停顿接近整个加载时长。

## 影响范围

- 单人使用、同一时间只做一件事时，只表现为空闲一段时间后的第一次请求变慢；
- 加载期间若有其他活动，例如前端另一个会话正在流式输出，或多个 Claude Code 会话同时导入、检索，用户会看到输出卡住 1～4 秒；
- 连接器的导入请求超时为 60 秒，当前的停顿不会让导入失败。

可能的处理（未决定）：

- 两个模型的加载改为串行：同时加载时的停顿降到单个模型的水平，两个模型都加载完的时间变长；
- 把模型加载与推理放到独立子进程：消除停顿，需要处理进程间传输、子进程生命周期与内存归还，是空闲卸载计划的非目标；
- 评估卸载时完整的 `gc.collect()` 对内存归还是否必要。

## 完成条件

- 重新加载与卸载期间，其他请求的最长等待降到 owner 认可的水平，按验收时的同一方法实测并记录；
- 驻留语义不变：推理进行中不卸载，加载与卸载互斥，卸载后内存归还操作系统；
- [Patchouli](../patchouli/README.md) 第 7 节的已知限制按最终行为更新。

## 追踪

- 2026-10-30：owner 决定记为已知限制，之后单独优化。未排期，未绑定 Issue。
