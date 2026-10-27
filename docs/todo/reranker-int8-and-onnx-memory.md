---
title: Int8 Reranker And ONNX Memory Arena
status: todo
owner: patchouli
scope: reranker-quantization-evaluation-onnx-cpu-arena-and-embedding-truncation
code_paths:
  - src/hivememory/infrastructure/rerank/fast_embed_reranker.py
  - src/hivememory/infrastructure/rerank/base.py
  - src/hivememory/infrastructure/embedding/bge_m3.py
  - src/hivememory/engines/retrieval/reranker.py
  - src/hivememory/config/patchouli.py
  - src/hivememory/config/shared.py
  - configs/config.yaml
  - docker/Dockerfile
related_docs:
  - docs/patchouli/retrieval.md
  - docs/system/configuration.md
  - docs/help/setup.md
  - docs/archive/todo/inference-off-event-loop.md
  - docs/plans/v0.7.1-model-idle-unload.md
last_reviewed: 2026-10-27
---

# 重排模型 int8 与 ONNX 内存池

## 问题

plugin 模式下后端与 Qdrant 需要常驻，后端进程的常驻内存主要来自两个 ONNX 模型，以及 ONNX Runtime 在推理后保留的内存池。16 GB 内存的笔记本或小主机上压力明显。

本项处理三处与模型加载生命周期无关的开销：重排模型的精度、ONNX 的 CPU 内存池、嵌入输入的截断长度。模型的空闲卸载与按需加载由[计划](../plans/v0.7.1-model-idle-unload.md)承担。

Owner 决定（2026-10-27）：重排模型改用 int8 之前先做实验评估，按结果决定是否采用。

## 证据

2026-10-27 开发机实测，fastembed 0.8.0，onnxruntime 1.30.0。

- 开发机上常驻运行的后端进程峰值常驻内存（VmHWM）3.0 GB；Qdrant 当前活跃内存约 80 MB（`hivememory_main` 35 条记录）。
- 后端内存组成：

  | 组成 | 常驻内存 |
  |:---|:---|
  | Python 与依赖导入（`hivememory.server.app`） | 约 300 MB，其中 litellm 约 200 MB |
  | bge-m3（`Xenova/bge-m3` 的 `onnx/model_int8.onnx`） | 约 850 MB |
  | bge-reranker-base（fp32 `onnx/model.onnx`，文件 1.11 GB） | 约 1.36 GB，加载时瞬时峰值约 1.8 GB |

- **配置项不生效**：`FastEmbedRerankerService` 不读取 `use_fp16` 与 `device`，`CrossEncoderReranker` 调用 `compute_score` 时不传 `batch_size`。仓库配置 `use_fp16: true` 时实际加载的仍是 fp32 模型。
- **int8 候选**：`Xenova/bge-reranker-base` 提供 `onnx/model_int8.onnx`，可以用 `TextCrossEncoder.add_custom_model` 接入，方式与 bge-m3 相同。
  - 加载约 0.9 秒，常驻约 0.56 GB；
  - 重排 20 条 0.74～0.82 秒，fp32 为 1.1～1.2 秒；
  - 同一仓库的 `onnx/model_q4.onnx` 常驻约 1.19 GB、重排 1.05～1.15 秒，不作候选。
  - 一组 8 条中文候选上，int8 与 fp32 的排序完全一致，原始分数最多偏移约 0.9（如 2.25 → 1.38）。重排分数不设阈值，归一化分数只用于排序与展示。这组样本只能说明方向，不能作为采用依据。
- **CPU 内存池**：ONNX Runtime 默认开启 CPU 内存池，推理峰值分配在推理结束后保留在进程内；fastembed 0.8 以 `enable_cpu_mem_arena` 暴露该选项。
  - bge-m3 嵌入一条约 4000 token 的文本后，常驻内存从 0.93 GB 升到 3.26 GB 并保持；关闭内存池后回到 0.93 GB。
  - 重排推理后，开启内存池多保留 0.3～0.6 GB，关闭时约 0.1 GB。
  - 这几次测量中，关闭内存池对推理耗时没有可见影响。
- **截断长度**：bge-m3 分词器的 `model_max_length` 为 8192，fastembed 按此截断。
  - 关闭内存池时，一条 8003 token 的输入瞬时峰值 9.26 GB，耗时约 25 秒。
  - 存储写入的 dense 文本由标题（≤200 字符）、类型、标签与摘要（≤500 字符）组成，有长度上限；查询文本的上限未找到。
- **Docker 预下载落空**（按代码核对，未实际构建镜像验证）：`docker/Dockerfile` 的预下载从配置顶层读取 `embedding.default` 与 `retrieval.retriever.reranker`，而当前配置位于 `shared.embedding.default` 与 `patchouli.retrieval.reranker`。读取落空后回退为默认模型名，`cache_dir` 为空，模型下载到 fastembed 的默认缓存目录，而运行时使用 `data/model_cache`。

## 实验评估

- 样本：取真实记忆库中的查询及其召回候选池（候选数与运行时的重排 `top_k` 一致），覆盖中文、英文与代码片段类记忆；
- 指标：fp32 与 int8 的 top-1 一致率、top-5 重合率与排序相关系数，以及两者的加载耗时、重排耗时与常驻内存；
- 采用门槛在评估前写定，结果与样本说明记入本 Todo，随归档保存。

## 完成条件

- 完成评估并按结果决定是否采用 int8；
- 采用时，重排模型改从 int8 文件加载，配置、Docker 预下载与模型缓存目录同步更新；
- 两个模型的 ONNX 会话关闭 CPU 内存池；
- 嵌入输入设截断上限，上限值依据 dense 文本的实际长度与检索质量确定；
- `use_fp16`、`device`、`batch_size` 三项配置删除或接通。删除时，配置中残留的旧键会被 `extra="forbid"` 拒绝，需要像重排配置迁移那样给出明确提示；
- Docker 预下载读取当前的配置路径，并写入运行时使用的缓存目录；
- 归属层测试覆盖配置到会话选项、截断长度与模型文件的映射，不加载真实模型；
- 更新 [Patchouli 检索](../patchouli/retrieval.md)、[配置](../system/configuration.md)与[安装指南](../help/setup.md)中受影响的模型、配置与下载说明。

## 相关文档

- [模型推理移出事件循环](../archive/todo/inference-off-event-loop.md)：同批登记，已于 2026-10-27 完成；
- [推理模型的空闲卸载与按需加载](../plans/v0.7.1-model-idle-unload.md)：模型加载生命周期；
- [Patchouli 检索](../patchouli/retrieval.md)：重排阶段与分数语义。
