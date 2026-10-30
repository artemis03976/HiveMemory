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
  - docs/archive/plans/v0.7.1-model-idle-unload.md
last_reviewed: 2026-10-29
---

# 重排模型 int8 与 ONNX 内存池

## 问题

plugin 模式下后端与 Qdrant 需要常驻，后端进程的常驻内存主要来自两个 ONNX 模型，以及 ONNX Runtime 在推理后保留的内存池。16 GB 内存的笔记本或小主机上压力明显。

本项处理三处与模型加载生命周期无关的开销：重排模型的精度、ONNX 的 CPU 内存池、嵌入输入的截断长度。模型的空闲卸载与按需加载已由[归档计划](../archive/plans/v0.7.1-model-idle-unload.md)完成（2026-10-29），当前事实见 [Patchouli](../patchouli/README.md#41-模型驻留与空闲卸载)。

Owner 决定（2026-10-27）：重排模型改用 int8 之前先做实验评估，按结果决定是否采用。

进度（2026-10-27）：不需要实验的四项（ONNX 内存池、嵌入截断、不生效的配置、Docker 预下载）已完成，见文末实施记录；int8 评估尚未开始，样本来源与采用门槛待 owner 确定。

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
- 两个模型的 ONNX 会话关闭 CPU 内存池（已完成）；
- 嵌入输入设截断上限，上限值依据 dense 文本的实际长度与检索质量确定（已完成，1024 token）；
- `use_fp16`、`device`、`batch_size` 三项配置删除或接通（已完成：前两项删除，`batch_size` 接通）。删除的键按从未生效键的先例处理：残留时由所在配置段拒绝未知键而加载失败，[配置](../system/configuration.md)中写明升级注意；只有设置被迁到别处的键才给出专门的迁移提示；
- Docker 预下载读取当前的配置路径，并写入运行时使用的缓存目录（已完成）；
- 归属层测试覆盖配置到会话选项、截断长度与模型文件的映射，不加载真实模型（已完成前四项的部分）；
- 更新 [Patchouli](../patchouli/README.md)、[配置](../system/configuration.md)与[安装指南](../help/setup.md)中受影响的模型、配置与下载说明（前四项已更新）。

## 相关文档

- [模型推理移出事件循环](../archive/todo/inference-off-event-loop.md)：同批登记，已于 2026-10-27 完成；
- [推理模型的空闲卸载与按需加载](../archive/plans/v0.7.1-model-idle-unload.md)：模型加载生命周期，已于 2026-10-29 完成并归档；
- [Patchouli 检索](../patchouli/retrieval.md)：重排阶段与分数语义。

## 实施记录

### 不需要实验的四项（2026-10-27）

**实现**

- 嵌入与重排的 ONNX 会话以 `enable_cpu_mem_arena=False` 创建（fastembed 0.8 暴露的会话选项）。
- BGE-M3 加载后把分词器截断设为 1024 token；取不到可设置截断的分词器时加载失败，不交出未截断的模型；截断设置完成后才对其他线程可见。重排模型沿用分词器自带的 512 token。
- 删除 `EmbeddingConfig.device`、`RerankerConfig.device` 与 `use_fp16`，随仓库发布的 `configs/config.yaml` 同步移除；嵌入服务的 multiton 实例只按模型名区分。`RerankerConfig.batch_size` 增加至少 1 的约束，`compute_score` 去掉未使用的 `batch_size`、`max_length` 参数，按配置的 `batch_size` 分批送入模型。
- 存储层“配置的模型不是 BGE-M3 时改用 Xenova/bge-m3”的规则从 `QdrantMemoryStore` 移入 `get_bge_m3_service`，Docker 预下载与运行时因此加载同一个模型。
- `docker/Dockerfile` 的预下载改为调用 `load_app_config`、`get_bge_m3_service` 与 `get_fast_embed_reranker_service` 后 `warmup()`，模型名、自定义模型注册与缓存目录都与运行时一致；重排关闭时不下载重排模型。

**核对结论**

- 用真实模型实测：一条约 8000 token 的输入被截断为 1024 token，嵌入耗时 0.45 秒，进程峰值约 1.14 GB，结束后回落到 0.98 GB（截断前为 25 秒、峰值 9.26 GB）；重排连续两次推理后常驻内存不再增长。
- 预下载脚本在本地以 `configs/config.yaml` 运行，解析出 `Xenova/bge-m3` 与 `BAAI/bge-reranker-base`，缓存目录均为 `data/model_cache`；本机没有 Docker，镜像构建未实际验证。
- 前端设置页的主配置仍是旧的扁平结构（顶层 `embedding`、`retrieval.retriever.reranker`），根配置忽略未知的顶层键，这些表单项此前就不生效（已知限制见[管理页面](../frontend/management-views.md)第 3.3 节）。前端类型、mock 配置、校验与内部引擎页随之删除后端已不存在的字段：嵌入的 `device`、`batch_size`、`normalize_embeddings`、`dimension`，重排的 `device`、`use_fp16`；结构对齐不在本项范围内。

**验证**

- `tests/unit/infrastructure/test_fastembed_services.py`：加载时关闭内存池（嵌入与重排各一）、分词器按 1024 token 截断、取不到分词器时加载失败且模型不可见、重排按配置的 `batch_size` 送入模型、非 BGE-M3 配置改用 Xenova/bge-m3。去掉内存池选项、去掉截断、先赋值再检查分词器、忽略 `batch_size`、去掉模型名改写五种改坏方式均使对应测试失败。原有加载测试改用 monkeypatch，不再把假模型留在进程级单例上。
- `tests/integration/patchouli/test_qdrant_real_infra_local_mutation.py` 的两个测试原先共用固定集合名，本地 Qdrant 可达时在并行执行下创建冲突，改为每个测试独占一个集合。
- `pytest tests/unit tests/integration -m "not live_llm and not e2e and not slow" -n auto` 连续三次 3053 passed。
- 前端 `npm run lint`、`npm run build` 通过，`npm run test:session` 18 passed。
