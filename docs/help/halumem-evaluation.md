---
title: 在 HaluMem 上评测记忆生成
status: current
owner: project
scope: trial-instance-and-halumem-evaluation-scripts
code_paths:
  - scripts/trial_instance.py
  - scripts/halumem_eval.py
  - scripts/memory_replay.py
related_contracts:
  - docs/system/import-bus.md
last_reviewed: 2026-11-20
---

# 在 HaluMem 上评测记忆生成

本文说明怎样在开发 checkout 中建立一个隔离的试验实例，把 [HaluMem](https://huggingface.co/datasets/IAAR-Shanghai/HaluMem) 的对话导入它，再按 HaluMem 官方协议对写出的记忆判分。两个脚本都在 `scripts/` 下，只供开发者评测，不属于产品：

- [`trial_instance.py`](../../scripts/trial_instance.py)：建立、运行与删除试验实例；
- [`halumem_eval.py`](../../scripts/halumem_eval.py)：取出官方评判提示词，驱动试验实例逐次会话写入记忆，判分并写出报告。

评测只覆盖记忆的抽取与更新：完整度、准确度、不记干扰，以及更新的正确、幻觉与遗漏。问答与依赖日期的题不在其中，对话按轮导入，不带原始时间。选这个评测集的理由与评测结果见[评测集 Idea](../ideas/memory-evaluation-benchmarks.md)第 6 节。

## 1. 前置条件

- 开发 checkout 的 Python 环境已经装好（[安装与启动](./setup.md)第 4.2 节），开发时用过一次后端，嵌入与重排模型已经下载到 `data/model_cache`。试验实例直接用这个目录，不重复下载。
- `.hivememory/dev.env` 中的 `QDRANT_BIN` 指向本机的 Qdrant 可执行文件（[安装与启动](./setup.md)第 4.5、4.6 节）。试验实例用同一个版本。
- 生成模型与评判模型都登记在开发 checkout 的 `configs/models.yaml`，凭证在 `configs/providers.secrets.yaml` 或 `.env` 的 `HIVEMEMORY__PROVIDERS__*` 中（[配置指南](./configuration.md)）。评判模型必须与生成模型不同。

## 2. 数据与许可

HaluMem 的数据与官方评测代码都是 CC BY-NC-ND 4.0 许可：只做内部评测，不改写后分发，**不提交进仓库**。下面的路径都在被 git 忽略的 `.hivememory/` 下。

1. 从 Hugging Face 数据集 `IAAR-Shanghai/HaluMem` 下载 `HaluMem-Medium.jsonl`，放到 `.hivememory/lab/datasets/halumem/`。
2. 从官方仓库（GitHub `MemTensor/HaluMem`）取得 `eval/eval_tools.py`，用 `prompts` 子命令取出三份评判提示词。脚本只解析源码中的字符串常量，不执行它：

   ```powershell
   python scripts/halumem_eval.py prompts path\to\eval_tools.py .hivememory/lab/halumem/prompts
   ```

数据集中的用户按行序编号，试验实例中的用户 id 是 `halumem-<两位编号>`，例如第 15 行是 `halumem-15`。

## 3. 试验实例

一个试验实例是一次运行目录下的数据根 `<运行目录>/instance/`：

- 有自己的配置覆盖、访问登记、Qdrant 存储、日志与端口，每次运行从空库开始；
- 每个登记的用户有自己的默认 Workspace，记忆互不可见；
- 对话由一个导入来源经 `POST /api/v1/import/turns` 写入，管理请求用后端首次启动时写到数据根的 `configs/management.token`；
- 生成模型、结算方式（`one_shot` 或 `librarian`）与提示词语言写在数据根的 `configs/config.local.yaml`，Gateway 与其余配置沿用主配置。

隔离规则由脚本强制执行，违反时直接失败：

- 端口缺省为 18769、16333、16334（后端、Qdrant HTTP、Qdrant gRPC），用 `--ports` 换另一组。端口已被占用，或者是日常实例的 8769、6333、6334，都会拒绝启动；同时运行两个实例时给它们不同的端口组。
- 数据根的路径不能超过 100 个字符（Qdrant 的路径限制），运行目录不要放得太深。
- 运行目录已经存在时，`create` 拒绝覆盖。

`halumem_eval.py run` 会自己建立与启停实例，通常不需要单独操作。需要手动查看时：

```powershell
python scripts/trial_instance.py create .hivememory/lab/runs/manual --users halumem-15
python scripts/trial_instance.py serve .hivememory/lab/runs/manual
python scripts/trial_instance.py status .hivememory/lab/runs/manual
python scripts/trial_instance.py stop .hivememory/lab/runs/manual
python scripts/trial_instance.py delete .hivememory/lab/runs/manual
```

`serve` 在前台运行，直到 Ctrl+C 或另一个终端里执行 `stop`；退出时先请后端正常关闭，超时才强制结束，再停 Qdrant。后端启动要加载嵌入与重排模型，就绪最多等 10 分钟。`delete` 删除整个运行目录，实例还在运行时拒绝。

## 4. 运行评测

```powershell
python scripts/halumem_eval.py run .hivememory/lab/datasets/halumem/HaluMem-Medium.jsonl .hivememory/lab/runs/eval-one-shot --users 15,0,19,3,6 --settle-mode one_shot
```

- `--users`：逗号分隔的用户行序号；`--sessions`：每个用户最多处理的会话数，试跑时用（例如 `--sessions 10`）；`--concurrency`：同时处理的用户数，缺省 5。
- `--settle-mode`、`--generation-model`（缺省 `deepseek-chat`）、`--language`（缺省 `en`，HaluMem 是英文对话）、`--ports` 与 `--log-level` 写进实例的配置。
- 同一用户的会话严格按顺序处理：一次会话的各轮导入之后，结算这个 Workspace 的全部话题，等生成任务都结束，再取出这次会话新建与更新的记忆，并对每条更新要点检索前 10 条记忆。
- 结果写到 `<运行目录>/driven/halumem-<编号>.jsonl`，每次会话一行；某次会话失败时写进同目录的 `.errors.jsonl`，这个用户停在那里，其余用户照常。
- 中断后用同样的命令再运行：导入按外部标识去重，从每个用户最后一个完成的会话之后继续。

一次性生成与馆长各用一个运行目录、一组端口，可以同时运行。参考规模：Medium 的 5 个用户（346 次会话、7,499 轮）并行跑时，一次性生成约 3 小时，馆长约 2 小时。

## 5. 判分与报告

```powershell
python scripts/halumem_eval.py judge .hivememory/lab/datasets/halumem/HaluMem-Medium.jsonl .hivememory/lab/runs/eval-one-shot --prompts .hivememory/lab/halumem/prompts --judge-model gpt-6-luna --reasoning-effort none --concurrency 16
python scripts/halumem_eval.py report .hivememory/lab/runs/eval-one-shot --compare .hivememory/lab/runs/eval-librarian
```

- **judge**：按官方协议列出全部评判调用，温度为 0，要求 JSON 输出，结果追加到 `<运行目录>/judged.jsonl`。
  - 评判模型与这次运行的生成模型相同时拒绝运行。
  - 推理模型可能只在关闭推理时接受温度，例如 `gpt-6-luna` 要 `--reasoning-effort none`；不给这个参数时不发送。
  - 随机抽约 10% 的调用再判一次，报告两次是否一致。
  - 可以续跑：已判过的调用跳过；提供商失败的调用不写入，再运行一次 `judge` 时补判。模型回了话但解析不出结果的，按无分写入。
- **report**：按官方的口径汇总，写出 `<运行目录>/report.md` 与 `metrics.json`；`--compare` 把其他运行目录并排，`--ledger` 向一份账本文件追加一条摘要。
- **用量**：报告记录话题路由的调用次数与生成任务数（来自试验实例的运行时事件）、馆长的步数与 token（来自结算日志），以及评判的 token。路由与一次性生成的 token 没有记录，花费以提供商的账单为准。

## 6. 排查

| 现象 | 原因与处理 |
|:---|:---|
| “端口 … 已被占用”或“属于日常实例” | 换一组 `--ports`；不要停掉日常实例来腾端口 |
| 数据根路径过长 | 运行目录换到更浅的位置 |
| “没有找到模型缓存”或“没有可用的 QDRANT_BIN” | 先按第 1 节准备 `data/model_cache` 与 `.hivememory/dev.env` |
| “… 在 600 秒内没有就绪” | 看 `<运行目录>/instance/logs/backend.log` 与 `qdrant.log` |
| “试验实例没有登记这些用户” | 运行目录已用其他用户建立过；换一个运行目录 |
| `judge` 结束时提示提供商失败 | 再运行一次同样的 `judge` 命令补判；失败全是 `UnsupportedParamsError` 时，推理模型需要 `--reasoning-effort none` |
