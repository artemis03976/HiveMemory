---
title: Package Version Metadata Consistency
status: todo
owner: project
scope: package-version-and-release-metadata
related_docs:
  - docs/PROJECT.md
  - docs/ROADMAP.md
  - docs/help/setup.md
last_reviewed: 2026-07-29
---

# 统一包版本元数据与发布口径

## 问题与证据

项目文档和根 README 已采用“最新发布 v0.5.0、当前开发基线 v0.6.0”的口径，但 `pyproject.toml` 仍声明 `0.1.0-beta`，包内 `__version__` 声明 `0.6.0`。这会让构建产物、运行时诊断和文档显示不同版本。

## 影响

- 发布包无法准确表达其开发基线或 tag；
- 用户报告问题时，版本信息可能不足以定位代码状态；
- Help、CI、README 和 release 检查容易再次漂移。

## 完成条件

- 明确唯一版本来源，以及开发快照、预发布和正式 tag 的规则；
- `pyproject.toml`、包内版本、构建 metadata、README、Help 和 Roadmap 对同一提交给出一致口径；
- CI 或 release check 在版本不一致时失败；
- 保留 v0.5.0/v0.6.0 的历史关系，不把未发布开发基线写成已发布版本。
