---
title: Agent Error Formatter XML Escaping
status: todo
owner: contracts
scope: agent-facing-error-payload-escaping
related_docs:
  - docs/contracts/error-model.md
  - docs/contracts/mtp.md
  - docs/alice/mtp-runtime.md
last_reviewed: 2026-07-29
---

# 补齐 Agent-facing 错误 payload 的 XML escaping

## 问题与证据

当前错误结构、业务 code 和控制异常已经有分层，但 formatter 尚未为所有业务 content/reply/warning 提供统一 XML escaping。若错误正文或警告含有 XML 保留字符，MTP 文本可能无法被严格解析。

## 影响

- handler 各自转义会造成结果不一致；
- 错误本身可能进一步触发 parser failure，掩盖原始原因；
- Agent-facing 协议的格式合法性依赖具体调用点。

## 完成条件

- 在统一 formatter 层覆盖 content、reply、warning、error reason 和可选摘要；
- 明确 CDATA、实体转义、换行和空值语义，不重复转义已经编码的内容；
- 增加保留字符、Unicode、超长文本和嵌套 payload 的协议测试；
- 删除各 verb handler 的重复 escaping，并更新 MTP/error model 文档。
