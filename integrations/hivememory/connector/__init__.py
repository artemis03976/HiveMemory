"""HiveMemory 的外部 harness 连接器：把 harness 会话记录中已结束的轮次导入 Import Bus。

分为两层：``connector.core`` 是与 harness 无关的公共骨架（配置、状态与游标、
提交与重试分类、同步循环），``connector.claude_code`` 是 Claude Code 的会话
记录读取器与 hooks 入口。连接器只依赖 Python 标准库，不导入 HiveMemory 包。
"""
