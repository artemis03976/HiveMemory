"""Envelope templates used by MemoryCompiler."""

MEMORY_HEADER = """<memory_context>
[System Guidance]: 帕秋莉 (记忆库的管理者) 为你取回了以下相关的历史记忆与可用子代理。
你可以将记忆信息视为你脑海里自然而然浮现的"潜意识"，作为背景知识直接融合到你的思考中，无需刻意生硬地声明"根据记忆显示"。
"""

MEMORY_FOOTER = """
\n[System Guidance]:
- 若上述记忆摘要符合当前用户意图，但摘要信息不足，希望查看完整的记忆内容，请立即使用 `⟪ READ | alias | ⟫` 指令（**严禁自行猜测或编造**）。
- 带有 [未验证] 或 (警告：陈旧) 状态的记忆可能包含错误或过时信息，请结合常识注意甄别。
- 若任务需要专项能力（如数据分析、代码生成等），且上方列出了对应子代理，请优先使用 `⟪ CALL | agent_alias | topic="..." ⟫` 委托给子代理执行，不要自行承担。
</memory_context>
"""


__all__ = [
    "MEMORY_HEADER",
    "MEMORY_FOOTER",
]
