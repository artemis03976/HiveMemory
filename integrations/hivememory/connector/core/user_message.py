"""用户消息中宿主附加的内容。

Claude 桌面端等宿主会把状态说明以标签块的形式附加在用户消息里（关联会话的状态、
画布的视图状态、最近的项目文件夹等）。它们不是用户说的话，导入前去掉：话题路由
读取用户消息全文，记忆生成也以它为材料。

只去掉 ``INJECTED_BLOCKS`` 中列出的完整块，且块必须从行首开始。代码块与用户粘贴的
内容（``<pasted_content>``，保留标签以区分粘贴与本人的话）中出现的同名标签原样保留，
那是用户自己的内容。表中没有的标签不猜测：``scripts/transcript_stats.py`` 列出会话记录
中出现过的行首标签，确认是宿主附加的再加入。
"""

from __future__ import annotations

import re

INJECTED_BLOCKS = ("system-reminder", "artifact-view-context", "recent-project-folders")

_PROTECTED = re.compile(r"(?ms)^```.*?^```|^<pasted_content\b[^>]*>.*?</pasted_content\b[^>]*>")
_INJECTED = re.compile(r"(?ms)^<(" + "|".join(INJECTED_BLOCKS) + r")\b[^>]*>.*?</\1>\s*")


def strip_injected(text: str) -> str:
    """去掉宿主附加的块，代码块与粘贴的内容不动。"""
    parts = []
    start = 0
    for protected in _PROTECTED.finditer(text):
        parts.append(_INJECTED.sub("", text[start : protected.start()]))
        parts.append(protected.group(0))
        start = protected.end()
    parts.append(_INJECTED.sub("", text[start:]))
    return "".join(parts).strip()
