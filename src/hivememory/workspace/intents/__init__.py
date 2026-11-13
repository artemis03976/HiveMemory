"""Workspace 共享写入意图登记：状态唯一所有者，与任务进程生命周期解耦。"""

from hivememory.workspace.intents.registry import WRITE_INTENT_ALIAS_PREFIXES, WriteIntentRegistry

__all__ = ["WRITE_INTENT_ALIAS_PREFIXES", "WriteIntentRegistry"]
