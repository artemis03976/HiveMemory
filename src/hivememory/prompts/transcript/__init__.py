"""面向提示词的对话视图 Transcript 构建器。"""

from hivememory.prompts.transcript.generation import GenerationTranscriptBuilder
from hivememory.prompts.transcript.history import HistoryTranscriptBuilder

__all__ = [
    "GenerationTranscriptBuilder",
    "HistoryTranscriptBuilder",
]
