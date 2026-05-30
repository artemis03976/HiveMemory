"""
Relay Summary System Prompt Templates

Generates structured state snapshots for LLM-based relay compression.
Used by LLMRelayController to compress conversation history into dense summaries.

Design: LLMSummary.md
Author: HiveMemory Team
Version: 1.0
"""

from hivememory.i18n import get_relay_prompt_text

def get_relay_system_prompt(language: str = "zh") -> str:
    """
    Get relay compression system prompt

    Args:
        language: "zh" or "en"

    Returns:
        System prompt string
    """
    return get_relay_prompt_text("system_prompt", language)


__all__ = ["get_relay_system_prompt"]
