"""进程提示词中的记忆与附件顺序，以及空段省略与正文保真。"""

import pytest

from hivememory.core.models import AgentProfile
from hivememory.prompts.compilation import compile_prompt_input


@pytest.mark.parametrize(
    "memory,attachment,expected",
    [
        ("MEMORY", " \nATTACHMENT\n ", "MEMORY\n\n \nATTACHMENT\n "),
        ("MEMORY", "", "MEMORY"),
        ("", "ATTACHMENT", "ATTACHMENT"),
    ],
)
def test_turn_context_preserves_material_order_and_omits_empty_sections(
    memory, attachment, expected
):
    """记忆总在附件前，空段不占位置；附件中的空白和边界文本不改写。"""
    compiled = compile_prompt_input(
        agent_profile=AgentProfile(),
        storage_available=True,
        memory_context=memory,
        attachment_context=attachment,
    )
    assert compiled.turn_context == expected
