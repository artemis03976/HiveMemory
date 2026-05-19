from types import SimpleNamespace
from unittest.mock import MagicMock

from hivememory.core.protocol.models import RetrievalResponse
from hivememory.patchouli.message_assembler import MessageAssembler


def _make_runtime():
    mtp_prompt_config = SimpleNamespace(
        enabled=True,
        language="zh",
        include_demo=False,
        include_error_handling=False,
    )
    koakuma_config = SimpleNamespace(
        enabled=True,
        mtp_prompt=mtp_prompt_config,
    )
    config = SimpleNamespace(koakuma=koakuma_config)

    runtime = MagicMock()
    runtime.config = config
    runtime.check_storage_health.return_value = True
    return runtime


def test_assemble_builds_prompt_without_runtime_get_mtp_prompt():
    runtime = _make_runtime()
    assembler = MessageAssembler(runtime)
    profile = SimpleNamespace(
        persona="你是一个测试人偶。",
        allowed_mtp_verbs=["SEARCH", "READ"],
        allowed_sys_tools=["sys_clock"],
    )
    retrieval_result = RetrievalResponse(
        memories=[],
        rendered_context="<memory>ctx</memory>",
    )

    messages = assembler.assemble(
        topic_context={"state_summary": "state", "blocks": []},
        retrieval_result=retrieval_result,
        user_message="hello",
        profile=profile,
    )

    assert messages[0]["role"] == "system"
    assert "协议规则" in messages[0]["content"]
    assert "你是一个测试人偶。" in messages[0]["content"]
    assert "<memory>ctx</memory>" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "hello"}
