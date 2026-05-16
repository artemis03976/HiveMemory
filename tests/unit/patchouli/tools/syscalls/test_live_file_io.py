import logging
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hivememory.system.config import KoakumaConfig
from hivememory.patchouli.kernel.koakuma import KoakumaRuntime

from .live_support import (
    MTPLoopRunner,
    _build_mtp_system_prompt_with_file_io,
    _create_llm_service,
    _get_llm_config,
)

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.live_llm


@pytest.fixture(scope="module")
def llm_config():
    config = _get_llm_config()
    if config is None:
        pytest.skip(
            "LLM API not configured. Set MTP_TEST_MODEL and MTP_TEST_API_KEY environment variables."
        )
    return config


@pytest.fixture(scope="module")
def llm_service(llm_config):
    return _create_llm_service(llm_config)


@pytest.fixture
def workspace_dir():
    tmp = tempfile.mkdtemp(prefix="mtp_test_workspace_")
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def koakuma_with_workspace(workspace_dir):
    return KoakumaRuntime(
        retrieval_familiar=MagicMock(),
        librarian_core=MagicMock(),
        storage=MagicMock(),
        config=KoakumaConfig(workspace_path=workspace_dir),
    )


@pytest.fixture
def file_io_loop_runner(llm_service, koakuma_with_workspace):
    return MTPLoopRunner(
        llm_service=llm_service,
        koakuma=koakuma_with_workspace,
        max_rounds=5,
        temperature=0.0,
        max_tokens=1024,
    )


@pytest.fixture
def mtp_system_prompt_file_io():
    return _build_mtp_system_prompt_with_file_io(language="en")


@pytest.fixture
def mtp_system_prompt_file_io_zh():
    return _build_mtp_system_prompt_with_file_io(language="zh")


class TestMTPLoopReadFile:
    """验证 sys_read_file 在真实 LLM 场景中的完整 MTP 循环"""

    def test_read_file_full_loop(
        self, file_io_loop_runner, mtp_system_prompt_file_io, workspace_dir,
    ):
        """LLM 读取工作区文件并将内容呈现给用户"""
        seed_path = Path(workspace_dir) / "notes.txt"
        seed_path.write_text("Meeting at 3pm with the design team.", encoding="utf-8")

        final_text, messages = file_io_loop_runner.run(
            system_prompt=mtp_system_prompt_file_io,
            user_message="Read the file notes.txt from the workspace and tell me what it says.",
        )

        assert len(file_io_loop_runner.round_log) >= 2
        round1 = file_io_loop_runner.round_log[0]
        assert round1["mtp_triggered"] is True
        assert round1["mtp_result"]["success"] is True

        all_text = " ".join(
            m["content"] for m in messages if m["role"] == "assistant"
        ) + " " + final_text
        assert "3pm" in all_text or "design team" in all_text

    def test_read_file_not_found_recovery(
        self, file_io_loop_runner, mtp_system_prompt_file_io,
    ):
        """LLM 尝试读取不存在的文件后能正确处理错误"""
        final_text, messages = file_io_loop_runner.run(
            system_prompt=mtp_system_prompt_file_io,
            user_message="Read the file missing.txt from the workspace.",
        )

        assert len(file_io_loop_runner.round_log) >= 1
        if file_io_loop_runner.round_log[0]["mtp_triggered"]:
            all_text = " ".join(
                m["content"] for m in messages if m["role"] == "assistant"
            ) + " " + final_text
            has_error_awareness = any(
                kw in all_text.lower()
                for kw in ["not found", "error", "does not exist", "doesn't exist", "no such", "找不到", "不存在"]
            )
            assert has_error_awareness

    def test_read_file_content_used_in_answer(
        self, file_io_loop_runner, mtp_system_prompt_file_io, workspace_dir,
    ):
        """LLM 读取文件后能基于内容回答问题"""
        seed_path = Path(workspace_dir) / "config.txt"
        seed_path.write_text("server_port=8080\ndebug_mode=true\nmax_connections=100", encoding="utf-8")

        final_text, messages = file_io_loop_runner.run(
            system_prompt=mtp_system_prompt_file_io,
            user_message="Read config.txt and tell me what port the server is running on.",
        )

        all_text = " ".join(
            m["content"] for m in messages if m["role"] == "assistant"
        ) + " " + final_text
        assert "8080" in all_text


# ========== Test 9: sys_write_file 完整循环 ==========

class TestMTPLoopWriteFile:
    """验证 sys_write_file 在真实 LLM 场景中的完整 MTP 循环"""

    def test_write_file_full_loop(
        self, file_io_loop_runner, mtp_system_prompt_file_io, workspace_dir,
    ):
        """LLM 写入文件到工作区并确认成功"""
        final_text, messages = file_io_loop_runner.run(
            system_prompt=mtp_system_prompt_file_io,
            user_message='Write a file called hello.txt in the workspace with the content "Hello, World!".',
        )

        assert len(file_io_loop_runner.round_log) >= 1
        if file_io_loop_runner.round_log[0]["mtp_triggered"]:
            assert file_io_loop_runner.round_log[0]["mtp_result"]["success"] is True

        written = Path(workspace_dir) / "hello.txt"
        assert written.exists(), f"File was not created in workspace: {workspace_dir}"
        content = written.read_text(encoding="utf-8")
        assert "Hello" in content

    def test_write_then_read_round_trip(
        self, file_io_loop_runner, mtp_system_prompt_file_io, workspace_dir,
    ):
        """LLM 先写入文件再读取，验证完整读写往返"""
        final_text, messages = file_io_loop_runner.run(
            system_prompt=mtp_system_prompt_file_io,
            user_message=(
                'First, write a file called data.txt with the content "price=42". '
                'Then read data.txt and tell me the price.'
            ),
        )

        mtp_rounds = [r for r in file_io_loop_runner.round_log if r["mtp_triggered"]]
        assert len(mtp_rounds) >= 1

        all_text = " ".join(
            m["content"] for m in messages if m["role"] == "assistant"
        ) + " " + final_text
        assert "42" in all_text

    def test_write_file_confirms_success(
        self, file_io_loop_runner, mtp_system_prompt_file_io, workspace_dir,
    ):
        """LLM 写入文件后向用户确认操作成功"""
        final_text, messages = file_io_loop_runner.run(
            system_prompt=mtp_system_prompt_file_io,
            user_message='Save the text "TODO: fix bug #123" to a file called todo.txt in the workspace.',
        )

        if file_io_loop_runner.round_log[0]["mtp_triggered"]:
            assert file_io_loop_runner.round_log[0]["mtp_result"]["success"] is True

        all_text = " ".join(
            m["content"] for m in messages if m["role"] == "assistant"
        ) + " " + final_text
        has_confirmation = any(
            kw in all_text.lower()
            for kw in ["saved", "written", "created", "success", "done", "已保存", "已写入", "完成"]
        )
        assert has_confirmation

    def test_write_file_chinese_scenario(
        self, file_io_loop_runner, mtp_system_prompt_file_io_zh, workspace_dir,
    ):
        """中文场景下写入文件"""
        final_text, messages = file_io_loop_runner.run(
            system_prompt=mtp_system_prompt_file_io_zh,
            user_message='把"今天天气不错"写入到 weather.txt 文件中。',
        )

        assert len(file_io_loop_runner.round_log) >= 1
        if file_io_loop_runner.round_log[0]["mtp_triggered"]:
            assert file_io_loop_runner.round_log[0]["mtp_result"]["success"] is True

