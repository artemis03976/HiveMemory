from hivememory.patchouli.mtp.models import MTP_LEFT_DELIMITER, MTP_RIGHT_DELIMITER, MTPVerb


class TestSyscallPromptTeaching:
    """MTP System Prompt 教学内容验证"""

    def test_prompt_contains_mtp_syntax(self, mtp_prompt_en):
        assert MTP_LEFT_DELIMITER in mtp_prompt_en
        assert MTP_RIGHT_DELIMITER in mtp_prompt_en
        assert "VERB" in mtp_prompt_en
        assert "TARGET" in mtp_prompt_en
        assert "ARGS" in mtp_prompt_en

    def test_prompt_lists_mvp_syscalls(self, mtp_prompt_en):
        assert "sys_clock" in mtp_prompt_en
        assert "sys_python_repl" in mtp_prompt_en

    def test_prompt_contains_run_verb(self, mtp_prompt_en):
        assert "RUN" in mtp_prompt_en
        assert "Execute" in mtp_prompt_en or "execute" in mtp_prompt_en

    def test_prompt_contains_demo(self, mtp_prompt_en):
        assert "<mtp_response" in mtp_prompt_en
        assert "</mtp_response>" in mtp_prompt_en

    def test_prompt_contains_error_recovery(self, mtp_prompt_en):
        assert "ERROR RECOVERY" in mtp_prompt_en or "error" in mtp_prompt_en.lower()
        assert "retry" in mtp_prompt_en.lower()

    def test_prompt_forbids_json(self, mtp_prompt_en):
        assert "JSON" in mtp_prompt_en
        assert "NEVER" in mtp_prompt_en or "NOT" in mtp_prompt_en

    def test_prompt_zh_structure(self, mtp_prompt_zh):
        assert MTP_LEFT_DELIMITER in mtp_prompt_zh
        assert "sys_clock" in mtp_prompt_zh
        assert "sys_python_repl" in mtp_prompt_zh
        assert "<mtp_response" in mtp_prompt_zh

    def test_prompt_teaches_inline_flow(self, mtp_prompt_en):
        lower = mtp_prompt_en.lower()
        assert "inline" in lower or "thought process" in lower

    def test_prompt_demo_parseable(self, mtp_prompt_en):
        """Prompt 演示中的 MTP 指令可被解析器正确解析"""
        from hivememory.patchouli.mtp.parser import MTPParser
        parser = MTPParser()

        demo_marker = "ONE-SHOT DEMONSTRATION"
        demo_start = mtp_prompt_en.find(demo_marker)
        assert demo_start != -1, "Prompt 中未找到演示部分"

        demo_text = mtp_prompt_en[demo_start:]
        left = demo_text.find(MTP_LEFT_DELIMITER)
        right = demo_text.find(MTP_RIGHT_DELIMITER, left)
        assert left != -1 and right != -1

        demo_cmd = demo_text[left:right + 1]
        cmd = parser.parse(demo_cmd)
        assert cmd.verb in (
            MTPVerb.SEARCH, MTPVerb.READ,
            MTPVerb.RUN, MTPVerb.WRITE, MTPVerb.UPDATE,
        )
