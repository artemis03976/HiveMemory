"""MTP CALL 指令解析测试。"""

import json

from hivememory.core.mtp import MTPParser, MTPVerb


class TestMTPCallParsing:
    """MTP CALL 指令解析测试"""

    def test_parse_call_basic(self):
        """基本 CALL 指令解析"""
        parser = MTPParser()
        cmd = parser.parse('⟪ CALL | coder_doll | task="Write unit tests" ⟫')
        assert cmd.verb == MTPVerb.CALL
        assert cmd.target.single_alias == "coder_doll"
        assert cmd.args.get("task") == "Write unit tests"

    def test_parse_call_with_context_refs(self):
        """带 context_refs 的 CALL 指令"""
        parser = MTPParser()
        cmd = parser.parse(
            '⟪ CALL | backend_doll | task="实现接口" context_refs=["mem_api_spec", "mem_db_schema"] ⟫'
        )
        assert cmd.verb == MTPVerb.CALL
        assert cmd.target.single_alias == "backend_doll"
        assert cmd.args.get("task") == "实现接口"
        refs = json.loads(cmd.args["context_refs"])
        assert refs == ["mem_api_spec", "mem_db_schema"]

    def test_parse_call_without_context_refs(self):
        """不带 context_refs 的 CALL 指令"""
        parser = MTPParser()
        cmd = parser.parse('⟪ CALL | tester_doll | task="Run all tests" ⟫')
        assert cmd.verb == MTPVerb.CALL
        assert "context_refs" not in cmd.args

    def test_parse_list_args_single_item(self):
        """列表参数 — 单个元素"""
        parser = MTPParser()
        cmd = parser.parse(
            '⟪ CALL | coder | task="test" context_refs=["mem_spec"] ⟫'
        )
        refs = json.loads(cmd.args["context_refs"])
        assert refs == ["mem_spec"]
