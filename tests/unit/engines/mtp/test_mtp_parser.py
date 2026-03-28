"""
MTP 协议解析器与格式化器单元测试

测试覆盖:
- MTPParser: 5 种 VERB 解析、3 种 TARGET 形态、2 种 ARGS 格式
- MTPFormatter: XML 响应容器格式化、回填文本生成
- 边界情况: 不完整指令补全、ARGS 内部管道符、解析错误

对应设计文档: MemoryToolProtocol.md Chapter 2 & 3.3
"""

import pytest

from hivememory.patchouli.protocol.mtp import (
    MTP_LEFT_DELIMITER,
    MTP_RIGHT_DELIMITER,
    MTP_STOP_SEQUENCE,
    MTPVerb,
    MTPResponseStatus,
    MTPTarget,
    MTPCommand,
    MTPResponse,
    MTPParser,
    MTPParseError,
    MTPFormatter,
    create_parser,
    create_formatter,
)


# ========== Fixtures ==========

@pytest.fixture
def parser() -> MTPParser:
    return MTPParser()


@pytest.fixture
def formatter() -> MTPFormatter:
    return MTPFormatter()


# ========== 常量测试 ==========

class TestConstants:
    """测试 MTP 协议常量"""

    def test_delimiters(self):
        assert MTP_LEFT_DELIMITER == "\u27EA"
        assert MTP_RIGHT_DELIMITER == "\u27EB"
        assert MTP_STOP_SEQUENCE == MTP_RIGHT_DELIMITER
# PLACEHOLDER_PARSER_TESTS


# ========== 解析器测试 ==========

class TestMTPParser:
    """测试 MTP 协议解析器"""

    # --- VERB 解析 ---

    def test_parse_search(self, parser: MTPParser):
        """测试 SEARCH 指令解析"""
        cmd = parser.parse('⟪ SEARCH | * | query="Redis configuration" ⟫')
        assert cmd.verb == MTPVerb.SEARCH
        assert cmd.target.is_wildcard is True
        assert cmd.args["query"] == "Redis configuration"

    def test_parse_read_single(self, parser: MTPParser):
        """测试 READ 单别名解析"""
        cmd = parser.parse("⟪ READ | fact_api_spec | ⟫")
        assert cmd.verb == MTPVerb.READ
        assert cmd.target.single_alias == "fact_api_spec"
        assert cmd.target.is_list is False

    def test_parse_read_list(self, parser: MTPParser):
        """测试 READ 列表目标解析 (Section 2.4 场景1)"""
        cmd = parser.parse("⟪ READ | [fact_api_spec, tool_db_connector] | ⟫")
        assert cmd.verb == MTPVerb.READ
        assert cmd.target.is_list is True
        assert cmd.target.aliases == ["fact_api_spec", "tool_db_connector"]

    def test_parse_run_sys_tool(self, parser: MTPParser):
        """测试 RUN 系统工具解析 (Section 2.4 场景2)"""
        cmd = parser.parse('⟪ RUN | sys_read_file | path="./config/settings.yaml" ⟫')
        assert cmd.verb == MTPVerb.RUN
        assert cmd.target.single_alias == "sys_read_file"
        assert cmd.args["path"] == "./config/settings.yaml"

    def test_parse_run_mem_tool(self, parser: MTPParser):
        """测试 RUN 记忆工具解析 (Section 2.4 场景3)"""
        cmd = parser.parse('⟪ RUN | tool_deploy_script | env="production" retries="3" ⟫')
        assert cmd.verb == MTPVerb.RUN
        assert cmd.target.single_alias == "tool_deploy_script"
        assert cmd.args["env"] == "production"
        assert cmd.args["retries"] == "3"

    def test_parse_write(self, parser: MTPParser):
        """测试 WRITE 指令解析 (Section 2.4 场景4)"""
        text = '⟪ WRITE | * | title="Login Auth Logic" content=`\ndef login(user):\n    pass\n` ⟫'
        cmd = parser.parse(text)
        assert cmd.verb == MTPVerb.WRITE
        assert cmd.target.is_wildcard is True
        assert cmd.args["title"] == "Login Auth Logic"
        assert "def login(user):" in cmd.args["content"]

    def test_parse_update(self, parser: MTPParser):
        """测试 UPDATE 指令解析"""
        cmd = parser.parse('⟪ UPDATE | fact_old_config | patch=`new_value=42` ⟫')
        assert cmd.verb == MTPVerb.UPDATE
        assert cmd.target.single_alias == "fact_old_config"
        assert cmd.args["patch"] == "new_value=42"

    # --- VERB 大小写 ---

    def test_parse_verb_case_insensitive(self, parser: MTPParser):
        """测试动词大小写不敏感"""
        cmd = parser.parse("⟪ read | fact_api_spec | ⟫")
        assert cmd.verb == MTPVerb.READ

    # --- TARGET 形态 ---

    def test_parse_target_wildcard_star(self, parser: MTPParser):
        """测试通配符 *"""
        cmd = parser.parse('⟪ SEARCH | * | query="test" ⟫')
        assert cmd.target.is_wildcard is True

    def test_parse_target_wildcard_global(self, parser: MTPParser):
        """测试通配符 global"""
        cmd = parser.parse('⟪ SEARCH | global | query="test" ⟫')
        assert cmd.target.is_wildcard is True

    def test_parse_target_list_three(self, parser: MTPParser):
        """测试三元素列表"""
        cmd = parser.parse("⟪ READ | [doc_a, doc_b, doc_c] | ⟫")
        assert cmd.target.aliases == ["doc_a", "doc_b", "doc_c"]
        assert cmd.target.is_list is True

    # --- ARGS 格式 ---

    def test_parse_args_kv(self, parser: MTPParser):
        """测试 key=value 双引号参数"""
        cmd = parser.parse('⟪ SEARCH | * | query="hello" filter="type:CODE" ⟫')
        assert cmd.args["query"] == "hello"
        assert cmd.args["filter"] == "type:CODE"

    def test_parse_args_raw_backtick(self, parser: MTPParser):
        """测试反引号原始内容参数"""
        cmd = parser.parse('⟪ WRITE | * | content=`print("hello")` ⟫')
        assert cmd.args["content"] == 'print("hello")'

    def test_parse_args_mixed(self, parser: MTPParser):
        """测试混合参数格式"""
        text = '⟪ WRITE | * | title="Test" content=`code here` ⟫'
        cmd = parser.parse(text)
        assert cmd.args["title"] == "Test"
        assert cmd.args["content"] == "code here"

    def test_parse_args_empty(self, parser: MTPParser):
        """测试空参数"""
        cmd = parser.parse("⟪ READ | fact_api_spec | ⟫")
        assert cmd.args == {}

    def test_parse_no_args_segment(self, parser: MTPParser):
        """测试无 ARGS 段 (仅两段)"""
        cmd = parser.parse("⟪ READ | fact_api_spec ⟫")
        assert cmd.verb == MTPVerb.READ
        assert cmd.target.single_alias == "fact_api_spec"
        assert cmd.args == {}

    # --- 边界情况 ---

    def test_parse_pipe_in_args(self, parser: MTPParser):
        """测试 ARGS 内部的管道符不被误分割 (Section 2.1)"""
        cmd = parser.parse('⟪ WRITE | * | content=`a | b | c` ⟫')
        assert cmd.args["content"] == "a | b | c"

    def test_parse_multiline_content(self, parser: MTPParser):
        """测试多行内容"""
        text = '⟪ WRITE | * | content=`line1\nline2\nline3` ⟫'
        cmd = parser.parse(text)
        assert "line1\nline2\nline3" == cmd.args["content"]

    def test_parse_raw_text_preserved(self, parser: MTPParser):
        """测试原始文本保留"""
        raw = "⟪ READ | fact_api_spec | ⟫"
        cmd = parser.parse(raw)
        assert cmd.raw_text == raw

    # --- 错误处理 ---

    def test_parse_no_command(self, parser: MTPParser):
        """测试无 MTP 指令"""
        with pytest.raises(MTPParseError, match="No MTP command found"):
            parser.parse("just some text")

    def test_parse_unknown_verb(self, parser: MTPParser):
        """测试未知动词"""
        with pytest.raises(MTPParseError, match="Unknown verb"):
            parser.parse("⟪ DELETE | * | ⟫")

    def test_parse_missing_separator(self, parser: MTPParser):
        """测试缺少分隔符"""
        with pytest.raises(MTPParseError, match="Missing separator"):
            parser.parse("⟪ READ ⟫")
# PLACEHOLDER_COMPLETE_AND_DETECT


# ========== 补全与检测测试 ==========

class TestMTPParserCompleteAndDetect:
    """测试 MTP 指令补全与检测"""

    def test_complete_and_parse(self, parser: MTPParser):
        """测试不完整指令补全 (Section 3.1.2 Stop Sequence)"""
        # 模拟 LLM 在 ⟫ 处被截断
        incomplete = "⟪ READ | fact_api_spec |"
        cmd = parser.complete_and_parse(incomplete)
        assert cmd.verb == MTPVerb.READ
        assert cmd.target.single_alias == "fact_api_spec"

    def test_complete_already_complete(self, parser: MTPParser):
        """测试已完整的指令不会重复补全"""
        complete = "⟪ READ | fact_api_spec | ⟫"
        cmd = parser.complete_and_parse(complete)
        assert cmd.verb == MTPVerb.READ

    def test_detect_command_present(self, parser: MTPParser):
        """测试检测到 MTP 指令"""
        text = "I need to check ⟪ SEARCH | * | query=\"test\" ⟫"
        assert parser.detect_command(text) is True

    def test_detect_command_absent(self, parser: MTPParser):
        """测试未检测到 MTP 指令"""
        assert parser.detect_command("just normal text") is False

    def test_detect_command_left_delimiter_only(self, parser: MTPParser):
        """测试仅有左定界符 (Stop Sequence 截断场景)"""
        text = "Let me search ⟪ SEARCH | * | query=\"test\""
        assert parser.detect_command(text) is True


# ========== 格式化器测试 ==========

class TestMTPFormatter:
    """测试 MTP 响应格式化器"""

    def test_format_success_response(self, formatter: MTPFormatter):
        """测试成功响应格式化"""
        response = MTPResponse(
            status=MTPResponseStatus.SUCCESS,
            content="[mem_01]: def login(): ...",
            execution_time_ms=42.0,
        )
        result = formatter.format_response(response)
        assert '<mtp_response status="success" time="42ms">' in result
        assert "[mem_01]: def login(): ..." in result
        assert "</mtp_response>" in result

    def test_format_error_response(self, formatter: MTPFormatter):
        """测试错误响应格式化"""
        response = MTPResponse(
            status=MTPResponseStatus.ERROR,
            content="Tool alias not found.",
        )
        result = formatter.format_response(response)
        assert '<mtp_response status="error">' in result
        assert "Tool alias not found." in result

    def test_format_ack_response(self, formatter: MTPFormatter):
        """测试 ACK 响应格式化"""
        response = MTPResponse(
            status=MTPResponseStatus.ACK,
            content='Memory saved.',
        )
        result = formatter.format_response(response)
        assert '<mtp_response status="ack">' in result

    def test_format_command_with_response(self, formatter: MTPFormatter):
        """测试完整回填文本格式化 (Section 3.3.1)"""
        command = MTPCommand(
            verb=MTPVerb.READ,
            target=MTPTarget(aliases=["mem_01"]),
            raw_text="⟪ READ | [mem_01] | ⟫",
        )
        response = MTPResponse(
            status=MTPResponseStatus.SUCCESS,
            content="[mem_01]: def login(): ...",
            execution_time_ms=15.0,
        )
        result = formatter.format_command_with_response(command, response)
        assert result.startswith("⟪ READ | [mem_01] | ⟫")
        assert '<mtp_response status="success"' in result
        assert "[mem_01]: def login(): ..." in result

    def test_format_no_time_when_zero(self, formatter: MTPFormatter):
        """测试耗时为 0 时不显示 time 属性"""
        response = MTPResponse(
            status=MTPResponseStatus.SUCCESS,
            content="ok",
            execution_time_ms=0.0,
        )
        result = formatter.format_response(response)
        assert "time=" not in result


# ========== 工厂函数测试 ==========

class TestFactoryFunctions:
    """测试工厂函数"""

    def test_create_parser(self):
        p = create_parser()
        assert isinstance(p, MTPParser)

    def test_create_formatter(self):
        f = create_formatter()
        assert isinstance(f, MTPFormatter)

