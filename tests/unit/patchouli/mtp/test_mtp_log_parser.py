"""
MTPLogParser 单元测试

测试覆盖:
- parse: 空输入 / 纯文本 / 各种 MTP 指令
- _extract_arg: 双引号 / 反引号 / 缺失 key
- 边界: 跨行指令 / 空指令体 / 多余空行折叠
"""

import pytest

from hivememory.patchouli.mtp.log_parser import MTPLogParser
from hivememory.patchouli.mtp import MTP_LEFT_DELIMITER, MTP_RIGHT_DELIMITER


L = MTP_LEFT_DELIMITER  # ⟪
R = MTP_RIGHT_DELIMITER  # ⟫


class TestMTPLogParserParse:
    """parse() 主方法测试"""

    def test_empty_input(self):
        """空字符串返回空"""
        clean, traces = MTPLogParser.parse("")
        assert clean == ""
        assert traces == []

    def test_plain_text_passthrough(self):
        """无 MTP 指令的文本原样返回"""
        text = "这是一段普通文本，没有任何 MTP 指令。"
        clean, traces = MTPLogParser.parse(text)
        assert clean == text
        assert traces == []

    def test_single_read_command(self):
        """解析 READ 指令"""
        text = f"前文{L}READ|mem_auth_doc{R}后文"
        clean, traces = MTPLogParser.parse(text)
        assert "前文" in clean
        assert "后文" in clean
        assert L not in clean
        assert len(traces) == 1
        assert traces[0].action == "READ"
        assert traces[0].target == "mem_auth_doc"

    def test_single_search_command(self):
        """解析 SEARCH 指令提取 query"""
        text = f'{L}SEARCH | * | query="docker config"{R}'
        clean, traces = MTPLogParser.parse(text)
        assert len(traces) == 1
        assert traces[0].action == "SEARCH"
        assert traces[0].query == "docker config"

    def test_single_run_command(self):
        """解析 RUN 指令"""
        text = f"{L}RUN|sys_write_file{R}"
        clean, traces = MTPLogParser.parse(text)
        assert len(traces) == 1
        assert traces[0].action == "RUN"
        assert traces[0].tool == "sys_write_file"
        assert traces[0].status == "unknown"

    def test_write_command_no_trace(self):
        """WRITE 指令不生成 TraceItem"""
        text = f"{L}WRITE|some content{R}"
        clean, traces = MTPLogParser.parse(text)
        assert traces == []

    def test_update_command_no_trace(self):
        """UPDATE 指令不生成 TraceItem"""
        text = f"{L}UPDATE|mem_123{R}"
        clean, traces = MTPLogParser.parse(text)
        assert traces == []

    def test_unknown_verb_no_trace(self):
        """未知动词不生成 TraceItem"""
        text = f"{L}FOOBAR|something{R}"
        clean, traces = MTPLogParser.parse(text)
        assert traces == []

    def test_multiple_mixed_commands(self):
        """多个混合指令正确解析"""
        text = (
            f"开头文本 "
            f'{L}READ|mem_1{R} '
            f'中间文本 '
            f'{L}SEARCH|query="python tips"{R} '
            f'{L}WRITE|content{R} '
            f'{L}RUN|tool_x{R} '
            f"结尾文本"
        )
        clean, traces = MTPLogParser.parse(text)
        assert "开头文本" in clean
        assert "结尾文本" in clean
        assert L not in clean
        # WRITE 不生成 trace，所以只有 3 个
        assert len(traces) == 3
        assert traces[0].action == "READ"
        assert traces[1].action == "SEARCH"
        assert traces[2].action == "RUN"

    def test_mtp_response_not_in_assistant_text(self):
        """角色分离注入后，<mtp_response> 隔离在独立的 user 消息中，
        不再出现在 assistant 文本里，MTPLogParser 只需清理 MTP 指令"""
        text = "正文后续"
        clean, _ = MTPLogParser.parse(text)
        assert clean == "正文后续"

    def test_excessive_newlines_collapsed(self):
        """3+ 空行折叠为 2 行"""
        text = "第一段\n\n\n\n\n第二段"
        clean, traces = MTPLogParser.parse(text)
        assert "\n\n\n" not in clean
        assert "第一段\n\n第二段" == clean


class TestMTPLogParserEdgeCases:
    """边界情况测试"""

    def test_multiline_command_body(self):
        """跨行 MTP 指令"""
        text = f"{L}READ|mem_doc\nsome extra line{R}"
        clean, traces = MTPLogParser.parse(text)
        assert len(traces) == 1
        assert traces[0].action == "READ"

    def test_empty_command_body(self):
        """空指令体"""
        text = f"{L}{R}"
        clean, traces = MTPLogParser.parse(text)
        # 空 body strip 后为 ""，parts[0] 为 ""，upper() 为 ""，不匹配任何 verb
        assert traces == []
