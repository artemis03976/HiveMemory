"""
MTP System Prompt 构建器单元测试

测试覆盖:
- MTPPromptBuilder: 各模块组装、语言切换、角色切换
- get_mtp_prompt: 便捷函数
- 模板内容验证: 定界符正确、指令集完整、演示格式正确
- 配置驱动: 可选模块开关

对应设计文档: MemoryToolProtocol.md Chapter 5
"""

import pytest

from hivememory.prompts.mtp import (
    MTPPromptBuilder,
    get_mtp_prompt,
    DEFAULT_KERNEL_TOOLS,
)
from hivememory.patchouli.protocol.mtp import (
    MTP_LEFT_DELIMITER,
    MTP_RIGHT_DELIMITER,
)
from hivememory.patchouli.config import MTPPromptConfig, KoakumaConfig


# ========== MTPPromptBuilder 单元测试 ==========

class TestMTPPromptBuilder:
    """测试 MTPPromptBuilder 各模块组装"""

    def test_build_default_contains_all_sections(self):
        """默认构建包含所有模块"""
        output = MTPPromptBuilder().build()

        # 标题
        assert "HIVE MEMORY" in output
        # 全部 5 个 MTP 动词
        for verb in ["SEARCH", "READ", "RUN", "WRITE", "UPDATE"]:
            assert verb in output
        # Unicode 定界符
        assert MTP_LEFT_DELIMITER in output
        assert MTP_RIGHT_DELIMITER in output
        # 内核工具
        assert "[KERNEL TOOLS]" in output
        # 演示部分
        assert "mtp_response" in output

    def test_build_english(self):
        """英文模式输出英文内容"""
        output = MTPPromptBuilder(language="en").build()

        assert "PROTOCOL RULES" in output
        assert "CONSTRAINTS" in output
        assert "BEHAVIORAL GUIDELINES" in output
        assert "ONE-SHOT DEMONSTRATION" in output

    def test_build_chinese(self):
        """中文模式输出中文内容"""
        output = MTPPromptBuilder(language="zh").build()

        assert "协议规则" in output
        assert "约束" in output
        assert "行为准则" in output
        assert "示例演示" in output

    def test_build_without_demo(self):
        """关闭演示模块"""
        output = MTPPromptBuilder(include_demo=False, language="en").build()

        # 演示部分不存在
        assert "ONE-SHOT DEMONSTRATION" not in output
        # 协议规则仍在
        assert "PROTOCOL RULES" in output

    def test_build_without_error_handling(self):
        """关闭错误恢复模块"""
        output = MTPPromptBuilder(include_error_handling=False, language="en").build()

        assert "ERROR RECOVERY" not in output
        # 其他模块仍在
        assert "PROTOCOL RULES" in output

    def test_build_without_kernel_tools(self):
        """空白名单不渲染内核工具列表"""
        output = MTPPromptBuilder(allowed_kernel_tools=[]).build()
        assert "[KERNEL TOOLS]" not in output

    def test_build_custom_kernel_tools(self):
        """自定义内核工具列表"""
        custom_tools = [("my_tool", "Does something cool")]
        output = MTPPromptBuilder(kernel_tools=custom_tools).build()

        assert "my_tool" in output
        assert "Does something cool" in output
        # 默认工具不应出现
        assert "sys_clock" not in output

    def test_build_empty_kernel_tools(self):
        """空工具列表不生成工具部分"""
        output = MTPPromptBuilder(kernel_tools=[]).build()
        assert "[KERNEL TOOLS]" not in output

    def test_delimiters_are_actual_unicode(self):
        """定界符是实际的 Unicode 字符"""
        output = MTPPromptBuilder(language="en").build()
        assert "\u27EA" in output  # ⟪
        assert "\u27EB" in output  # ⟫

    def test_demo_shows_search_read_run_flow(self):
        """演示展示完整的 SEARCH → READ → RUN 流程"""
        output = MTPPromptBuilder(language="en").build()

        # 找到演示部分中 SEARCH, READ, RUN 的位置
        demo_start = output.index("ONE-SHOT DEMONSTRATION")
        demo_text = output[demo_start:]

        search_pos = demo_text.index("SEARCH")
        read_pos = demo_text.index("READ")
        run_pos = demo_text.index("RUN")

        # 顺序正确
        assert search_pos < read_pos < run_pos

        # 包含 mtp_response XML 块
        assert "<mtp_response" in demo_text
        assert "</mtp_response>" in demo_text

    def test_default_kernel_tools_complete(self):
        """默认工具列表包含 MVP 工具集"""
        output = MTPPromptBuilder().build()

        for alias, _ in DEFAULT_KERNEL_TOOLS:
            assert alias in output

    def test_all_optional_modules_disabled(self):
        """全部可选模块关闭时仍有核心内容"""
        output = MTPPromptBuilder(
            include_demo=False,
            include_error_handling=False,
            allowed_kernel_tools=[],
            language="en",
        ).build()

        # 核心模块仍在
        assert "HIVE MEMORY" in output
        assert "PROTOCOL RULES" in output
        assert "CONSTRAINTS" in output
        assert "BEHAVIORAL GUIDELINES" in output

        # 可选模块不在
        assert "[KERNEL TOOLS]" not in output
        assert "ONE-SHOT DEMONSTRATION" not in output
        assert "ERROR RECOVERY" not in output


# ========== get_mtp_prompt 便捷函数测试 ==========

class TestGetMTPPrompt:
    """测试 get_mtp_prompt 便捷函数"""

    def test_returns_string(self):
        """返回非空字符串"""
        result = get_mtp_prompt()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_passes_language_param(self):
        """语言参数正确传递"""
        result_zh = get_mtp_prompt(language="zh")
        result_en = get_mtp_prompt(language="en")

        assert "协议规则" in result_zh
        assert "PROTOCOL RULES" in result_en

    def test_passes_kernel_tools_param(self):
        """工具列表参数正确传递"""
        custom = [("test_tool", "A test tool")]
        result = get_mtp_prompt(kernel_tools=custom)
        assert "test_tool" in result
        assert "sys_clock" not in result


# ========== MTPPromptConfig 配置测试 ==========

class TestMTPPromptConfig:
    """测试 MTPPromptConfig 配置模型"""

    def test_default_values(self):
        """默认配置值正确"""
        config = MTPPromptConfig()
        assert config.enabled is True
        assert config.language == "zh"
        assert config.include_demo is True
        assert config.include_error_handling is True

    def test_nested_in_koakuma(self):
        """MTPPromptConfig 嵌套在 KoakumaConfig 中"""
        config = KoakumaConfig()
        assert hasattr(config, "mtp_prompt")
        assert isinstance(config.mtp_prompt, MTPPromptConfig)
        assert config.mtp_prompt.enabled is True

    def test_custom_values(self):
        """自定义配置值"""
        config = MTPPromptConfig(
            enabled=False,
            language="en",
            include_demo=False,
        )
        assert config.enabled is False
        assert config.language == "en"
        assert config.include_demo is False

    def test_koakuma_config_with_mtp_prompt(self):
        """KoakumaConfig 接受嵌套的 mtp_prompt 配置"""
        config = KoakumaConfig(
            mtp_prompt=MTPPromptConfig(language="en")
        )
        assert config.mtp_prompt.language == "en"
