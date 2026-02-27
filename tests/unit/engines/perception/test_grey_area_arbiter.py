"""
GreyAreaArbiter 单元测试

测试覆盖:
- NoOpArbiter 基本功能
- RerankerArbiter（使用 mock）
- SLMArbiter（使用 mock）
- 接口定义
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from hivememory.engines.perception.interfaces import BaseArbiter
from hivememory.patchouli.config import (
    ArbiterConfig,
    RerankerArbiterConfig,
    SLMArbiterConfig,
)
from hivememory.engines.perception.grey_area_arbiter import (
    RerankerArbiter,
    SLMArbiter,
    NoOpArbiter,
    DEFAULT_ARBITER_PROMPT,
    create_arbiter,
)


class TestNoOpArbiter:
    """测试 NoOpArbiter"""

    def test_default_action_true(self):
        """测试默认动作为 True"""
        arbiter = NoOpArbiter(default_action=True)

        result = arbiter.should_continue_topic(
            previous_context="写代码",
            current_query="部署服务器",
            similarity_score=0.55
        )

        assert result is True

    def test_default_action_false(self):
        """测试默认动作为 False"""
        arbiter = NoOpArbiter(default_action=False)

        result = arbiter.should_continue_topic(
            previous_context="写代码",
            current_query="做菜",
            similarity_score=0.3
        )

        assert result is False


class TestRerankerArbiter:
    """测试 RerankerArbiter"""

    def setup_method(self):
        """每个测试前的设置"""
        self.mock_reranker = Mock()
        self.mock_reranker.is_loaded.return_value = True
        self.config = RerankerArbiterConfig(threshold=0.5)
        self.arbiter = RerankerArbiter(
            config=self.config,
            reranker_service=self.mock_reranker,
        )

    def test_should_continue_above_threshold(self):
        """测试分数高于阈值时继续"""
        self.mock_reranker.compute_score.return_value = [0.8]

        result = self.arbiter.should_continue_topic(
            previous_context="写代码",
            current_query="部署服务器",
            similarity_score=0.55
        )

        assert result is True
        self.mock_reranker.compute_score.assert_called_once_with(
            pairs=[["写代码", "部署服务器"]],
            batch_size=1,
        )

    def test_should_continue_below_threshold(self):
        """测试分数低于阈值时切分"""
        self.mock_reranker.compute_score.return_value = [0.3]

        result = self.arbiter.should_continue_topic(
            previous_context="写代码",
            current_query="做菜",
            similarity_score=0.55
        )

        assert result is False

    def test_should_continue_empty_context(self):
        """测试空上下文时默认继续"""
        result = self.arbiter.should_continue_topic(
            previous_context="",
            current_query="部署服务器",
            similarity_score=0.55
        )

        assert result is True
        self.mock_reranker.compute_score.assert_not_called()

    def test_should_continue_empty_query(self):
        """测试空查询时默认继续"""
        result = self.arbiter.should_continue_topic(
            previous_context="写代码",
            current_query="",
            similarity_score=0.55
        )

        assert result is True
        self.mock_reranker.compute_score.assert_not_called()

    def test_should_continue_reranker_unavailable(self):
        """测试 Reranker 不可用时默认继续"""
        self.mock_reranker.is_loaded.return_value = False

        result = self.arbiter.should_continue_topic(
            previous_context="写代码",
            current_query="部署服务器",
            similarity_score=0.55
        )

        assert result is True

    def test_should_continue_reranker_error(self):
        """测试 Reranker 出错时默认继续"""
        self.mock_reranker.compute_score.side_effect = Exception("Reranker error")

        result = self.arbiter.should_continue_topic(
            previous_context="写代码",
            current_query="部署服务器",
            similarity_score=0.55
        )

        assert result is True

    def test_should_continue_empty_scores(self):
        """测试 Reranker 返回空列表时默认继续"""
        self.mock_reranker.compute_score.return_value = []

        result = self.arbiter.should_continue_topic(
            previous_context="写代码",
            current_query="部署服务器",
            similarity_score=0.55
        )

        assert result is True


class TestSLMArbiter:
    """测试 SLMArbiter"""

    def setup_method(self):
        """每个测试前的设置"""
        # 使用一个 Mock 对象来模拟 LLM
        self.llm_response = "YES"  # 默认返回 YES

        self.mock_llm = Mock()
        self.mock_llm.complete.side_effect = lambda messages: self.llm_response

        self.config = SLMArbiterConfig()
        self.arbiter = SLMArbiter(
            config=self.config,
            llm_service=self.mock_llm,
        )

    def test_should_continue_yes_response(self):
        """测试 LLM 返回 YES 时继续"""
        self.llm_response = "YES"

        result = self.arbiter.should_continue_topic(
            previous_context="写代码",
            current_query="部署服务器",
            similarity_score=0.55
        )

        assert result is True

    def test_should_continue_no_response(self):
        """测试 LLM 返回 NO 时切分"""
        self.llm_response = "NO"

        result = self.arbiter.should_continue_topic(
            previous_context="写代码",
            current_query="做菜",
            similarity_score=0.55
        )

        assert result is False

    def test_should_continue_case_insensitive(self):
        """测试大小写不敏感"""
        self.llm_response = "yes"

        result = self.arbiter.should_continue_topic(
            previous_context="写代码",
            current_query="部署服务器",
            similarity_score=0.55
        )

        assert result is True

    def test_should_continue_whitespace_handling(self):
        """测试空白字符处理"""
        self.llm_response = "  YES  "

        result = self.arbiter.should_continue_topic(
            previous_context="写代码",
            current_query="部署服务器",
            similarity_score=0.55
        )

        assert result is True

    def test_should_continue_empty_context(self):
        """测试空上下文时默认继续"""
        result = self.arbiter.should_continue_topic(
            previous_context="",
            current_query="部署服务器",
            similarity_score=0.55
        )

        assert result is True

    def test_should_continue_llm_none(self):
        """测试 LLM 为 None 时默认继续"""
        arbiter = SLMArbiter(config=SLMArbiterConfig(), llm_service=None)

        result = arbiter.should_continue_topic(
            previous_context="写代码",
            current_query="部署服务器",
            similarity_score=0.55
        )

        assert result is True

    def test_should_continue_llm_error(self):
        """测试 LLM 调用出错时默认继续"""
        # 创建一个会抛出异常的 Mock 对象
        error_llm = Mock()
        error_llm.complete.side_effect = Exception("LLM error")

        arbiter = SLMArbiter(config=SLMArbiterConfig(), llm_service=error_llm)

        result = arbiter.should_continue_topic(
            previous_context="写代码",
            current_query="部署服务器",
            similarity_score=0.55
        )

        assert result is True

    def test_should_continue_unparseable_response(self):
        """测试无法解析的响应时默认继续"""
        self.llm_response = "I don't know"

        result = self.arbiter.should_continue_topic(
            previous_context="写代码",
            current_query="部署服务器",
            similarity_score=0.55
        )

        assert result is True


class TestDEFAULT_ARBITER_PROMPT:
    """测试默认仲裁提示词"""

    def test_prompt_contains_placeholders(self):
        """测试提示词包含占位符"""
        assert "{previous_context}" in DEFAULT_ARBITER_PROMPT
        assert "{current_query}" in DEFAULT_ARBITER_PROMPT

    def test_prompt_format(self):
        """测试提示词格式化"""
        formatted = DEFAULT_ARBITER_PROMPT.format(
            previous_context="写代码",
            current_query="部署服务器"
        )

        assert "写代码" in formatted
        assert "部署服务器" in formatted
        assert "YES" in formatted
        assert "NO" in formatted


class TestGreyAreaArbiterInterface:
    """测试 BaseArbiter 抽象接口"""

    def test_cannot_instantiate_abstract(self):
        """测试不能直接实例化抽象类"""
        with pytest.raises(TypeError):
            BaseArbiter()

    def test_concrete_implementation(self):
        """测试具体实现可以实例化"""
        # NoOpArbiter 是一个具体实现
        arbiter = NoOpArbiter()
        assert isinstance(arbiter, BaseArbiter)

        # RerankerArbiter 需要参数
        mock_reranker = Mock()
        mock_reranker.is_loaded.return_value = True
        config = RerankerArbiterConfig(threshold=0.5)
        arbiter = RerankerArbiter(config=config, reranker_service=mock_reranker)
        assert isinstance(arbiter, BaseArbiter)


class TestCreateArbiter:
    """测试工厂函数 create_arbiter"""

    def test_disabled(self):
        """测试禁用时返回 NoOpArbiter"""
        config = ArbiterConfig(enabled=False)
        arbiter = create_arbiter(config)
        assert isinstance(arbiter, NoOpArbiter)
        assert arbiter.default_action is True

    def test_reranker_impl(self):
        """测试创建 RerankerArbiter"""
        config = ArbiterConfig(
            enabled=True,
            engine=RerankerArbiterConfig()
        )
        mock_reranker = Mock()
        arbiter = create_arbiter(config, reranker_service=mock_reranker)
        assert isinstance(arbiter, RerankerArbiter)

    def test_slm_impl(self):
        """测试创建 SLMArbiter"""
        config = ArbiterConfig(
            enabled=True,
            engine=SLMArbiterConfig()
        )
        mock_llm = Mock()
        arbiter = create_arbiter(config, llm_service=mock_llm)
        assert isinstance(arbiter, SLMArbiter)

    def test_reranker_missing_service(self):
        """测试缺少 Reranker 服务时回退"""
        config = ArbiterConfig(
            enabled=True,
            engine=RerankerArbiterConfig()
        )
        arbiter = create_arbiter(config, reranker_service=None)
        assert isinstance(arbiter, NoOpArbiter)

    def test_slm_missing_service(self):
        """测试缺少 LLM 服务时回退"""
        config = ArbiterConfig(
            enabled=True,
            engine=SLMArbiterConfig()
        )
        arbiter = create_arbiter(config, llm_service=None)
        assert isinstance(arbiter, NoOpArbiter)
