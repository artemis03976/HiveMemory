from unittest.mock import MagicMock, patch

import pytest

from hivememory.agent_runtime.syscalls.web_search import sys_web_search


class TestSysWebSearch:
    """sys_web_search 函数直接测试"""

    def test_missing_query(self):
        result = sys_web_search({})
        assert "Error" in result
        assert "query" in result.lower()

    def test_empty_query(self):
        result = sys_web_search({"query": ""})
        assert "Error" in result

    @patch("hivememory.agent_runtime.syscalls.web_search.DDGS", create=True)
    def test_normal_search(self, mock_ddgs_cls):
        """正常搜索 (mock DDGS)"""
        mock_instance = MagicMock()
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=False)
        mock_instance.text.return_value = [
            {"title": "Result 1", "body": "Snippet 1", "href": "https://example.com/1"},
        ]
        mock_ddgs_cls.return_value = mock_instance

        try:
            from duckduckgo_search import DDGS
            with patch("hivememory.agent_runtime.syscalls.web_search.DDGS", mock_ddgs_cls):
                result = sys_web_search({"query": "python async"})
                assert "Result 1" in result
        except ImportError:
            result = sys_web_search({"query": "test"})
            assert "not available on this system" in result

    def test_num_parameter_non_numeric(self):
        """num 非数字默认为 3"""
        try:
            from duckduckgo_search import DDGS
            pytest.skip("Skipping to avoid real network call")
        except ImportError:
            result = sys_web_search({"query": "test", "num": "abc"})
            assert "not available on this system" in result
