import sys
from types import SimpleNamespace

import pytest

from hivememory.agent_runtime.mtp.syscalls.web_search import sys_web_search
from hivememory.core.mtp.exceptions import (
    SyscallInvalidArgumentError,
    SyscallUnavailableError,
)


class TestSysWebSearch:
    """sys_web_search 函数直接测试。"""

    def test_missing_query(self):
        with pytest.raises(SyscallInvalidArgumentError) as exc_info:
            sys_web_search({})

        assert exc_info.value.message_key == "syscall.web_search.missing_query"

    def test_empty_query(self):
        with pytest.raises(SyscallInvalidArgumentError):
            sys_web_search({"query": ""})

    def test_normal_search(self, monkeypatch):
        """正常搜索使用 fake DDGS，避免真实网络调用。"""

        class FakeDDGS:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def text(self, query, max_results):
                return [
                    {
                        "title": "Result 1",
                        "body": "Snippet 1",
                        "href": "https://example.com/1",
                    }
                ]

        monkeypatch.setitem(
            sys.modules,
            "duckduckgo_search",
            SimpleNamespace(DDGS=FakeDDGS),
        )

        result = sys_web_search({"query": "python async"})

        assert "Result 1" in result.content
        assert "摘要：Snippet 1" in result.content

    def test_missing_result_fields_use_i18n_placeholder(self, monkeypatch):
        """搜索结果缺字段时使用 syscall info 文本兜底。"""

        class FakeDDGS:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def text(self, query, max_results):
                return [{}]

        monkeypatch.setitem(
            sys.modules,
            "duckduckgo_search",
            SimpleNamespace(DDGS=FakeDDGS),
        )

        result = sys_web_search({"query": "python async"})

        assert "标题：无" in result.content
        assert "摘要：无" in result.content
        assert "URL：无" in result.content

    def test_num_parameter_non_numeric(self, monkeypatch):
        """num 非数字时回退为 3。"""

        class FakeDDGS:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def text(self, query, max_results):
                assert max_results == 3
                return []

        monkeypatch.setitem(
            sys.modules,
            "duckduckgo_search",
            SimpleNamespace(DDGS=FakeDDGS),
        )

        result = sys_web_search({"query": "test", "num": "abc"})

        assert "未找到与 query 'test' 相关的结果。" in result.content

    def test_search_unavailable(self, monkeypatch):
        # 模拟依赖缺失，确认 handler 抛出结构化 unavailable 异常。
        monkeypatch.delitem(sys.modules, "duckduckgo_search", raising=False)

        with pytest.raises(SyscallUnavailableError) as exc_info:
            sys_web_search({"query": "test"})

        assert exc_info.value.message_key == "syscall.web_search.unavailable"
