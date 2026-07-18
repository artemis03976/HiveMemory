"""Gateway Entry 能力接口。"""

from abc import ABC, abstractmethod

from hivememory.engines.gateway.models import InterceptorResult


class BaseInterceptor(ABC):
    """低成本 Entry Interceptor 抽象接口。"""

    @abstractmethod
    def intercept(
        self,
        query: str,
        *,
        allow_system: bool = True,
    ) -> InterceptorResult | None:
        """匹配系统指令或 simple chat，未命中时返回 ``None``。"""

        raise NotImplementedError


__all__ = ["BaseInterceptor"]
