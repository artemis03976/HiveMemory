"""
Mock LLM 服务

用于集成测试中替代真实的 LLM 调用。
"""

from typing import List, Optional, Any
from pydantic import BaseModel


class MockLLMResponse(BaseModel):
    """Mock LLM 响应"""
    content: str
    model: str = "mock-model"
    tokens_used: int = 0


class MockLLMService:
    """
    Mock LLM 服务，用于测试组件协作。

    不调用真实 API，返回预设的响应或基于规则的响应。
    """

    def __init__(
        self,
        default_response: str = "This is a mock response.",
        raise_on_call: bool = False,
    ):
        """
        初始化 Mock LLM 服务。

        Args:
            default_response: 默认响应内容
            raise_on_call: 是否在调用时抛出异常（用于测试错误处理）
        """
        self.default_response = default_response
        self.raise_on_call = raise_on_call
        self.call_count = 0
        self.call_history: List[dict] = []

    def complete(
        self,
        messages: List[dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> MockLLMResponse:
        """
        Mock completion 调用。

        Args:
            messages: 对话消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            **kwargs: 其他参数

        Returns:
            MockLLMResponse: Mock 响应
        """
        self.call_count += 1
        self.call_history.append({
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })

        if self.raise_on_call:
            raise RuntimeError("Mock LLM service configured to raise error")

        # 根据最后一条消息内容返回不同响应
        if messages:
            last_msg = messages[-1].get("content", "")
            if "提取" in last_msg or "extract" in last_msg.lower():
                return MockLLMResponse(
                    content='{"memories": [{"title": "Mock Memory", "summary": "Test summary", "tags": ["test"], "content": "Test content"}]}',
                    model=model or "mock-model",
                    tokens_used=100,
                )
            elif "分析" in last_msg or "analyze" in last_msg.lower():
                return MockLLMResponse(
                    content='{"intent": "RAG", "keywords": ["test"], "worth_saving": true}',
                    model=model or "mock-model",
                    tokens_used=50,
                )

        return MockLLMResponse(
            content=self.default_response,
            model=model or "mock-model",
            tokens_used=50,
        )

    async def acomplete(
        self,
        messages: List[dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> MockLLMResponse:
        """异步版本"""
        return self.complete(messages, model, temperature, max_tokens, **kwargs)

    def get_last_call(self) -> Optional[dict]:
        """获取最后一次调用记录"""
        return self.call_history[-1] if self.call_history else None

    def reset(self):
        """重置调用记录"""
        self.call_count = 0
        self.call_history.clear()


class MockEmbeddingService:
    """
    Mock Embedding 服务，用于测试。

    返回固定维度的随机向量。
    """

    def __init__(self, dimension: int = 1024):
        """
        初始化 Mock Embedding 服务。

        Args:
            dimension: 向量维度
        """
        self.dimension = dimension
        self.call_count = 0

    def encode(self, text: str, normalize_embeddings: bool = True) -> List[float]:
        """
        Mock 编码调用。

        Args:
            text: 输入文本
            normalize_embeddings: 是否归一化

        Returns:
            List[float]: 嵌入向量
        """
        self.call_count += 1
        # 基于文本生成确定性向量（简单hash）
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()

        # 扩展到指定维度
        vector = []
        for i in range(self.dimension):
            byte_val = hash_bytes[i % len(hash_bytes)]
            # 归一化到 -1 到 1
            val = (byte_val - 128) / 128.0
            vector.append(val)

        if normalize_embeddings:
            # 简单归一化
            norm = sum(v * v for v in vector) ** 0.5
            if norm > 0:
                vector = [v / norm for v in vector]

        return vector

    def reset(self):
        """重置调用记录"""
        self.call_count = 0
