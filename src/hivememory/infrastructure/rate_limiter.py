"""
Rate Limiter - Token bucket algorithm for log broadcasting rate limiting

用途：防止高频日志淹没 WebSocket 客户端
算法：Token Bucket - 每秒补充 tokens，每次操作消耗 1 token
"""

import time


class RateLimiter:
    """
    Token bucket rate limiter

    使用场景：限制 WebSocket 日志广播速率，防止客户端被淹没

    Example:
        >>> limiter = RateLimiter(max_rate=100)  # 100 logs/second
        >>> if limiter.allow():
        ...     send_log_to_client(log)
    """

    def __init__(self, max_rate: int):
        """
        初始化 rate limiter

        Args:
            max_rate: 最大速率（操作数/秒）
        """
        self.max_rate = max_rate
        self.tokens = float(max_rate)
        self.last_update = time.time()

    def allow(self) -> bool:
        """
        检查是否允许操作（消耗 1 token）

        Returns:
            True if allowed, False if rate limit exceeded
        """
        now = time.time()
        elapsed = now - self.last_update

        # 补充 tokens（每秒补充 max_rate 个）
        self.tokens = min(self.max_rate, self.tokens + elapsed * self.max_rate)
        self.last_update = now

        # 尝试消耗 1 token
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True

        return False

    def reset(self) -> None:
        """重置 rate limiter"""
        self.tokens = float(self.max_rate)
        self.last_update = time.time()


__all__ = ["RateLimiter"]
