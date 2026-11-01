"""响应中的密钥脱敏：HTTP 接口对本机任何进程开放，返回密钥的接口只返回前后片段。"""


def mask_api_key(api_key: str | None) -> str | None:
    """
    对 API 密钥做脱敏处理，避免通过 API 泄露明文密钥。

    规则：
    - None → None（未设置）
    - 长度 ≤ 8 → "***"
    - 长度 > 8 → 保留前 3 位 + "..." + 后 4 位，如 "sk-...abcd"
    """
    if api_key is None:
        return None
    if len(api_key) <= 8:
        return "***"
    return f"{api_key[:3]}...{api_key[-4:]}"
