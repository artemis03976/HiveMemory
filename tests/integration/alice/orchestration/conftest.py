"""CALL Profile 测试复用真实 Patchouli 管理与 workspace 失效事件链。"""

from tests.integration.workspace.test_intent_registry_and_read_cache import (
    chain as profile_chain,  # noqa: F401 -- pytest 按 fixture 名称注入。
)
