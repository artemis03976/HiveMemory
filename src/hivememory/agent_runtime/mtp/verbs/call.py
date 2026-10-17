"""CALL 指令的语义处理。"""

from __future__ import annotations

import json
import logging

from hivememory.agent_runtime.mtp.context import MTPExecutionContext
from hivememory.core.mtp import (
    MTPCallRequest,
    MTPCommand,
    MTPResponse,
    MTPResponseStatus,
)
from hivememory.core.mtp.exceptions import (
    InvalidArgumentError,
)

logger = logging.getLogger(__name__)


async def handle_call(
    command: MTPCommand,
    context: MTPExecutionContext,
) -> MTPResponse:
    """
    处理 CALL 指令 - 触发子代理调用

    此处理器只验证参数，执行许可由分派入口统一检查，实际调用由编排层处理。
    返回特殊的 SUSPEND 状态，通知运行时交还 frame 控制权。

    Args:
        command: CALL 指令 (target=agent_alias, args: task="...", context_refs=["..."])

    Returns:
        MTPResponse: SUSPEND 状态，携带 CALL 参数
    """
    del context

    # 在 frame policy 检查之后校验 target 与 task。
    target_alias = command.target.single_alias
    if not target_alias:
        raise InvalidArgumentError(message_key="mtp.call.missing_single_target")

    task = command.args.get("task", "")
    if not task:
        raise InvalidArgumentError(message_key="mtp.call.missing_task")

    # 解析 context_refs (选填)
    context_refs_str = command.args.get("context_refs", "")
    context_refs = []
    if context_refs_str:
        try:
            # context_refs 已被 MTPParser 序列化为 JSON 字符串
            context_refs = json.loads(context_refs_str)
            if not isinstance(context_refs, list):
                context_refs = [context_refs]
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse context_refs: {context_refs_str}")
            # 降级为逗号分隔
            context_refs = [ref.strip() for ref in context_refs_str.split(",") if ref.strip()]

    logger.info(
        f"MTP CALL: target={target_alias}, task='{task[:50]}...', " f"context_refs={context_refs}"
    )

    # 返回 SUSPEND 信号
    return MTPResponse(
        status=MTPResponseStatus.SUSPEND,
        content="",
        call_request=MTPCallRequest(
            target_alias=target_alias,
            task=task,
            context_refs=context_refs,
        ),
    )
