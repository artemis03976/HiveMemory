"""WRITE 指令的语义处理。"""

from __future__ import annotations

import logging

from hivememory.agent_runtime.mtp.context import MTPExecutionContext, resolve_context_language
from hivememory.core.models import WriteFocus
from hivememory.core.mtp import (
    MTPCommand,
    MTPResponse,
    MTPResponseStatus,
)
from hivememory.core.mtp.exceptions import (
    InvalidArgumentError,
)
from hivememory.i18n.mtp_runtime import get_mtp_info_text
from hivememory.workspace.contracts import (
    SubmitWriteIntentRequest,
)

logger = logging.getLogger(__name__)


async def handle_write(
    command: MTPCommand,
    context: MTPExecutionContext,
) -> MTPResponse:
    """
    处理 WRITE 指令

    将 WRITE 内容打包为操作请求，经 workspace 登记并实时派发；
    生成队列接纳后返回 ACK，正式记忆由 Patchouli 异步生成。

    Type B 动作类响应

    Args:
        command: WRITE 指令 (target=*, args: content=`...`, reason="...", title="...")

    Returns:
        MTPResponse: ACK 确认
    """
    content = command.args.get("content", "")
    if not content:
        raise InvalidArgumentError(message_key="mtp.write.missing_content")

    reason = command.args.get("reason", "")
    title = command.args.get("title", "")

    # 注册 pending atom 并生成 pending alias
    pending = await context.submit_operation(
        SubmitWriteIntentRequest(
            focus=WriteFocus(content=content, title=title or None, reason=reason or None)
        )
    )

    logger.info(
        f"MTP WRITE admitted: content='{content[:50]}...', "
        f"pending_alias='{pending.pending_alias}'"
    )

    return MTPResponse(
        status=MTPResponseStatus.ACK,
        content=get_mtp_info_text(
            "mtp.write.ack",
            {"pending_alias": pending.pending_alias},
            resolve_context_language(context),
        ),
        pending_alias=pending.pending_alias,
    )
