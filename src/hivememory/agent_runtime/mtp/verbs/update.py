"""UPDATE 指令的语义处理。"""

from __future__ import annotations

import logging

from hivememory.agent_runtime.mtp.context import MTPExecutionContext, resolve_context_language
from hivememory.core.errors import PendingUpdateNotAllowedError, ResourceNotFoundError
from hivememory.core.mtp import (
    MTPCommand,
    MTPResponse,
    MTPResponseStatus,
)
from hivememory.core.mtp.exceptions import (
    AliasNotFoundError,
    InvalidArgumentError,
)
from hivememory.i18n.mtp_runtime import get_mtp_info_text
from hivememory.workspace.contracts import (
    SubmitUpdateIntentRequest,
)

logger = logging.getLogger(__name__)


async def handle_update(
    command: MTPCommand,
    context: MTPExecutionContext,
) -> MTPResponse:
    """
    处理 UPDATE 指令

    将 UPDATE 意图交给 workspace 验证、登记并实时派发；
    生成队列接纳后返回 ACK，正式记忆由 Patchouli 异步更新。

    Type B 动作类响应

    Args:
        command: UPDATE 指令 (target=alias, args: instruction="...", content=`...`)

    Returns:
        MTPResponse: ACK 确认或 ERROR
    """
    alias = command.target.single_alias
    if alias is None:
        raise InvalidArgumentError(message_key="mtp.update.missing_single_target")

    instruction = command.args.get("instruction", "")
    if not instruction:
        raise InvalidArgumentError(message_key="mtp.update.missing_instruction")

    # 能力层在操作授权后解析可信基线，并负责登记成功后的缓存失效。
    try:
        pending = await context.submit_operation(
            SubmitUpdateIntentRequest(
                base_alias=alias,
                instruction=instruction,
                content=command.args.get("content"),
            )
        )
    except PendingUpdateNotAllowedError as error:
        raise InvalidArgumentError(
            message_key="mtp.update.pending_not_updatable",
            params={"alias": alias},
        ) from error
    except ResourceNotFoundError as error:
        # 能力层把不可见与不存在合并为 not_found，UPDATE 不区分两者。
        raise AliasNotFoundError(
            message_key="mtp.update.alias_not_found",
            params={"alias": alias},
        ) from error

    logger.info(
        f"MTP UPDATE admitted: alias='{alias}', " f"pending_alias='{pending.pending_alias}'"
    )

    return MTPResponse(
        status=MTPResponseStatus.ACK,
        content=get_mtp_info_text(
            "mtp.update.ack",
            {"base_alias": alias, "pending_alias": pending.pending_alias},
            resolve_context_language(context),
        ),
        pending_alias=pending.pending_alias,
    )
