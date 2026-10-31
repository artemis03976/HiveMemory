"""Import Bus 编排器：串行门内完成查重、Gateway 落位、封口与提交接纳。

Import Bus 只向记忆网络输入，不检索记忆、不返回记忆上下文：

    按外部会话串行 -> 推导 interaction_id -> 查询进度 -> 没有输出则跳过
    -> 写入时已落位则沿用落点，否则 Gateway PASSIVE_MEMORY（附最近轮次与
    上一轮的落位）-> 共用封口函数 -> Patchouli 公开提交路由 -> 等待感知应用
    完成 -> 确认

确认表示交互已经应用：应用失败或等待超时都不确认，连接器保留这一轮
稍后重发。等待在串行门内完成，同一外部会话的下一轮路由时，上一轮落位
的话题已经建立，可以成为候选与续接的落点。

写入与更新在交互进行中提交，需要这一轮最终所在的话题。落位在同一串行门
内以与导入相同的输入路由，经 Patchouli prepare（关闭检索）准备话题，新话题
在此预建；落点按 interaction_id 登记，同一轮再次落位直接返回，回流时以落点
话题提交、不再路由。落位只读取候选话题，不读取记忆。

传入的 ``IdentityScope`` 只在这一次导入的调用链内经公共路由到达 Gateway
与 Patchouli，不写入记录、事件或后台任务；提交的归属取目标 Workspace，
发起者取认证通过的 (user, Agent)。编排器不持有提交队列，也不登记维护
任务；关闭时只等待在途导入（含其等待应用的时间），队列由 Patchouli
自己排空。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.components.events.bus import RuntimeEventSink
from hivememory.components.serial_gate import KeyedSerialGate
from hivememory.core.models import IdentityScope
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    GatewayIngressMode,
    GatewaySessionTurn,
)
from hivememory.core.protocol.sealing import seal_interaction
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.patchouli.contracts import PatchouliRoutes, PreparedAgentRun
from hivememory.patchouli.contracts.errors import InteractionAdmissionUnavailableError
from hivememory.patchouli.contracts.interaction_submission import InteractionStatus
from hivememory.system.services.import_bus.errors import (
    ImportBusApplyFailedError,
    ImportBusContractError,
    ImportBusUnavailableError,
    ImportUnavailableReason,
    is_recoverable_gateway_error,
)
from hivememory.system.services.import_bus.events import ImportBusEventEmitter
from hivememory.system.services.import_bus.models import (
    ImportConversationKey,
    ImportedTurn,
    ImportTurnResult,
    InFlightTurn,
    derive_import_interaction_id,
)

logger = logging.getLogger(__name__)

_NEW_TOPIC = "NEW_TOPIC"


class ImportBusIngestor:
    """导入编排：同一外部会话内按到达顺序处理，不同会话互不阻塞。

    门内依次完成进度查询、Gateway、提交与等待应用：同一外部会话的并发请求
    （例如两个 hook 同时补发）因此按到达顺序处理，后到的请求看到前一个的
    结果。已成功应用的轮次直接返回 duplicate；仍在应用的继续等待；两者都
    不再调用 Gateway——Gateway 给出的目标话题与新话题标题写入提交，重新路由
    可能得到不同的提交而被队列判为冲突。只有应用失败的轮次重新路由后再提交。

    写入时的落位与导入共用同一串行门与在途计数：同一外部会话的落位与回流
    按到达顺序处理，关闭时同样等待在途的落位。
    """

    def __init__(
        self,
        bus: GlobalSystemBus,
        *,
        gateway_request_timeout_ms: int,
        gateway_session_turn_limit: int,
        apply_wait_seconds: float,
        runtime_events: RuntimeEventSink | None = None,
    ) -> None:
        self._bus = bus
        self._gateway_request_timeout_ms = gateway_request_timeout_ms
        self._gateway_session_turn_limit = gateway_session_turn_limit
        self._apply_wait_seconds = apply_wait_seconds
        self._events = ImportBusEventEmitter(runtime_events)
        self._serial_gate = KeyedSerialGate[ImportConversationKey]()
        self._closed = False
        self._inflight = 0
        self._idle = asyncio.Event()
        self._idle.set()
        # 写入时为仍在进行的一轮 prepare 的结果，只读取其中的话题与是否新建；
        # 按 interaction_id 保存，回流的提交被接纳后取走，不进入 finalize 或
        # cleanup。按轮次而不是按外部会话保存：下一轮的写入可能早于上一轮的
        # 回流到达。只在进程内、不持久化；从未回流的轮次的落点留到进程重启。
        self._placements: dict[str, PreparedAgentRun] = {}

    async def ingest(
        self,
        turn: ImportedTurn,
        *,
        identity_scope: IdentityScope,
        principal_id: str,
    ) -> ImportTurnResult:
        """导入一轮完整记录，应用完成后返回接纳；已应用的返回重复，没有输出的跳过。

        ``identity_scope`` 与 ``principal_id`` 来自导入授权。Gateway 整体抛出的
        可恢复异常、提交队列拥塞、等待应用超时与关闭中以
        ``ImportBusUnavailableError`` 交回，应用失败以 ``ImportBusApplyFailedError``
        交回，都可用同一轮次重发；契约错误、程序错误与取消原样传播。
        """
        interaction_id = self._interaction_id(turn, identity_scope, principal_id)
        async with self._serialized(turn, identity_scope):
            return await self._ingest_serialized(turn, identity_scope, interaction_id, principal_id)

    async def place(
        self,
        turn: InFlightTurn,
        *,
        identity_scope: IdentityScope,
        principal_id: str,
    ) -> str:
        """为仍在进行的一轮落位，返回落点话题的真实 id；同一轮再次落位返回同一落点。

        第一次落位以与导入相同的输入与续接规则做 PASSIVE_MEMORY 路由，再经
        Patchouli prepare（关闭检索）准备话题，新话题在此预建，然后登记落点。
        这一轮已经回流并应用时（后台运行的子代理可能在这一轮结束、回流之后
        才写入），直接返回它实际落位的话题，不再路由，也不登记落点；已经提交
        但仍在应用时以 ``apply_pending`` 交回，写入方稍后重试。
        Gateway 整体抛出的可恢复异常与关闭中以 ``ImportBusUnavailableError``
        交回；写入正等待这个结果，失败直接交给调用方，不记为导入推迟。prepare
        失败时由 Patchouli 清理它预建的空话题，异常原样传播，不登记落点。
        """
        interaction_id = self._interaction_id(turn, identity_scope, principal_id)
        async with self._serialized(turn, identity_scope):
            placement = self._placements.get(interaction_id)
            if placement is not None:
                return placement.topic_id
            status = await self._bus.request(
                PatchouliRoutes.INTERACTION_STATUS,
                identity_scope=identity_scope,
                interaction_id=interaction_id,
            )
            # 只有已应用的轮次带真实落点。
            if status is not None and status.topic_id is not None:
                return status.topic_id
            if status is not None and status.state == "pending":
                # 这一轮已经提交、等待应用超时，落点要等应用完成才确定；写入方稍后
                # 重试时它已应用。此时重新路由会让写入与这一轮落在不同的话题。
                raise ImportBusUnavailableError(reason="apply_pending")
            decision = await self._request_gateway_decision(turn, identity_scope, principal_id)
            placement = await self._bus.request(
                PatchouliRoutes.PREPARE_AGENT_RUN,
                identity_scope=identity_scope,
                interaction_id=interaction_id,
                gateway_decision=decision,
                enable_memory_retrieval=False,
            )
            self._placements[interaction_id] = placement
            logger.info(
                "Import Bus 为在途轮次落位: source=%s, interaction_id=%s, topic_id=%s, "
                "new_topic=%s",
                turn.source,
                interaction_id,
                placement.topic_id,
                placement.is_new_topic,
            )
            return placement.topic_id

    async def shutdown(self) -> dict[str, Any]:
        """停止接收并等待在途导入结束；不排空提交队列，队列由 Patchouli 排空。

        关闭开始后，门外等待的请求在取得串行门时即以 ``shutting_down`` 拒绝，
        只有每个会话正在处理的请求继续完成；它的等待受 Gateway 请求时限与
        等待应用的时限约束，提交队列此时仍在运行，已提交的轮次因此能得到确认。
        """
        self._closed = True
        waited_requests = self._inflight
        await self._idle.wait()
        return {"waited_requests": waited_requests}

    # ------------------------------------------------------------------
    # 串行门内的编排
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def _serialized(self, turn: InFlightTurn, scope: IdentityScope) -> AsyncIterator[None]:
        """在这一轮所属外部会话的串行门内执行，并计入在途请求；关闭开始后拒绝。"""
        if self._closed:
            # 认证网关先于编排器关闭；关闭前已通过认证的请求在这里收到同一
            # 可重试结论，而不是在已停止的子系统上失败。
            raise ImportBusUnavailableError(reason="shutting_down")
        key = ImportConversationKey.for_turn(scope.actor_identity, turn)
        self._enter()
        try:
            async with self._serial_gate.hold(key):
                if self._closed:
                    # 在门外等待时关闭开始：不再为它调用 Gateway，关闭只等待
                    # 每个会话正在处理的那一个请求。
                    raise ImportBusUnavailableError(reason="shutting_down")
                yield
        finally:
            self._leave()

    @staticmethod
    def _interaction_id(turn: InFlightTurn, scope: IdentityScope, principal_id: str) -> str:
        """落位与导入对同一轮推导出同一个标识，回流因此能找到落点。"""
        return derive_import_interaction_id(
            principal_id=principal_id,
            actor=scope.actor_identity,
            source=turn.source,
            external_conversation_id=turn.external_conversation_id,
            external_turn_id=turn.external_turn_id,
        )

    async def _ingest_serialized(
        self, turn: ImportedTurn, scope: IdentityScope, interaction_id: str, principal_id: str
    ) -> ImportTurnResult:
        status = await self._bus.request(
            PatchouliRoutes.INTERACTION_STATUS,
            identity_scope=scope,
            interaction_id=interaction_id,
        )
        if status is not None and status.state in ("succeeded", "unknown"):
            return self._duplicate(turn, scope, interaction_id, status.state)
        if status is not None and status.state == "pending":
            # 上一次提交已被接纳、仍在应用：继续等待，不再路由与提交。
            settled = await self._wait_applied(turn, scope, interaction_id)
            return self._duplicate(turn, scope, interaction_id, settled.state)

        if status is None and not turn.has_output:
            self._events.skipped(turn=turn, scope=scope, interaction_id=interaction_id)
            return ImportTurnResult(status="skipped", interaction_id=interaction_id)

        # 尚未提交，或上一次提交在应用时失败：（再次）提交。写入时已落位的轮次
        # 沿用落点话题，不再路由；其余轮次重新路由。
        placement = self._placements.get(interaction_id)
        if placement is not None:
            # 落点话题在落位时已经建立，新话题的标题与摘要随之写入。
            topic_id, is_new_topic = placement.topic_id, placement.is_new_topic
            new_topic_title = new_topic_summary = None
        else:
            try:
                decision = await self._request_gateway_decision(turn, scope, principal_id)
            except ImportBusUnavailableError as error:
                self._record_deferred(
                    turn, scope, interaction_id, "gateway_unavailable", error.__cause__
                )
                raise
            topic_id = decision.target_topic_id
            is_new_topic = topic_id == _NEW_TOPIC
            # 新话题以 Gateway 给出的标题与摘要建立，后续落位依赖它们。
            new_topic_title = decision.new_topic_title if is_new_topic else None
            new_topic_summary = decision.new_topic_summary if is_new_topic else None
        payload = seal_interaction(
            turn_events=turn.to_turn_events(),
            model_used=turn.model_used or "",
            used_attachments=(),
            outcome=turn.outcome,
        )
        try:
            await self._bus.request(
                PatchouliRoutes.INTERACTION_SUBMIT,
                identity_scope=scope,
                payload=payload,
                requested_topic_id=topic_id,
                interaction_id=interaction_id,
                origin="import_bus",
                new_topic_title=new_topic_title,
                new_topic_summary=new_topic_summary,
            )
        except InteractionAdmissionUnavailableError as error:
            self._record_deferred(turn, scope, interaction_id, "queue_unavailable", error)
            raise ImportBusUnavailableError(reason="queue_unavailable") from error
        # 提交已被接纳，落点改由提交记录承载：此后的重发按提交进度处理，应用
        # 失败后的重发重新路由。队列拥塞时落点保留，重发仍然沿用。
        self._placements.pop(interaction_id, None)

        applied = await self._wait_applied(turn, scope, interaction_id)
        logger.info(
            "Import Bus 确认轮次: source=%s, interaction_id=%s, new_topic=%s",
            turn.source,
            interaction_id,
            is_new_topic,
        )
        self._events.accepted(
            turn=turn,
            scope=scope,
            interaction_id=interaction_id,
            topic_id=applied.topic_id,
            is_new_topic=is_new_topic,
        )
        return ImportTurnResult(status="accepted", interaction_id=interaction_id)

    async def _wait_applied(
        self, turn: ImportedTurn, scope: IdentityScope, interaction_id: str
    ) -> InteractionStatus:
        """等待已接纳的交互应用完成；超时与应用失败都不确认，交回给连接器重发。"""
        status = await self._bus.request(
            PatchouliRoutes.INTERACTION_WAIT,
            identity_scope=scope,
            interaction_id=interaction_id,
            timeout_seconds=self._apply_wait_seconds,
        )
        if status is None:
            raise ImportBusContractError("已接纳的交互在提交队列中查不到")
        if status.state == "failed":
            logger.warning(
                "Import Bus 轮次应用失败: source=%s, interaction_id=%s",
                turn.source,
                interaction_id,
            )
            self._events.deferred(
                turn=turn,
                scope=scope,
                interaction_id=interaction_id,
                reason="apply_failed",
                error_class=None,
            )
            raise ImportBusApplyFailedError()
        if status.state == "pending":
            self._events.deferred(
                turn=turn,
                scope=scope,
                interaction_id=interaction_id,
                reason="apply_pending",
                error_class=None,
            )
            raise ImportBusUnavailableError(reason="apply_pending")
        return status

    def _duplicate(
        self, turn: ImportedTurn, scope: IdentityScope, interaction_id: str, state: str
    ) -> ImportTurnResult:
        logger.info(
            "Import Bus 忽略已应用的轮次: source=%s, interaction_id=%s, state=%s",
            turn.source,
            interaction_id,
            state,
        )
        self._events.duplicate(turn=turn, scope=scope, interaction_id=interaction_id)
        return ImportTurnResult(status="duplicate", interaction_id=interaction_id)

    async def _request_gateway_decision(
        self, turn: InFlightTurn, scope: IdentityScope, principal_id: str
    ) -> GatewayDecision:
        """以 PASSIVE_MEMORY 模式只取得话题落位与新话题的标题和摘要。

        最近一轮带外部标识时附上它实际落位的话题：话题路由失败时，Gateway
        续接上一轮的话题，而不是新建无标题话题。Gateway 整体抛出的可恢复
        异常转为 ``gateway_unavailable``，原异常作为 cause；是否记为导入推迟
        由调用方决定。
        """
        recent_turns = turn.recent_turns[-self._gateway_session_turn_limit :]
        previous_topic_id = (
            await self._previous_topic_id(turn, scope, principal_id) if recent_turns else None
        )
        session_turns = tuple(
            GatewaySessionTurn(
                user_message=recent.user_message,
                assistant_text=recent.assistant_text,
                topic_id=previous_topic_id if index == len(recent_turns) - 1 else None,
            )
            for index, recent in enumerate(recent_turns)
        )
        try:
            result = await self._bus.request(
                GatewayPublicRoutes.PROCESS,
                message=turn.user_message,
                identity_scope=scope,
                ingress_mode=GatewayIngressMode.PASSIVE_MEMORY,
                request_timeout_ms=self._gateway_request_timeout_ms,
                session_turns=session_turns,
            )
        except Exception as error:
            if not is_recoverable_gateway_error(error):
                raise
            raise ImportBusUnavailableError(reason="gateway_unavailable") from error
        if result.kind != "decision":
            raise ImportBusContractError("PASSIVE_MEMORY 不得返回命令结果")
        return result.decision

    async def _previous_topic_id(
        self, turn: InFlightTurn, scope: IdentityScope, principal_id: str
    ) -> str | None:
        """查出最近一轮的落位：写入时已落位、尚未回流的取落点，否则取它应用后的话题。

        该轮未带外部标识、既没有落点也未导入或尚未应用成功时为空。连接器的
        回流可能晚于下一轮的写入到达，此时上一轮只有落点。
        """
        previous = turn.recent_turns[-1]
        if previous.external_turn_id is None:
            return None
        previous_interaction_id = derive_import_interaction_id(
            principal_id=principal_id,
            actor=scope.actor_identity,
            source=turn.source,
            external_conversation_id=turn.external_conversation_id,
            external_turn_id=previous.external_turn_id,
        )
        placement = self._placements.get(previous_interaction_id)
        if placement is not None:
            return placement.topic_id
        status = await self._bus.request(
            PatchouliRoutes.INTERACTION_STATUS,
            identity_scope=scope,
            interaction_id=previous_interaction_id,
        )
        return status.topic_id if status is not None else None

    def _record_deferred(
        self,
        turn: ImportedTurn,
        scope: IdentityScope,
        interaction_id: str,
        reason: ImportUnavailableReason,
        cause: BaseException | None,
    ) -> None:
        """记录交回连接器重发的可重试失败；只记录引发它的异常类型名。"""
        error_class = None if cause is None else type(cause).__name__
        logger.warning(
            "Import Bus 推迟轮次: source=%s, interaction_id=%s, reason=%s, error=%s",
            turn.source,
            interaction_id,
            reason,
            error_class,
        )
        self._events.deferred(
            turn=turn,
            scope=scope,
            interaction_id=interaction_id,
            reason=reason,
            error_class=error_class,
        )

    # ------------------------------------------------------------------
    # 在途导入计数
    # ------------------------------------------------------------------

    def _enter(self) -> None:
        self._inflight += 1
        self._idle.clear()

    def _leave(self) -> None:
        self._inflight -= 1
        if self._inflight == 0:
            self._idle.set()


__all__ = ["ImportBusIngestor"]
