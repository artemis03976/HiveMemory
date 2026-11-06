"""Import Bus 编排器：串行门内完成查重、Gateway 路由、封口与提交接纳。

Import Bus 只向记忆网络输入，不检索记忆、不返回记忆上下文：

    按外部会话串行 -> 推导 interaction_id -> 查询进度 -> 没有输出则跳过
    -> 写入时已确定话题则沿用该话题，否则 Gateway PASSIVE_MEMORY（附最近轮次与
    上一轮所在的话题）-> 共用封口函数 -> root API 提交交互 -> 等待感知应用
    完成 -> 确认

确认表示交互已经应用：应用失败或等待超时都不确认，连接器保留这一轮
稍后重发。等待在串行门内完成，同一外部会话的下一轮路由时，上一轮所在
的话题已经建立，可以成为候选话题与续接目标。

写入与更新在交互进行中提交，需要这一轮最终所在的话题。为这一轮确定话题时，
在同一串行门内以与导入相同的输入路由，经 Patchouli prepare（关闭检索）准备
话题，新话题在此预建；已确定的话题按 interaction_id 登记，同一轮再次请求直接
返回，回流时以该话题提交、不再路由。确定话题只读取候选话题，不读取记忆。

查询进度、路由、准备话题、提交与等待应用都是系统级操作：编排器以 Import Bus
服务认证得到的系统 root 工作 actor 调用 root API，每次调用由 root API 在方法
入口授权，编排器自己不授权、也不持有 scope。目标不是参与者驻留的 Workspace
时，第一次 root 调用即被拒绝，先于路由与提交。交互中的 Agent 与认证通过的
principal 标识作为数据显式传入：外部会话键、交互标识与事件中的 Agent 取自
它们，提交的发起者是参与者。
编排器不持有提交队列，也不登记维护任务；关闭时只等待在途导入（含其等待
应用的时间），队列由 Patchouli 自己排空。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from hivememory.components.events.bus import RuntimeEventSink
from hivememory.components.serial_gate import KeyedSerialGate
from hivememory.core.access import WorkingActor
from hivememory.core.models import ActorIdentity, WorkspaceIdentity
from hivememory.core.protocol.gateway import GatewayDecision, GatewaySessionTurn
from hivememory.core.protocol.sealing import seal_interaction
from hivememory.patchouli.contracts import PreparedAgentRun
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
from hivememory.workspace.capability.root.interaction import RootInteractionService
from hivememory.workspace.capability.root.topic import RootTopicService

logger = logging.getLogger(__name__)

_NEW_TOPIC = "NEW_TOPIC"


class ImportBusIngestor:
    """导入编排：同一外部会话内按到达顺序处理，不同会话互不阻塞。

    门内依次完成进度查询、Gateway、提交与等待应用：同一外部会话的并发请求
    （例如两个 hook 同时补发）因此按到达顺序处理，后到的请求看到前一个的
    结果。已成功应用的轮次直接返回 duplicate；仍在应用的继续等待；两者都
    不再调用 Gateway——Gateway 给出的目标话题与新话题标题写入提交，重新路由
    可能得到不同的提交而被队列判为冲突。只有应用失败的轮次重新路由后再提交。

    写入时确定话题与导入共用同一串行门与在途计数：同一外部会话的确定话题请求
    与回流按到达顺序处理，关闭时同样等待在途的确定话题请求。``interactions``
    与 ``topics`` 是组合根注入的 root 交互 API 与 root 话题 API。
    """

    def __init__(
        self,
        *,
        interactions: RootInteractionService,
        topics: RootTopicService,
        gateway_request_timeout_ms: int,
        gateway_session_turn_limit: int,
        apply_wait_seconds: float,
        runtime_events: RuntimeEventSink | None = None,
    ) -> None:
        self._interactions = interactions
        self._topics = topics
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
        # 回流到达。只在进程内、不持久化；从未回流的轮次的登记留到进程重启。
        self._placements: dict[str, PreparedAgentRun] = {}

    async def ingest(
        self,
        turn: ImportedTurn,
        *,
        system_actor: WorkingActor,
        target_workspace: WorkspaceIdentity,
        participant: ActorIdentity,
        principal_id: str,
    ) -> ImportTurnResult:
        """导入一轮完整记录，应用完成后返回接纳；已应用的返回重复，没有输出的跳过。

        ``system_actor`` 是系统的 root 工作 actor，以它调用 root API；
        ``target_workspace`` 是这次导入的目标。``participant`` 与
        ``principal_id`` 是请求认证通过的 Agent 与调用来源。
        Gateway 整体抛出的可恢复异常、提交队列拥塞、等待应用超时与关闭中以
        ``ImportBusUnavailableError`` 交回，应用失败以 ``ImportBusApplyFailedError``
        交回，都可用同一轮次重发；契约错误、程序错误与取消原样传播。
        """
        interaction_id = self._interaction_id(turn, participant, principal_id)
        async with self._serialized(turn, participant):
            return await self._ingest_serialized(
                turn,
                system_actor=system_actor,
                target_workspace=target_workspace,
                participant=participant,
                principal_id=principal_id,
                interaction_id=interaction_id,
            )

    async def place(
        self,
        turn: InFlightTurn,
        *,
        system_actor: WorkingActor,
        target_workspace: WorkspaceIdentity,
        participant: ActorIdentity,
        principal_id: str,
    ) -> str:
        """为仍在进行的一轮确定话题，返回该话题的真实 id；同一轮再次请求返回同一话题。

        第一次请求以与导入相同的输入与续接规则做 PASSIVE_MEMORY 路由，再经
        Patchouli prepare（关闭检索）准备话题，新话题在此预建，然后登记已确定的话题。
        这一轮已经回流并应用时（后台运行的子代理可能在这一轮结束、回流之后
        才写入），直接返回它实际所在的话题，不再路由，也不登记；已经提交
        但仍在应用时以 ``apply_pending`` 交回，写入方稍后重试。
        Gateway 整体抛出的可恢复异常与关闭中以 ``ImportBusUnavailableError``
        交回；写入正等待这个结果，失败直接交给调用方，不记为导入推迟。prepare
        失败时由 Patchouli 清理它预建的空话题，异常原样传播，不登记。确定话题时
        不检索、不读取记忆；参数的含义与导入相同。

        已确定话题的登记命中时直接返回，不经过 root API：登记项只由此前经过 root
        授权的请求写入，键由同一 principal 与参与者推导。
        """
        interaction_id = self._interaction_id(turn, participant, principal_id)
        async with self._serialized(turn, participant):
            placement = self._placements.get(interaction_id)
            if placement is not None:
                return placement.topic_id
            status = await self._interactions.interaction_status(
                target_workspace=target_workspace,
                interaction_id=interaction_id,
                actor=system_actor,
            )
            # 只有已应用的轮次带实际所在的话题。
            if status is not None and status.topic_id is not None:
                return status.topic_id
            if status is not None and status.state == "pending":
                # 这一轮已经提交、等待应用超时，所在的话题要等应用完成才确定；写入方稍后
                # 重试时它已应用。此时重新路由会让写入与这一轮落在不同的话题。
                raise ImportBusUnavailableError(reason="apply_pending")
            decision = await self._request_gateway_decision(
                turn, system_actor, target_workspace, participant, principal_id
            )
            placement = await self._topics.prepare_topic(
                target_workspace=target_workspace,
                interaction_id=interaction_id,
                gateway_decision=decision,
                actor=system_actor,
            )
            self._placements[interaction_id] = placement
            logger.info(
                "Import Bus 为在途轮次确定话题: source=%s, interaction_id=%s, topic_id=%s, "
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
    async def _serialized(
        self, turn: InFlightTurn, participant: ActorIdentity
    ) -> AsyncIterator[None]:
        """在这一轮所属外部会话的串行门内执行，并计入在途请求；关闭开始后拒绝。"""
        if self._closed:
            # 认证网关先于编排器关闭；关闭前已通过认证的请求在这里收到同一
            # 可重试结论，而不是在已停止的子系统上失败。
            raise ImportBusUnavailableError(reason="shutting_down")
        key = ImportConversationKey.for_turn(participant, turn)
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
    def _interaction_id(turn: InFlightTurn, participant: ActorIdentity, principal_id: str) -> str:
        """写入时确定话题与导入对同一轮推导出同一个标识，回流因此能找到已确定的话题。"""
        return derive_import_interaction_id(
            principal_id=principal_id,
            actor=participant,
            source=turn.source,
            external_conversation_id=turn.external_conversation_id,
            external_turn_id=turn.external_turn_id,
        )

    async def _ingest_serialized(
        self,
        turn: ImportedTurn,
        *,
        system_actor: WorkingActor,
        target_workspace: WorkspaceIdentity,
        participant: ActorIdentity,
        principal_id: str,
        interaction_id: str,
    ) -> ImportTurnResult:
        status = await self._interactions.interaction_status(
            target_workspace=target_workspace,
            interaction_id=interaction_id,
            actor=system_actor,
        )
        if status is not None and status.state in ("succeeded", "unknown"):
            return self._duplicate(
                turn, target_workspace, participant, interaction_id, status.state
            )
        if status is not None and status.state == "pending":
            # 上一次提交已被接纳、仍在应用：继续等待，不再路由与提交。
            settled = await self._wait_applied(
                turn, system_actor, target_workspace, participant, interaction_id
            )
            return self._duplicate(
                turn, target_workspace, participant, interaction_id, settled.state
            )

        if status is None and not turn.has_output:
            self._events.skipped(
                turn=turn,
                workspace=target_workspace,
                participant=participant,
                interaction_id=interaction_id,
            )
            return ImportTurnResult(status="skipped", interaction_id=interaction_id)

        # 尚未提交，或上一次提交在应用时失败：（再次）提交。写入时已确定话题的
        # 轮次沿用该话题，不再路由；其余轮次重新路由。
        placement = self._placements.get(interaction_id)
        if placement is not None:
            # 该话题在写入时已经建立，新话题的标题与摘要随之写入。
            topic_id, is_new_topic = placement.topic_id, placement.is_new_topic
            new_topic_title = new_topic_summary = None
        else:
            try:
                decision = await self._request_gateway_decision(
                    turn, system_actor, target_workspace, participant, principal_id
                )
            except ImportBusUnavailableError as error:
                self._record_deferred(
                    turn,
                    target_workspace,
                    participant,
                    interaction_id,
                    "gateway_unavailable",
                    error.__cause__,
                )
                raise
            topic_id = decision.target_topic_id
            is_new_topic = topic_id == _NEW_TOPIC
            # 新话题以 Gateway 给出的标题与摘要建立，后续路由依赖它们。
            new_topic_title = decision.new_topic_title if is_new_topic else None
            new_topic_summary = decision.new_topic_summary if is_new_topic else None
        payload = seal_interaction(
            turn_events=turn.to_turn_events(),
            model_used=turn.model_used or "",
            used_attachments=(),
            outcome=turn.outcome,
        )
        try:
            # 交互提交是 root 能力：由系统执行，参与的 Agent 记为发起者。
            await self._interactions.submit_interaction(
                target_workspace=target_workspace,
                participant=participant,
                payload=payload,
                requested_topic_id=topic_id,
                interaction_id=interaction_id,
                origin="import_bus",
                new_topic_title=new_topic_title,
                new_topic_summary=new_topic_summary,
                actor=system_actor,
            )
        except InteractionAdmissionUnavailableError as error:
            self._record_deferred(
                turn, target_workspace, participant, interaction_id, "queue_unavailable", error
            )
            raise ImportBusUnavailableError(reason="queue_unavailable") from error
        # 提交已被接纳，这一轮的话题改由提交记录承载：此后的重发按提交进度处理，
        # 应用失败后的重发重新路由。队列拥塞时登记保留，重发仍然沿用。
        self._placements.pop(interaction_id, None)

        applied = await self._wait_applied(
            turn, system_actor, target_workspace, participant, interaction_id
        )
        logger.info(
            "Import Bus 确认轮次: source=%s, interaction_id=%s, new_topic=%s",
            turn.source,
            interaction_id,
            is_new_topic,
        )
        self._events.accepted(
            turn=turn,
            workspace=target_workspace,
            participant=participant,
            interaction_id=interaction_id,
            topic_id=applied.topic_id,
            is_new_topic=is_new_topic,
        )
        return ImportTurnResult(status="accepted", interaction_id=interaction_id)

    async def _wait_applied(
        self,
        turn: ImportedTurn,
        system_actor: WorkingActor,
        target_workspace: WorkspaceIdentity,
        participant: ActorIdentity,
        interaction_id: str,
    ) -> InteractionStatus:
        """等待已接纳的交互应用完成；超时与应用失败都不确认，交回给连接器重发。"""
        status = await self._interactions.wait_interaction(
            target_workspace=target_workspace,
            interaction_id=interaction_id,
            timeout_seconds=self._apply_wait_seconds,
            actor=system_actor,
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
                workspace=target_workspace,
                participant=participant,
                interaction_id=interaction_id,
                reason="apply_failed",
                error_class=None,
            )
            raise ImportBusApplyFailedError()
        if status.state == "pending":
            self._events.deferred(
                turn=turn,
                workspace=target_workspace,
                participant=participant,
                interaction_id=interaction_id,
                reason="apply_pending",
                error_class=None,
            )
            raise ImportBusUnavailableError(reason="apply_pending")
        return status

    def _duplicate(
        self,
        turn: ImportedTurn,
        workspace: WorkspaceIdentity,
        participant: ActorIdentity,
        interaction_id: str,
        state: str,
    ) -> ImportTurnResult:
        logger.info(
            "Import Bus 忽略已应用的轮次: source=%s, interaction_id=%s, state=%s",
            turn.source,
            interaction_id,
            state,
        )
        self._events.duplicate(
            turn=turn, workspace=workspace, participant=participant, interaction_id=interaction_id
        )
        return ImportTurnResult(status="duplicate", interaction_id=interaction_id)

    async def _request_gateway_decision(
        self,
        turn: InFlightTurn,
        system_actor: WorkingActor,
        target_workspace: WorkspaceIdentity,
        participant: ActorIdentity,
        principal_id: str,
    ) -> GatewayDecision:
        """以 PASSIVE_MEMORY 模式只取得话题路由与新话题的标题和摘要。

        最近一轮带外部标识时附上它实际所在的话题：话题路由失败时，Gateway
        续接上一轮的话题，而不是新建无标题话题。Gateway 整体抛出的可恢复
        异常转为 ``gateway_unavailable``，原异常作为 cause；是否记为导入推迟
        由调用方决定。
        """
        recent_turns = turn.recent_turns[-self._gateway_session_turn_limit :]
        previous_topic_id = (
            await self._previous_topic_id(
                turn, system_actor, target_workspace, participant, principal_id
            )
            if recent_turns
            else None
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
            result = await self._topics.route_passive_turn(
                target_workspace=target_workspace,
                message=turn.user_message,
                session_turns=session_turns,
                request_timeout_ms=self._gateway_request_timeout_ms,
                actor=system_actor,
            )
        except Exception as error:
            if not is_recoverable_gateway_error(error):
                raise
            raise ImportBusUnavailableError(reason="gateway_unavailable") from error
        if result.kind != "decision":
            raise ImportBusContractError("PASSIVE_MEMORY 不得返回命令结果")
        return result.decision

    async def _previous_topic_id(
        self,
        turn: InFlightTurn,
        system_actor: WorkingActor,
        target_workspace: WorkspaceIdentity,
        participant: ActorIdentity,
        principal_id: str,
    ) -> str | None:
        """查出最近一轮所在的话题：写入时已确定话题、尚未回流的取该话题，否则取它应用后的话题。

        该轮未带外部标识、写入时没有确定话题也未导入或尚未应用成功时为空。连接器的
        回流可能晚于下一轮的写入到达，此时上一轮只有写入时确定的话题。
        """
        previous = turn.recent_turns[-1]
        if previous.external_turn_id is None:
            return None
        previous_interaction_id = derive_import_interaction_id(
            principal_id=principal_id,
            actor=participant,
            source=turn.source,
            external_conversation_id=turn.external_conversation_id,
            external_turn_id=previous.external_turn_id,
        )
        placement = self._placements.get(previous_interaction_id)
        if placement is not None:
            return placement.topic_id
        status = await self._interactions.interaction_status(
            target_workspace=target_workspace,
            interaction_id=previous_interaction_id,
            actor=system_actor,
        )
        return status.topic_id if status is not None else None

    def _record_deferred(
        self,
        turn: ImportedTurn,
        workspace: WorkspaceIdentity,
        participant: ActorIdentity,
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
            workspace=workspace,
            participant=participant,
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
