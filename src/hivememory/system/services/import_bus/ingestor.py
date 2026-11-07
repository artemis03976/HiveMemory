"""Import Bus 编排器：串行门内完成封口，经 root API 提交交互或为在途轮次确定话题。

Import Bus 只向记忆网络输入，不检索记忆、不返回记忆上下文：

    按外部会话串行 -> 推导 interaction_id -> 没有输出则跳过 -> 共用封口函数
    -> root API 提交交互（写入时已确定话题则以该话题提交，否则交给 root API
    以 PASSIVE_MEMORY 路由；查重、路由、提交与等待应用都在这一次调用内）
    -> 按结局确认或交回连接器重发

确认表示交互已经应用：应用失败或等待超时都不确认，连接器保留这一轮
稍后重发。等待在串行门内完成，同一外部会话的下一轮路由时，上一轮所在
的话题已经建立，可以成为候选话题与续接目标。

写入与更新在交互进行中提交，需要这一轮最终所在的话题。为这一轮确定话题时，
在同一串行门内经 root API 以与导入相同的输入路由并建立话题；已确定的话题按
interaction_id 登记，同一轮再次请求直接返回，回流时以该话题提交、不再路由。
确定话题只读取候选话题，不读取记忆。

提交与确定话题都是系统级操作：编排器以 Import Bus 服务认证得到的系统 root
工作 actor 调用 root API，每次调用由 root API 在方法入口授权一次，编排器
自己不授权、也不持有 scope。目标不是参与者驻留的 Workspace 时，root 调用
在任何路由与提交之前被拒绝。交互中的 Agent 与认证通过的 principal 标识作为
数据显式传入：外部会话键、交互标识与事件中的 Agent 取自它们，提交的发起者
是参与者。编排器不持有提交队列，也不登记维护任务；关闭时只等待在途导入
（含其等待应用的时间），队列由 Patchouli 自己排空。
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
from hivememory.core.protocol.gateway import GatewaySessionTurn
from hivememory.core.protocol.sealing import seal_interaction
from hivememory.patchouli.contracts.errors import InteractionAdmissionUnavailableError
from hivememory.system.services.import_bus.errors import (
    ImportBusApplyFailedError,
    ImportBusUnavailableError,
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
from hivememory.workspace.capability.root.passive_routing import PassiveTurnRouting
from hivememory.workspace.capability.root.topic import DeterminedTopic, RootTopicService

logger = logging.getLogger(__name__)


class ImportBusIngestor:
    """导入编排：同一外部会话内按到达顺序处理，不同会话互不阻塞。

    门内完成封口与一次 root 提交：同一外部会话的并发请求（例如两个 hook
    同时补发）因此按到达顺序处理，后到的请求看到前一个的结果。已应用的
    轮次返回 duplicate，仍在应用的继续等待，两者都不再调用 Gateway；只有
    应用失败的轮次重新路由后再提交，这些判断都在 root API 的提交方法内。

    写入时确定话题与导入共用同一串行门与在途计数：同一外部会话的确定话题请求
    与回流按到达顺序处理，关闭时同样等待在途的确定话题请求。``interactions``
    与 ``topics`` 是组合根注入的 root 交互 API 与 root 话题 API。
    """

    def __init__(
        self,
        *,
        interactions: RootInteractionService,
        topics: RootTopicService,
        gateway_session_turn_limit: int,
        apply_wait_seconds: float,
        runtime_events: RuntimeEventSink | None = None,
    ) -> None:
        self._interactions = interactions
        self._topics = topics
        self._gateway_session_turn_limit = gateway_session_turn_limit
        self._apply_wait_seconds = apply_wait_seconds
        self._events = ImportBusEventEmitter(runtime_events)
        self._serial_gate = KeyedSerialGate[ImportConversationKey]()
        self._closed = False
        self._inflight = 0
        self._idle = asyncio.Event()
        self._idle.set()
        # 写入时为仍在进行的一轮确定的话题，按 interaction_id 保存，回流的提交
        # 被接纳后取走。按轮次而不是按外部会话保存：下一轮的写入可能早于上一轮的
        # 回流到达。只在进程内、不持久化；从未回流的轮次的登记留到进程重启。
        self._placements: dict[str, DeterminedTopic] = {}

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
        路由时 Gateway 的可恢复异常、提交队列拥塞、等待应用超时与关闭中以
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

        第一次请求经 root API 以与导入相同的输入与续接规则路由并建立话题，
        然后登记。这一轮已经回流并应用时（后台运行的子代理可能在这一轮结束、
        回流之后才写入），root API 直接返回它实际所在的话题；已经提交但仍在
        应用时以 ``apply_pending`` 交回，写入方稍后重试。Gateway 的可恢复
        异常与关闭中以 ``ImportBusUnavailableError`` 交回；写入正等待这个结果，
        失败直接交给调用方，不记为导入推迟。确定话题时不检索、不读取记忆；参数
        的含义与导入相同。

        已确定话题的登记命中时直接返回，不经过 root API：登记项只由此前经过 root
        授权的请求写入，键由同一 principal 与参与者推导。
        """
        interaction_id = self._interaction_id(turn, participant, principal_id)
        async with self._serialized(turn, participant):
            placement = self._placements.get(interaction_id)
            if placement is not None:
                return placement.topic_id
            try:
                determined = await self._topics.determine_topic(
                    target_workspace=target_workspace,
                    interaction_id=interaction_id,
                    topic_source=self._routing(turn, participant, principal_id),
                    actor=system_actor,
                )
            except Exception as error:
                if not is_recoverable_gateway_error(error):
                    raise
                raise ImportBusUnavailableError(reason="gateway_unavailable") from error
            if determined is None:
                # 这一轮已经提交、等待应用超时，所在的话题要等应用完成才确定；写入方
                # 稍后重试时它已应用。
                raise ImportBusUnavailableError(reason="apply_pending")
            self._placements[interaction_id] = determined
            logger.info(
                "Import Bus 为在途轮次确定话题: source=%s, interaction_id=%s, topic_id=%s, "
                "new_topic=%s",
                turn.source,
                interaction_id,
                determined.topic_id,
                determined.is_new,
            )
            return determined.topic_id

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
        if not turn.has_output:
            # 没有输出的轮次从不提交，同一轮重发时同样没有输出，不必先查进度。
            self._events.skipped(
                turn=turn,
                workspace=target_workspace,
                participant=participant,
                interaction_id=interaction_id,
            )
            return ImportTurnResult(status="skipped", interaction_id=interaction_id)

        # 写入时已确定话题的轮次沿用该话题，不再路由；其余轮次交给 root API 路由。
        placement = self._placements.get(interaction_id)
        topic: str | PassiveTurnRouting = (
            placement.topic_id
            if placement is not None
            else self._routing(turn, participant, principal_id)
        )
        payload = seal_interaction(
            turn_events=turn.to_turn_events(),
            model_used=turn.model_used or "",
            used_attachments=(),
            outcome=turn.outcome,
        )
        try:
            # 交互提交是 root 能力：由系统执行，参与的 Agent 记为发起者。
            result = await self._interactions.submit_interaction(
                target_workspace=target_workspace,
                participant=participant,
                payload=payload,
                interaction_id=interaction_id,
                origin="import_bus",
                topic=topic,
                timeout_seconds=self._apply_wait_seconds,
                actor=system_actor,
            )
        except InteractionAdmissionUnavailableError as error:
            # 队列拥塞时登记保留，重发仍然沿用。
            self._record_deferred(
                turn, target_workspace, participant, interaction_id, "queue_unavailable", error
            )
            raise ImportBusUnavailableError(reason="queue_unavailable") from error
        except Exception as error:
            if not is_recoverable_gateway_error(error):
                raise
            self._record_deferred(
                turn, target_workspace, participant, interaction_id, "gateway_unavailable", error
            )
            raise ImportBusUnavailableError(reason="gateway_unavailable") from error
        # 提交已被接纳，这一轮的话题改由提交记录承载：此后的重发按提交进度处理，
        # 应用失败后的重发重新路由。
        self._placements.pop(interaction_id, None)

        if result.state == "failed":
            logger.warning(
                "Import Bus 轮次应用失败: source=%s, interaction_id=%s",
                turn.source,
                interaction_id,
            )
            self._record_deferred(
                turn, target_workspace, participant, interaction_id, "apply_failed", None
            )
            raise ImportBusApplyFailedError()
        if result.state == "pending":
            self._record_deferred(
                turn, target_workspace, participant, interaction_id, "apply_pending", None
            )
            raise ImportBusUnavailableError(reason="apply_pending")
        if result.previously_accepted:
            logger.info(
                "Import Bus 忽略已应用的轮次: source=%s, interaction_id=%s, state=%s",
                turn.source,
                interaction_id,
                result.state,
            )
            self._events.duplicate(
                turn=turn,
                workspace=target_workspace,
                participant=participant,
                interaction_id=interaction_id,
            )
            return ImportTurnResult(status="duplicate", interaction_id=interaction_id)

        is_new_topic = placement.is_new if placement is not None else result.new_topic
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
            topic_id=result.topic_id,
            is_new_topic=is_new_topic,
        )
        return ImportTurnResult(status="accepted", interaction_id=interaction_id)

    def _routing(
        self, turn: InFlightTurn, participant: ActorIdentity, principal_id: str
    ) -> PassiveTurnRouting:
        """这一轮的路由输入：用户消息、最近几轮，以及最近一轮所在的话题。

        最近一轮带外部标识时，写入时已为它确定话题、尚未回流的直接取该话题；
        否则给出它的交互标识，由 root API 查出它应用后实际所在的话题。连接器的
        回流可能晚于下一轮的写入到达，此时上一轮只有写入时确定的话题。
        """
        recent_turns = turn.recent_turns[-self._gateway_session_turn_limit :]
        previous_topic_id: str | None = None
        previous_interaction_id: str | None = None
        previous = recent_turns[-1] if recent_turns else None
        if previous is not None and previous.external_turn_id is not None:
            previous_interaction_id = derive_import_interaction_id(
                principal_id=principal_id,
                actor=participant,
                source=turn.source,
                external_conversation_id=turn.external_conversation_id,
                external_turn_id=previous.external_turn_id,
            )
            placement = self._placements.get(previous_interaction_id)
            if placement is not None:
                previous_topic_id = placement.topic_id
                previous_interaction_id = None
        return PassiveTurnRouting(
            message=turn.user_message,
            session_turns=tuple(
                GatewaySessionTurn(
                    user_message=recent.user_message,
                    assistant_text=recent.assistant_text,
                    topic_id=previous_topic_id if index == len(recent_turns) - 1 else None,
                )
                for index, recent in enumerate(recent_turns)
            ),
            previous_interaction_id=previous_interaction_id,
        )

    def _record_deferred(
        self,
        turn: ImportedTurn,
        workspace: WorkspaceIdentity,
        participant: ActorIdentity,
        interaction_id: str,
        reason: str,
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
