"""Workspace Actor 访问注册表：准入状态与行为白名单的权威配置。

两类登记之一：本注册表由 Workspace
访问基础设施持有和查询，回答"这个 Actor 在这个 Workspace 是否被允许进入、
进入后允许执行哪些 operation"。一条访问记录同时承载准入状态与行为白名单，
不强制拆成两个 store。

职责边界：

- 只保存授权配置，不保存 Memory 可见性，不执行 Patchouli 业务；
- 键使用完整 Workspace 归属坐标（``owner_user_id + workspace_id``）与
  Actor 的 ``user_id + agent_id``；外部会话、run/frame 和调用协议
  不进入权限键；
- ``team_id`` 由可信身份关系提供给资源 policy，不进入本注册表；
- 启动时从本地配置（``configs/workspace_actors.yaml``）装载，记录可以在运行时
  经 root API 新增、修改与删除，改动先写回生效的登记文件再替换内存中的内容。
  授权者每次授权都即时查询本注册表，编辑在下一次授权时生效。

用户级记录：``agent_id`` 为 ``None`` 的记录覆盖该用户的
所有具体 Agent，但不覆盖保留的 ``system``——system 必须单独显式登记。
匹配时精确记录优先；精确记录禁用即拒绝，不回落到用户级记录；每个
(owner, workspace, user) 至多一条用户级记录，重复在装载期失败。

``actor.user_id == workspace.owner_user_id`` 是准入前提，因此本注册表在
装载期拒绝跨 owner 的访问记录（跨 owner 的访问不可表达）；相同 owner 也不表示自动获准——缺失记录即准入失败。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from hivememory.config.access import (
    WorkspaceActorAccessEntry,
    WorkspaceActorsFile,
    file_digest,
    write_registration_file,
)
from hivememory.core.access import WorkspaceOperation
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.errors import AccessRegistrationConflictError, AccessRegistrationNotFoundError

if TYPE_CHECKING:
    from hivememory.core.models import ActorIdentity, WorkspaceIdentity

__all__ = [
    "WorkspaceActorAccessRecord",
    "WorkspaceActorAccessRegistry",
]

type RecordKey = tuple[str, str, str, str | None]


@dataclass(frozen=True)
class WorkspaceActorAccessRecord:
    """一条 Workspace Actor 访问记录：准入 + 行为白名单。

    ``enabled=False`` 表示该 Actor 当前不被允许进入该 Workspace；
    ``allowed_operations`` 是进入后的行为上限，**允许为空**——空集合代表
    可进入但未获准执行任何资源操作，不是身份认证失败。

    ``agent_id`` 为 ``None`` 表示用户级记录：覆盖该用户的所有具体 Agent，
    不覆盖保留的 ``system``。
    """

    owner_user_id: str
    workspace_id: str
    user_id: str
    agent_id: str | None
    enabled: bool = True
    allowed_operations: frozenset[WorkspaceOperation] = frozenset()

    @property
    def key(self) -> RecordKey:
        """注册表内部唯一键：完整 Workspace 坐标 + Actor 坐标。"""
        return (self.owner_user_id, self.workspace_id, self.user_id, self.agent_id)


class WorkspaceActorAccessRegistry:
    """Workspace Actor 访问注册表：启动时装载，可在运行时编辑。

    供认证一侧（``workspace.authentication.WorkspaceAuthenticator``）的
    内部准入与操作授权者（``workspace.authorization``）的逐次授权查询；
    认证网关经认证一侧完成 Workspace 准入，不直接读取本注册表。
    未登记 Workspace、未知 Actor 或缺失访问记录一律查无结果，由调用方
    fail closed；本类不区分"未登记"与"已吊销"，避免泄漏配置细节。

    运行时编辑由 root API 调用：每次编辑先构造新的完整记录集，并走与装载
    相同的校验，写回文件成功后才替换内存中的记录；校验或写入失败时内存与
    文件都保持原样。编辑方法都是同步的，在事件循环中调用，校验、写回与
    替换之间没有 ``await``，两次编辑不会交错。
    """

    def __init__(
        self,
        records: Iterable[WorkspaceActorAccessRecord],
        *,
        file_path: Path | None = None,
        file_digest: str | None = None,
    ) -> None:
        """装载访问记录；重复键、跨 owner 记录与配置矛盾在装载期显式失败。

        ``file_path`` 是写回的位置：没有它的注册表只能查询，不能在运行时
        编辑。``file_digest`` 是装载时文件内容的摘要，写回前以它核对文件
        是否被外部改动。
        """
        self._records = self._validated(records)
        self._file_path = file_path
        self._file_digest = file_digest

    @staticmethod
    def _validated(
        records: Iterable[WorkspaceActorAccessRecord],
    ) -> dict[RecordKey, WorkspaceActorAccessRecord]:
        by_key: dict[RecordKey, WorkspaceActorAccessRecord] = {}
        for record in records:
            if not isinstance(record, WorkspaceActorAccessRecord):
                raise TypeError("访问记录必须是 WorkspaceActorAccessRecord")
            if record.agent_id == SYSTEM_AGENT_ID and record.allowed_operations:
                # system 的 root 能力独立于 actor 操作白名单，不能借登记授予
                # actor API 权限，也不能用 operation 表达管理许可。
                raise ValueError(f"保留 system Actor 的访问记录不得配置 operation: {record.key}")
            if not record.enabled and record.allowed_operations:
                # 禁用记录上的白名单没有意义，属于配置矛盾，装载期拒绝。
                raise ValueError(f"Workspace Actor 访问记录已禁用却配置了行为白名单: {record.key}")
            if record.user_id != record.owner_user_id:
                # 准入要求 actor user 等于 workspace owner；
                # 跨 owner 的成员记录不可表达。
                raise ValueError(
                    f"Workspace Actor 访问记录的 user 必须等于 workspace owner: {record.key}"
                )
            if record.key in by_key:
                # 用户级记录键以 ``agent_id=None`` 收敛：同一 (owner,
                # workspace, user) 的第二条用户级记录在此被拒绝。
                raise ValueError(f"Workspace Actor 访问记录键重复: {record.key}")
            by_key[record.key] = record
        return by_key

    @classmethod
    def from_config(
        cls,
        workspace_actors: WorkspaceActorsFile,
        *,
        file_path: Path | None = None,
        file_digest: str | None = None,
    ) -> WorkspaceActorAccessRegistry:
        """由访问登记文件构造注册表；未知 operation 值装载期失败，不静默丢弃。"""
        return cls(
            [_record_from_config(entry) for entry in workspace_actors.workspace_actors],
            file_path=file_path,
            file_digest=file_digest,
        )

    def record_for(
        self,
        workspace_identity: WorkspaceIdentity,
        actor_identity: ActorIdentity,
    ) -> WorkspaceActorAccessRecord | None:
        """按完整坐标查询访问记录；查无结果返回 ``None``（调用方拒绝）。

        精确记录优先：命中（含禁用）即返回，由调用方按 ``enabled`` 拒绝，
        不回落到用户级记录。未命中精确记录且 Actor 不是保留 ``system``
        时，回落到该用户的用户级记录（``agent_id=None``）；``system``
        不被用户级记录覆盖，必须单独登记。
        """
        coordinates = (
            workspace_identity.owner_user_id,
            workspace_identity.workspace_id,
            actor_identity.user_id,
        )
        exact = self._records.get((*coordinates, actor_identity.agent_id))
        if exact is not None:
            return exact
        if actor_identity.agent_id == SYSTEM_AGENT_ID:
            return None
        return self._records.get((*coordinates, None))

    def records_in(self, workspace_identity: WorkspaceIdentity) -> list[WorkspaceActorAccessRecord]:
        """该 Workspace 坐标下的全部访问记录，按文件中的顺序。"""
        coordinates = (workspace_identity.owner_user_id, workspace_identity.workspace_id)
        return [
            record
            for record in self._records.values()
            if (record.owner_user_id, record.workspace_id) == coordinates
        ]

    # ---- 运行时编辑 ----

    def put(self, record: WorkspaceActorAccessRecord) -> None:
        """新增一条访问记录，或替换同一坐标的已有记录。"""
        if not isinstance(record, WorkspaceActorAccessRecord):
            raise TypeError("访问记录必须是 WorkspaceActorAccessRecord")
        records = dict(self._records)
        records[record.key] = record
        self._commit(records.values())

    def delete(self, key: RecordKey) -> None:
        """删除一条访问记录；之后该坐标按缺失记录拒绝，精确记录被删时回落到用户级记录。"""
        if key not in self._records:
            raise AccessRegistrationNotFoundError(details={"agent_id": key[3]})
        self._commit(record for record_key, record in self._records.items() if record_key != key)

    def _commit(self, records: Iterable[WorkspaceActorAccessRecord]) -> None:
        """校验新的完整记录集，写回文件，成功后替换内存中的记录。

        写回前核对摘要：文件在运行中被外部改动时以 ``file_modified`` 拒绝，
        不覆盖手工修改。
        """
        if self._file_path is None:
            raise RuntimeError("没有登记文件的访问注册表不能在运行时编辑")
        validated = self._validated(records)
        if file_digest(self._file_path) != self._file_digest:
            raise AccessRegistrationConflictError(
                details={"reason": "file_modified", "path": str(self._file_path)}
            )
        self._file_digest = write_registration_file(
            self._file_path,
            WorkspaceActorsFile(workspace_actors=_file_entries(validated)),
        )
        self._records = validated


def _record_from_config(entry: WorkspaceActorAccessEntry) -> WorkspaceActorAccessRecord:
    """把文件中的一条记录转为注册表记录；operation 枚举值在这里校验。"""
    operations = []
    for name in entry.allowed_operations:
        try:
            operations.append(WorkspaceOperation(name))
        except ValueError as exc:
            raise ValueError(
                f"Workspace Actor 访问登记包含未知 operation {name!r}: "
                f"({entry.owner_user_id}, {entry.workspace_id}, {entry.user_id}, {entry.agent_id})"
            ) from exc
    return WorkspaceActorAccessRecord(
        owner_user_id=entry.owner_user_id,
        workspace_id=entry.workspace_id,
        user_id=entry.user_id,
        agent_id=entry.agent_id,
        enabled=entry.enabled,
        allowed_operations=frozenset(operations),
    )


def _file_entries(
    records: Mapping[RecordKey, WorkspaceActorAccessRecord],
) -> list[WorkspaceActorAccessEntry]:
    """登记文件中的内容；operation 按枚举定义的顺序写出，文件 diff 稳定。"""
    order = list(WorkspaceOperation)
    return [
        WorkspaceActorAccessEntry(
            owner_user_id=record.owner_user_id,
            workspace_id=record.workspace_id,
            user_id=record.user_id,
            agent_id=record.agent_id,
            enabled=record.enabled,
            allowed_operations=[
                operation.value for operation in sorted(record.allowed_operations, key=order.index)
            ],
        )
        for record in records.values()
    ]
