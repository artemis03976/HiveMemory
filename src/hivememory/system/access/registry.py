"""System Actor 接入注册表：受信调用来源的准入配置。

两类访问登记之一：本注册表由 System 持有，回答"哪个调用来源被允许
接入、通过什么 adapter、以什么凭据证明、如何解析 Actor 身份"。它**不以
principal 配置授予任何 Workspace operation**——操作许可只由 Workspace
Actor 访问注册表（``workspace.registry``）表达。

最小内容：稳定调用来源标识、是否启用、连接/接收方式、可用 adapter、
身份解析规则、能否声明保留的 ``system``，以及外部来源的接入凭据。进程内
入口（server、本地脚本、系统自身）以自己的 principal 接入，不需要凭据；
外部入口传入的 principal 只是请求中的声明，必须以凭据证明。principal 同名
请求字段不构成"已注册"——请求必须匹配启用的接入登记、相应 adapter 与凭据。

登记在启动时装载，外部来源可以在运行时登记、修改、轮换凭据与删除，改动
先写回登记文件再替换内存中的内容；进程内来源只能经文件修改并重启，因为
它们关系到管理接口、chat 与导入本身能否工作。
"""

from __future__ import annotations

import dataclasses
import logging
import secrets
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from hivememory.config.access import (
    PrincipalCredentialsFile,
    SystemPrincipalAccessEntry,
    SystemPrincipalsFile,
    file_digest,
    load_principal_credentials,
    write_registration_file,
)
from hivememory.config.app import environment_value
from hivememory.core.errors import AccessRegistrationConflictError, AccessRegistrationNotFoundError

logger = logging.getLogger(__name__)

__all__ = [
    "HTTP_ADAPTER",
    "IMPORT_ADAPTER",
    "LOCAL_ADAPTER",
    "MCP_ADAPTER",
    "EXTERNAL_ADAPTERS",
    "SYSTEM_PRINCIPAL_ID",
    "SystemActorAccessEntry",
    "SystemActorAccessRegistry",
]

#: 本地受信 adapter 的默认标识：本地进程内调用方（Alice、System、测试组合）。
LOCAL_ADAPTER = "local"
#: server 的管理、chat 与取消路由（前端）以自身 principal 接入时使用的 adapter 标识。
HTTP_ADAPTER = "http"
#: MCP 入口包（外部 harness 的操作适配器）以请求声明的 principal 接入时使用的 adapter 标识。
MCP_ADAPTER = "mcp"
#: Import Bus 路由（外部 harness 的连接器）以请求声明的 principal 接入时使用的 adapter 标识。
IMPORT_ADAPTER = "import"
#: 外部入口的 adapter：运行时登记的来源只能列出它们。
EXTERNAL_ADAPTERS = frozenset({MCP_ADAPTER, IMPORT_ADAPTER})

#: 系统自身的调用来源：任务进程的注册入口与 Import Bus 服务以它经 ``local``
#: adapter 认证 (user, ``system``)，取得执行系统级操作（系统阶段、取消、确定话题与
#: 交互提交）的 root 工作 actor。须在接入登记中登记、允许声明 ``system``。
SYSTEM_PRINCIPAL_ID = "hivememory:system"

# 只有进程内入口以自己的 principal 接入、不携带凭据；其余 adapter 都是外部
# 入口，principal 取自请求声明，只有核对凭据后才成立。新增的 adapter 默认
# 属于外部入口，必须以凭据证明，因此没有凭据的 principal 无法被外部请求借用。
_IN_PROCESS_ADAPTERS = frozenset({LOCAL_ADAPTER, HTTP_ADAPTER})

#: 运行时登记的来源的连接方式标识。
_HARNESS_KIND = "harness"

type CredentialSource = Literal["env", "generated"]


@dataclass(frozen=True)
class SystemActorAccessEntry:
    """一条 System 接入登记：来源标识 + 接收方式 + 凭据 + 身份解析规则。

    ``adapters`` 声明该来源可经哪些 adapter 接入；网关认证时必须匹配，
    adapter 不匹配是第一层失败。一条登记要么只列进程内 adapter，要么只列
    外部 adapter，混用在装载期拒绝：外部来源的每次认证都核对凭据，进程内
    来源从不携带凭据。``allowed_user_ids`` 与 ``allowed_agent_ids`` 是可选的
    身份解析收紧（``None`` 表示不按用户或 Agent 限定，仍受 owner 约束与
    Workspace 登记约束）。``allow_system_actor`` 为 ``False`` 时不能声明保留
    的 ``system``。

    ``credential`` 是这个来源当前生效的凭据，``credential_source`` 说明它
    来自哪里：登记了 ``credential_env`` 且该环境变量有值时取环境变量
    （``env``），否则取生成凭据（``generated``）；两者都没有时为 ``None``，
    外部来源的每次认证都被拒绝。进程内来源不能声明 ``credential_env``。
    凭据不进入 ``repr``。
    """

    principal_id: str
    kind: str = "local-process"
    enabled: bool = True
    adapters: frozenset[str] = field(default_factory=lambda: frozenset({LOCAL_ADAPTER}))
    allowed_user_ids: frozenset[str] | None = None
    allowed_agent_ids: frozenset[str] | None = None
    allow_system_actor: bool = False
    credential_env: str | None = None
    credential: str | None = field(default=None, repr=False)
    credential_source: CredentialSource | None = None

    def __post_init__(self) -> None:
        if not self.principal_id.strip():
            raise ValueError("principal_id 不能为空")
        if not self.kind.strip():
            raise ValueError("kind 不能为空")
        if not self.adapters:
            raise ValueError(f"接入登记 {self.principal_id!r} 至少需要声明一个 adapter")
        if self.adapters & _IN_PROCESS_ADAPTERS and self.adapters - _IN_PROCESS_ADAPTERS:
            raise ValueError(
                f"接入登记 {self.principal_id!r} 不能同时列出进程内 adapter（local、http）"
                "与外部 adapter"
            )
        # 进程内入口总是不带凭据认证：与凭据同时登记会使这些入口一律被拒绝。
        if self.is_in_process and self.credential_env is not None:
            raise ValueError(
                f"接入登记 {self.principal_id!r} 声明了 credential_env，"
                "不能再列出进程内 adapter（local、http）"
            )

    @property
    def is_in_process(self) -> bool:
        """进程内来源：只列 ``local`` 或 ``http``，不携带凭据，只能经文件修改。"""
        return not (self.adapters - _IN_PROCESS_ADAPTERS)


class SystemActorAccessRegistry:
    """System 接入注册表：启动时装载，外部来源可在运行时编辑。

    供 ``SystemPrincipalAuthenticator`` 在第一项认证中查询。未登记与已禁用
    的来源统一按未知 principal 拒绝，不区分两者，避免向调用方泄漏配置细节。

    运行时编辑只针对外部来源，每次编辑先构造新的完整登记并校验，写回文件
    成功后才替换内存中的内容；校验或写入失败时内存与文件都保持原样。编辑
    方法都是同步的，在事件循环中调用，校验、写回与替换之间没有 ``await``，
    两次编辑不会交错。
    """

    def __init__(
        self,
        entries: Iterable[SystemActorAccessEntry],
        *,
        file_path: Path | None = None,
        file_digest: str | None = None,
        credentials_path: Path | None = None,
        generated_credentials: Mapping[str, str] | None = None,
    ) -> None:
        """装载接入登记。

        ``file_path`` 与 ``credentials_path`` 是写回的位置：没有它们的注册表
        只能查询，不能在运行时编辑。``file_digest`` 是装载时文件内容的摘要，
        写回前以它核对文件是否被外部改动。``generated_credentials`` 是凭据
        文件的内容，写回凭据文件时保留其中其他来源的凭据。
        """
        self._entries = self._validated(entries)
        self._file_path = file_path
        self._file_digest = file_digest
        self._credentials_path = credentials_path
        self._generated = dict(generated_credentials or {})

    @staticmethod
    def _validated(
        entries: Iterable[SystemActorAccessEntry],
    ) -> dict[str, SystemActorAccessEntry]:
        by_id: dict[str, SystemActorAccessEntry] = {}
        for entry in entries:
            if not isinstance(entry, SystemActorAccessEntry):
                raise TypeError("接入登记必须是 SystemActorAccessEntry")
            if entry.principal_id in by_id:
                raise ValueError(f"System 接入登记 principal 重复: {entry.principal_id!r}")
            by_id[entry.principal_id] = entry
        return by_id

    @classmethod
    def from_config(
        cls,
        principals: SystemPrincipalsFile,
        *,
        file_path: Path | None = None,
        file_digest: str | None = None,
        credentials_path: Path | None = None,
    ) -> SystemActorAccessRegistry:
        """由接入登记文件构造注册表，并取得各外部来源的凭据。

        环境变量凭据与其他密钥同源：进程环境变量优先，其次是 ``.env`` 文件，
        读取时去掉首尾空白，避免环境文件中的换行符使正确配置的凭据永远不
        匹配。环境变量有值时优先于凭据文件中的生成凭据，规则与 Provider 凭证
        相同。外部来源两者都没有时记录一条警告并照常装配：该来源此后的每次
        认证都以 ``invalid_credential`` 拒绝，不使用它的部署不受影响。凭据只
        存在于内存中的登记里，不写入日志。
        """
        generated = (
            load_principal_credentials(credentials_path).credentials
            if credentials_path is not None
            else {}
        )
        entries = []
        for config_entry in principals.principals:
            entry = _entry_from_config(config_entry, generated)
            if not entry.is_in_process and entry.credential is None:
                logger.warning(
                    "接入登记 %r 没有可用的凭据（环境变量 %s 未设置，也没有生成凭据），"
                    "该来源的认证将一律被拒绝",
                    entry.principal_id,
                    entry.credential_env,
                )
            entries.append(entry)
        return cls(
            entries,
            file_path=file_path,
            file_digest=file_digest,
            credentials_path=credentials_path,
            generated_credentials=generated,
        )

    def entry_for(self, principal_id: str) -> SystemActorAccessEntry | None:
        """按来源标识查询接入登记；查无结果返回 ``None``（调用方拒绝）。"""
        return self._entries.get(principal_id)

    def entries(self) -> list[SystemActorAccessEntry]:
        """全部接入登记，按文件中的顺序，运行时登记的排在最后。"""
        return list(self._entries.values())

    # ---- 运行时编辑：只针对外部来源 ----

    def register(
        self,
        principal_id: str,
        *,
        adapters: frozenset[str],
        allowed_agent_ids: frozenset[str] | None,
        enabled: bool = True,
    ) -> tuple[SystemActorAccessEntry, str]:
        """登记一个外部来源并生成它的凭据，返回登记与凭据明文。

        运行时登记的来源 ``kind`` 为 ``harness``，不能声明 ``system``；
        principal 已登记时以 ``duplicate`` 拒绝。
        """
        if principal_id in self._entries:
            raise AccessRegistrationConflictError(
                details={"reason": "duplicate", "principal_id": principal_id}
            )
        _require_external(adapters)
        credential = secrets.token_urlsafe(32)
        entry = SystemActorAccessEntry(
            principal_id=principal_id,
            kind=_HARNESS_KIND,
            enabled=enabled,
            adapters=adapters,
            allowed_agent_ids=allowed_agent_ids,
            credential=credential,
            credential_source="generated",
        )
        self._commit(
            {**self._entries, principal_id: entry},
            generated={**self._generated, principal_id: credential},
        )
        return entry, credential

    def update(
        self,
        principal_id: str,
        *,
        adapters: frozenset[str],
        allowed_agent_ids: frozenset[str] | None,
        enabled: bool,
    ) -> SystemActorAccessEntry:
        """修改外部来源的启用状态、adapter 与可声明的 Agent，凭据不变。"""
        current = self._editable(principal_id)
        _require_external(adapters)
        entry = dataclasses.replace(
            current, enabled=enabled, adapters=adapters, allowed_agent_ids=allowed_agent_ids
        )
        self._commit({**self._entries, principal_id: entry}, generated=self._generated)
        return entry

    def rotate_credential(self, principal_id: str) -> tuple[SystemActorAccessEntry, str]:
        """为外部来源生成新凭据，旧凭据随即失效；返回登记与新凭据明文。

        凭据来自环境变量时以 ``credential_from_env`` 拒绝：环境变量优先，
        生成的凭据不会生效。
        """
        current = self._editable(principal_id)
        if current.credential_source == "env":
            raise AccessRegistrationConflictError(
                details={"reason": "credential_from_env", "principal_id": principal_id}
            )
        credential = secrets.token_urlsafe(32)
        entry = dataclasses.replace(current, credential=credential, credential_source="generated")
        self._commit(
            {**self._entries, principal_id: entry},
            generated={**self._generated, principal_id: credential},
        )
        return entry, credential

    def delete(self, principal_id: str) -> None:
        """删除外部来源与它的生成凭据，它之后的认证以 ``unknown_principal`` 拒绝。"""
        self._editable(principal_id)
        entries = {pid: entry for pid, entry in self._entries.items() if pid != principal_id}
        generated = {pid: value for pid, value in self._generated.items() if pid != principal_id}
        self._commit(entries, generated=generated)

    def _editable(self, principal_id: str) -> SystemActorAccessEntry:
        entry = self._entries.get(principal_id)
        if entry is None:
            raise AccessRegistrationNotFoundError(details={"principal_id": principal_id})
        if entry.is_in_process:
            raise AccessRegistrationConflictError(
                details={"reason": "protected", "principal_id": principal_id}
            )
        return entry

    def _commit(
        self,
        entries: dict[str, SystemActorAccessEntry],
        *,
        generated: dict[str, str],
    ) -> None:
        """写回改动的文件，全部写入成功后替换内存中的登记与凭据。

        登记文件写回前核对摘要：文件在运行中被外部改动时以 ``file_modified``
        拒绝，不覆盖手工修改。生成凭据先于登记文件写入：登记文件写入失败时，
        凭据文件中多出的凭据不属于任何登记，下次装载时被忽略。
        """
        if self._file_path is None or self._credentials_path is None:
            raise RuntimeError("没有登记文件的接入注册表不能在运行时编辑")
        entries_changed = _file_entries(entries) != _file_entries(self._entries)
        if entries_changed and file_digest(self._file_path) != self._file_digest:
            raise AccessRegistrationConflictError(
                details={"reason": "file_modified", "path": str(self._file_path)}
            )
        if generated != self._generated:
            write_registration_file(
                self._credentials_path, PrincipalCredentialsFile(credentials=generated)
            )
        new_digest = self._file_digest
        if entries_changed:
            new_digest = write_registration_file(
                self._file_path, SystemPrincipalsFile(principals=_file_entries(entries))
            )
        self._entries = entries
        self._generated = generated
        self._file_digest = new_digest


def _require_external(adapters: frozenset[str]) -> None:
    """运行时登记与修改只接受外部 adapter 的非空子集。"""
    if not adapters or not adapters <= EXTERNAL_ADAPTERS:
        raise ValueError(f"运行时登记的来源只能列出外部 adapter：{sorted(EXTERNAL_ADAPTERS)}")


def _entry_from_config(
    config_entry: SystemPrincipalAccessEntry, generated: Mapping[str, str]
) -> SystemActorAccessEntry:
    """把文件中的一条登记转为注册表登记，按优先级取得凭据。"""
    credential: str | None = None
    source: CredentialSource | None = None
    if config_entry.credential_env is not None:
        credential = (environment_value(config_entry.credential_env) or "").strip() or None
        source = "env" if credential is not None else None
    if credential is None and config_entry.principal_id in generated:
        credential = generated[config_entry.principal_id]
        source = "generated"
    return SystemActorAccessEntry(
        principal_id=config_entry.principal_id,
        kind=config_entry.kind,
        enabled=config_entry.enabled,
        adapters=frozenset(config_entry.adapters),
        allowed_user_ids=_optional_set(config_entry.allowed_user_ids),
        allowed_agent_ids=_optional_set(config_entry.allowed_agent_ids),
        allow_system_actor=config_entry.allow_system_actor,
        credential_env=config_entry.credential_env,
        credential=credential,
        credential_source=source,
    )


def _optional_set(values: list[str] | None) -> frozenset[str] | None:
    return frozenset(values) if values is not None else None


def _optional_list(values: frozenset[str] | None) -> list[str] | None:
    return sorted(values) if values is not None else None


def _file_entries(
    entries: Mapping[str, SystemActorAccessEntry],
) -> list[SystemPrincipalAccessEntry]:
    """登记文件中的内容：凭据本身与凭据来源都不写入登记文件。"""
    return [
        SystemPrincipalAccessEntry(
            principal_id=entry.principal_id,
            kind=entry.kind,
            enabled=entry.enabled,
            adapters=sorted(entry.adapters),
            allowed_user_ids=_optional_list(entry.allowed_user_ids),
            allowed_agent_ids=_optional_list(entry.allowed_agent_ids),
            allow_system_actor=entry.allow_system_actor,
            credential_env=entry.credential_env,
        )
        for entry in entries.values()
    ]
