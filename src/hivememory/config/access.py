"""访问控制配置：System 接入登记与 Workspace Actor 访问登记的文件声明、装载与写回。

两类登记各用一个单独的 YAML 文件，
对应各自的配置所有者，``config.yaml`` 不再承载 access 段：

- ``configs/system_principals.yaml``：调用来源的接入登记（System 所有）；
- ``configs/workspace_actors.yaml``：Workspace Actor 的准入与行为白名单
  （workspace 所有）。

这两个生效文件是本机的个人配置，不受版本控制；仓库只发布同目录的
``*.example.yaml`` 模板。默认路径的生效文件不存在时，装载前从模板复制一份，
此后运行时编辑只写生效文件，模板保持原样。环境变量指定的路径不从模板生成：
它缺失时显式失败，避免路径写错时悄悄生成一份默认登记。拒绝未知字段
（``extra="forbid"``），文件内容非法时显式失败。

两份登记可以在运行时编辑，由各自的注册表写回当前生效的文件。装载时记下
文件内容的摘要，写回前核对：文件在运行中被外部改动时，注册表拒绝写回，
不覆盖手工修改。外部来源的生成凭据不进入登记文件，保存在接入登记文件同
目录的 ``principals.secrets.yaml``，该文件不受版本控制。
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from hivememory.config.files import seed_from_template, write_atomically

logger = logging.getLogger(__name__)

__all__ = [
    "SystemPrincipalAccessEntry",
    "WorkspaceActorAccessEntry",
    "SystemPrincipalsFile",
    "WorkspaceActorsFile",
    "PrincipalCredentialsFile",
    "AccessRegistration",
    "load_access_registration",
    "get_system_principals_file_path",
    "get_workspace_actors_file_path",
    "get_principal_credentials_file_path",
    "file_digest",
    "write_registration_file",
    "load_principal_credentials",
]

# 登记文件路径可用环境变量覆盖；缺省使用仓库 configs 目录下的生效文件。
PRINCIPALS_FILE_ENV = "HIVEMEMORY_PRINCIPALS_PATH"
WORKSPACE_ACTORS_FILE_ENV = "HIVEMEMORY_WORKSPACE_ACTORS_PATH"
PRINCIPALS_TEMPLATE = "system_principals.example.yaml"
WORKSPACE_ACTORS_TEMPLATE = "workspace_actors.example.yaml"


def _repo_configs_dir() -> Path:
    # src/hivememory/config/access.py → 仓库根目录 / configs
    return Path(__file__).resolve().parents[3] / "configs"


def get_system_principals_file_path() -> Path:
    """System 接入登记生效文件的路径（环境变量优先，缺省仓库 configs 目录）。"""
    configured = os.getenv(PRINCIPALS_FILE_ENV)
    if configured:
        return Path(configured)
    return _repo_configs_dir() / "system_principals.yaml"


def get_workspace_actors_file_path() -> Path:
    """Workspace Actor 访问登记生效文件的路径（环境变量优先，缺省仓库 configs 目录）。"""
    configured = os.getenv(WORKSPACE_ACTORS_FILE_ENV)
    if configured:
        return Path(configured)
    return _repo_configs_dir() / "workspace_actors.yaml"


def get_principal_credentials_file_path() -> Path:
    """外部来源生成凭据的文件路径：与生效的接入登记文件同目录。

    接入登记指向仓库外时，生成凭据随之放在仓库外，不需要另一个路径变量。
    """
    return get_system_principals_file_path().parent / "principals.secrets.yaml"


class SystemPrincipalAccessEntry(BaseModel):
    """System 接入登记配置：一个受信调用来源。

    ``adapters`` 声明可经哪些 adapter 接入（HTTP 入口使用 ``http``）；
    ``allowed_user_ids`` 与 ``allowed_agent_ids`` 是可选的身份解析收紧，
    ``None`` 表示不按用户或 Agent 限定。``credential_env`` 只写接入凭据所在的
    环境变量名，凭据本身不进入仓库；外部 adapter（``mcp``、``import``）的登记
    也可以不写它，改用生成凭据。``allow_system_actor`` 决定该来源能否声明保留
    的 ``system``、执行 root 授权的系统级操作，默认不能。principal 配置不授予
    任何 Workspace operation。
    """

    principal_id: str = Field(description="稳定调用来源标识（带命名空间）")
    kind: str = Field(default="local-process", description="连接/接收方式标识")
    enabled: bool = Field(default=True, description="是否启用该接入来源")
    adapters: list[str] = Field(
        default_factory=lambda: ["local"],
        description="该来源可使用的 adapter 标识",
    )
    allowed_user_ids: list[str] | None = Field(
        default=None,
        description="可选：该来源可服务的用户集合（身份解析收紧）",
    )
    allowed_agent_ids: list[str] | None = Field(
        default=None,
        description="可选：该来源可声明的 Agent 集合（身份解析收紧）",
    )
    allow_system_actor: bool = Field(
        default=False,
        description="该来源能否声明保留的 system actor（执行 root 授权的系统级操作）",
    )
    credential_env: str | None = Field(
        default=None,
        min_length=1,
        description="可选：接入凭据所在的环境变量名（凭据本身不写入配置）",
    )

    # 访问控制配置拼错字段会被静默丢弃并落到 deny-by-default，方向虽安全
    # 但掩盖配置错误；本节选择 forbid 在装载期显式失败。
    model_config = ConfigDict(extra="forbid")


class WorkspaceActorAccessEntry(BaseModel):
    """Workspace Actor 访问登记配置：准入状态 + 行为白名单。

    ``allowed_operations`` 使用 ``WorkspaceOperation`` 的枚举值字符串
    （如 ``"memory.read"``）；允许为空——代表可进入但未获准执行资源
    操作。准入要求 ``user_id == owner_user_id``，跨 owner 的访问不可表达。

    ``agent_id`` 省略（``None``）表示用户级记录：覆盖该用户的所有具体
    Agent，但不覆盖保留的 ``system``（system 单独显式登记）。
    """

    owner_user_id: str = Field(description="Workspace 归属 owner 用户 ID")
    workspace_id: str = Field(description="Workspace 标识")
    user_id: str = Field(description="Actor 用户 ID（必须等于 owner_user_id）")
    agent_id: str | None = Field(
        default=None,
        description="Actor Agent ID；省略表示用户级记录（覆盖所有具体 Agent）",
    )
    enabled: bool = Field(default=True, description="是否允许进入该 Workspace")
    allowed_operations: list[str] = Field(
        default_factory=list,
        description="进入后允许执行的 operation 枚举值（行为上限，可为空）",
    )

    model_config = ConfigDict(extra="forbid")


class SystemPrincipalsFile(BaseModel):
    """``configs/system_principals.yaml`` 的根模型。"""

    principals: list[SystemPrincipalAccessEntry] = Field(
        default_factory=list,
        description="System 接入登记",
    )

    model_config = ConfigDict(extra="forbid")


class WorkspaceActorsFile(BaseModel):
    """``configs/workspace_actors.yaml`` 的根模型。"""

    workspace_actors: list[WorkspaceActorAccessEntry] = Field(
        default_factory=list,
        description="Workspace Actor 访问登记（准入 + 行为白名单）",
    )

    model_config = ConfigDict(extra="forbid")


class PrincipalCredentialsFile(BaseModel):
    """``principals.secrets.yaml`` 的根模型：按 ``principal_id`` 存放生成凭据的明文。"""

    credentials: dict[str, str] = Field(
        default_factory=dict,
        description="外部来源的生成凭据，键为 principal_id",
    )

    model_config = ConfigDict(extra="forbid")


class AccessRegistration:
    """装载完成的两类登记与各自文件的内容摘要，供 System composition 构造注册表。

    摘要与内容取自同一次读取，注册表写回前以它核对文件是否被外部改动。
    """

    def __init__(
        self,
        principals: SystemPrincipalsFile,
        workspace_actors: WorkspaceActorsFile,
        *,
        principals_digest: str,
        workspace_actors_digest: str,
    ) -> None:
        self.principals = principals
        self.workspace_actors = workspace_actors
        self.principals_digest = principals_digest
        self.workspace_actors_digest = workspace_actors_digest


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def file_digest(path: Path) -> str | None:
    """文件当前内容的摘要；文件不存在时为 ``None``。"""
    if not path.exists():
        return None
    return _digest(path.read_bytes())


def _load_registration_file[ModelT: BaseModel](
    path: Path, model_type: type[ModelT], env_var: str, template: str
) -> tuple[ModelT, str]:
    """读取并校验一个登记文件，同时返回这次读取的内容摘要。

    默认路径的生效文件不存在时，先从仓库模板复制一份；显式指定的路径缺失时
    抛出 ``FileNotFoundError``。YAML/模型校验失败一律显式抛出，不静默降级。
    """
    if not path.exists():
        if os.getenv(env_var):
            raise FileNotFoundError(f"访问登记文件不存在: {path}")
        seed_from_template(path, _repo_configs_dir() / template)
    raw = path.read_bytes()
    content = yaml.safe_load(raw.decode("utf-8")) or {}
    return model_type.model_validate(content), _digest(raw)


def load_access_registration() -> AccessRegistration:
    """装载两类登记文件；装载规则见模块 docstring。"""
    principals, principals_digest = _load_registration_file(
        get_system_principals_file_path(),
        SystemPrincipalsFile,
        PRINCIPALS_FILE_ENV,
        PRINCIPALS_TEMPLATE,
    )
    workspace_actors, workspace_actors_digest = _load_registration_file(
        get_workspace_actors_file_path(),
        WorkspaceActorsFile,
        WORKSPACE_ACTORS_FILE_ENV,
        WORKSPACE_ACTORS_TEMPLATE,
    )
    return AccessRegistration(
        principals,
        workspace_actors,
        principals_digest=principals_digest,
        workspace_actors_digest=workspace_actors_digest,
    )


def write_registration_file(path: Path, document: BaseModel) -> str:
    """把登记文件或凭据文件原子写入 ``path``，返回写入内容的摘要。

    值为 ``None`` 的字段不写出：省略 ``agent_id`` 即用户级记录，省略
    ``credential_env`` 与两项收紧即不声明。
    """
    payload: dict[str, Any] = document.model_dump(mode="json", exclude_none=True)
    content = yaml.safe_dump(
        payload, allow_unicode=True, default_flow_style=False, sort_keys=False
    ).encode("utf-8")
    write_atomically(path, content)
    return _digest(content)


def load_principal_credentials(path: Path) -> PrincipalCredentialsFile:
    """读取生成凭据文件；文件不存在表示还没有生成过凭据。"""
    if not path.exists():
        return PrincipalCredentialsFile()
    content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return PrincipalCredentialsFile.model_validate(content)
