"""访问控制配置：System 接入登记与 Workspace Actor 访问登记的文件声明与装载。

两类登记各用一个单独的 YAML 文件（v0.7.0 A1 访问边界返工第 4.2 节），
对应各自的配置所有者，``config.yaml`` 不再承载 access 段：

- ``configs/system_principals.yaml``：调用来源的接入登记（System 所有）；
- ``configs/workspace_actors.yaml``：Workspace Actor 的准入与行为白名单
  （workspace 所有）。

装载规则沿用既有约束：拒绝未知字段（``extra="forbid"``），缺失登记即
fail closed——默认路径的登记文件缺失时按空登记装载并告警，网关随后拒绝
一切认证；显式指定路径缺失或文件内容非法则显式失败。登记不包含 context
有效期：context 只随进程关闭、请求结束与 System 停止失效。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

__all__ = [
    "SystemPrincipalAccessEntry",
    "WorkspaceActorAccessEntry",
    "SystemPrincipalsFile",
    "WorkspaceActorsFile",
    "AccessRegistration",
    "load_access_registration",
    "get_system_principals_file_path",
    "get_workspace_actors_file_path",
]

# 登记文件路径可用环境变量覆盖；缺省使用仓库 configs 目录下的默认登记。
PRINCIPALS_FILE_ENV = "HIVEMEMORY_PRINCIPALS_PATH"
WORKSPACE_ACTORS_FILE_ENV = "HIVEMEMORY_WORKSPACE_ACTORS_PATH"


def _repo_configs_dir() -> Path:
    # src/hivememory/config/access.py → 仓库根目录 / configs
    return Path(__file__).resolve().parents[3] / "configs"


def get_system_principals_file_path() -> Path:
    """System 接入登记文件的路径（环境变量优先，缺省仓库默认登记）。"""
    configured = os.getenv(PRINCIPALS_FILE_ENV)
    if configured:
        return Path(configured)
    return _repo_configs_dir() / "system_principals.yaml"


def get_workspace_actors_file_path() -> Path:
    """Workspace Actor 访问登记文件的路径（环境变量优先，缺省仓库默认登记）。"""
    configured = os.getenv(WORKSPACE_ACTORS_FILE_ENV)
    if configured:
        return Path(configured)
    return _repo_configs_dir() / "workspace_actors.yaml"


class SystemPrincipalAccessEntry(BaseModel):
    """System 接入登记配置：一个受信调用来源。

    ``adapters`` 声明可经哪些 adapter 接入（HTTP 入口使用 ``http``）；
    ``allowed_user_ids`` 为可选身份解析收紧，``None`` 表示不按用户限定。
    principal 配置不授予任何 Workspace operation。
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

    # 访问控制配置拼错字段会被静默丢弃并落到 deny-by-default，方向虽安全
    # 但掩盖配置错误；本节选择 forbid 在装载期显式失败。
    model_config = ConfigDict(extra="forbid")


class WorkspaceActorAccessEntry(BaseModel):
    """Workspace Actor 访问登记配置：准入状态 + 行为白名单。

    ``allowed_operations`` 使用 ``WorkspaceOperation`` 的枚举值字符串
    （如 ``"resource.read"``）；允许为空——代表可进入但未获准执行资源
    操作。W0 兼容基线要求 ``user_id == owner_user_id``。

    ``agent_id`` 省略（``None``）表示用户级记录：覆盖该用户的所有具体
    Agent，但不覆盖保留的 ``system``（system 单独显式登记）。
    """

    owner_user_id: str = Field(description="Workspace 归属 owner 用户 ID")
    workspace_id: str = Field(description="Workspace 标识")
    user_id: str = Field(description="Actor 用户 ID（W0 基线：等于 owner_user_id）")
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


class AccessRegistration:
    """装载完成的两类登记，供 System composition 构造注册表。"""

    def __init__(
        self,
        principals: SystemPrincipalsFile,
        workspace_actors: WorkspaceActorsFile,
    ) -> None:
        self.principals = principals
        self.workspace_actors = workspace_actors


def _load_registration_file(path: Path, model_type: type, env_var: str) -> BaseModel:
    """读取并校验一个登记文件。

    显式指定的路径缺失时抛出 ``FileNotFoundError``；默认路径缺失视为
    未提供登记，按空登记 fail closed 并告警。YAML/模型校验失败一律显式
    抛出，不静默降级。
    """
    if not path.exists():
        if os.getenv(env_var):
            raise FileNotFoundError(f"访问登记文件不存在: {path}")
        logger.warning(f"访问登记文件未找到: {path}，将按空登记 fail closed")
        return model_type()
    with open(path, encoding="utf-8") as f:
        content = yaml.safe_load(f) or {}
    return model_type.model_validate(content)


def load_access_registration() -> AccessRegistration:
    """装载两类登记文件；装载规则见模块 docstring。"""
    principals = _load_registration_file(
        get_system_principals_file_path(),
        SystemPrincipalsFile,
        PRINCIPALS_FILE_ENV,
    )
    workspace_actors = _load_registration_file(
        get_workspace_actors_file_path(),
        WorkspaceActorsFile,
        WORKSPACE_ACTORS_FILE_ENV,
    )
    return AccessRegistration(principals, workspace_actors)
