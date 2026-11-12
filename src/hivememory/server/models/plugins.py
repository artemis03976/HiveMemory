"""插件管理接口的请求/响应模型：外部 harness 的插件状态、安装与卸载的结果。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from hivememory.system.plugins import HarnessId, PluginFailure, PluginStatus


class PluginStatusResponse(BaseModel):
    """一个支持的 harness 的插件状态；“已安装”以 HiveMemory 一侧的接入登记为准。"""

    harness: HarnessId
    name: str
    principal_id: str
    default_agent_id: str = Field(description="没有设置 HIVEMEMORY_AGENT_ID 时插件声明的 Agent")
    install_mode: Literal["copy", "link"] = Field(
        description="copy：harness 复制插件，升级后要重新安装；link：链接插件目录、原地加载",
    )
    available: bool = Field(description="能否在这里代为安装：插件目录存在，并且运行在 Windows 上")
    command: str | None = Field(description="找到的 harness 命令；null 表示没有找到")
    installed: bool
    needs_update: bool = Field(description="复制安装的插件在升级之后还没有重新安装")
    allowed_agent_ids: list[str] | None = Field(
        description="来源可声明的 Agent；未安装或不限定时为 null",
    )
    credential_env: str = Field(description="harness 一侧读取接入凭据的环境变量")
    manual_commands: list[str] = Field(description="手动安装插件的命令")

    @classmethod
    def from_status(cls, status: PluginStatus) -> "PluginStatusResponse":
        harness = status.harness
        return cls(
            harness=harness.harness_id,
            name=harness.name,
            principal_id=harness.principal_id,
            default_agent_id=harness.default_agent_id,
            install_mode="copy" if harness.copies_plugin else "link",
            available=status.available,
            command=status.command,
            installed=status.installed,
            needs_update=status.needs_update,
            allowed_agent_ids=(
                sorted(status.allowed_agent_ids) if status.allowed_agent_ids is not None else None
            ),
            credential_env=harness.credential_env,
            manual_commands=status.manual_commands,
        )


class PluginListResponse(BaseModel):
    integrations_dir: str = Field(description="插件目录：harness 从这里安装插件")
    version: str = Field(description="插件与后端的版本")
    harnesses: list[PluginStatusResponse]


class PluginFailureResponse(BaseModel):
    reason: Literal[
        "unavailable", "harness_missing", "command_failed", "command_timeout", "unexpected_output"
    ]
    command: str | None = Field(description="失败的那条命令")
    output_tail: list[str] = Field(description="命令输出的最后几行")

    @classmethod
    def from_failure(cls, failure: PluginFailure) -> "PluginFailureResponse":
        return cls(
            reason=failure.reason,
            command=" ".join(failure.command) if failure.command is not None else None,
            output_tail=list(failure.output_tail),
        )


class PluginActionResponse(BaseModel):
    """安装或卸载之后的状态；``failure`` 不为 null 时说明哪一步没有完成。"""

    status: PluginStatusResponse
    failure: PluginFailureResponse | None


class PluginInstallRequest(BaseModel):
    run_plugin_commands: bool = Field(
        default=True,
        description="false：用户已经手动装好插件，只登记接入、交付凭据",
    )

    model_config = ConfigDict(extra="forbid")
