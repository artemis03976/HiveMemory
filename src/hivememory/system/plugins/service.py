"""插件服务：为支持的外部 harness 安装、卸载 HiveMemory 插件，并维护它的接入。

一次安装依次完成：
1. 用 harness 自己的命令安装插件；
2. 登记 harness 的来源，只使用生成的凭据；
3. 缺省 Agent 还没有访问记录时，为它补一条；
4. 把凭据（以及 hook 用的解释器）写进当前用户的环境变量。

命令失败时什么都不登记，不留下没有作用对象的来源。卸载撤销插件、来源、凭据与写入
的环境变量；Agent 的访问记录保留：它属于 Agent，不属于某个 harness，同一个 Agent
也可能经别的来源接入。

同一时刻只执行一个安装、卸载或更新：它们都会改动 harness 的配置与接入登记。插件
命令在线程中运行；接入登记与访问记录的编辑留在事件循环中，与认证、授权的查询不会
并发交错。
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from hivememory.config.files import write_atomically
from hivememory.core.access import WorkspaceOperation
from hivememory.i18n import Language
from hivememory.system.access import EXTERNAL_ADAPTERS
from hivememory.system.plugins.harnesses import (
    SUPPORTED_HARNESSES,
    CommandFailureReason,
    CommandRunner,
    HarnessId,
    HarnessSettingsError,
    PluginCommandError,
    SupportedHarness,
    harness_by_id,
    run_command,
)

if TYPE_CHECKING:
    from hivememory.core.access import WorkingActor
    from hivememory.system.access import SystemActorAccessRegistry
    from hivememory.system.plugins.environment import WindowsUserEnvironment
    from hivememory.workspace.capability.root.access import RootAccessService

logger = logging.getLogger(__name__)

#: harness 读取的 skill 在插件目录中的位置：四个 harness 的插件清单都指向 ``skills/``。
#: 它不受版本控制，后端启动时按语言设置，从插件目录 ``skill-languages`` 下对应语言的正文写入。
SKILL_PATH = Path("skills") / "hivememory" / "SKILL.md"

#: hook 以 Python 运行连接器的插件从这个变量取得解释器，没有时找 PATH 上的 Python。
PYTHON_ENV = "HIVEMEMORY_PYTHON"

#: 插件经 MCP 提供的四项操作；补给缺省 Agent 的访问记录只开放这些。
PLUGIN_OPERATIONS = frozenset(
    {
        WorkspaceOperation.MEMORY_SEARCH,
        WorkspaceOperation.MEMORY_READ,
        WorkspaceOperation.MEMORY_WRITE,
        WorkspaceOperation.MEMORY_UPDATE,
    }
)

#: 失败时交给页面的命令输出行数。
_OUTPUT_TAIL_LINES = 12

type FailureReason = (
    Literal["unavailable", "harness_missing", "settings_unreadable"] | CommandFailureReason
)


@dataclass(frozen=True)
class PluginStatus:
    """一个 harness 的插件状态；“已安装”以 HiveMemory 一侧的来源登记为准。"""

    harness: SupportedHarness
    #: 能否在这里代为安装：插件目录存在，并且能写入当前用户的环境变量（Windows）。
    available: bool
    #: 找到的 harness 命令；``None`` 表示没有找到。
    command: str | None
    installed: bool
    #: 复制安装的插件在升级之后还没有重新安装。
    needs_update: bool
    allowed_agent_ids: frozenset[str] | None
    manual_commands: list[str]


@dataclass(frozen=True)
class PluginFailure:
    reason: FailureReason
    command: tuple[str, ...] | None = None
    output_tail: tuple[str, ...] = ()


@dataclass(frozen=True)
class PluginActionResult:
    status: PluginStatus
    failure: PluginFailure | None = None


class PluginInstallationService:
    def __init__(
        self,
        *,
        principal_registry: SystemActorAccessRegistry,
        root_access: RootAccessService,
        integrations_dir: Path,
        install_record_path: Path,
        version: str,
        python_executable: str,
        user_environment: WindowsUserEnvironment | None,
        run: CommandRunner = run_command,
    ) -> None:
        """``user_environment`` 为 ``None`` 表示不在 Windows 上：凭据无法交给 harness，
        不能代为安装，卸载只删除接入登记。"""
        self._registry = principal_registry
        self._root_access = root_access
        self._integrations = integrations_dir
        self._record_path = install_record_path
        self._version = version
        self._python = python_executable
        self._environment = user_environment
        self._run = run
        self._lock = asyncio.Lock()

    # ---- 查询 ----

    @property
    def integrations_dir(self) -> Path:
        return self._integrations

    @property
    def version(self) -> str:
        """插件与后端同为这个版本：插件目录随程序发布。"""
        return self._version

    def statuses(self) -> list[PluginStatus]:
        return [self._status(harness) for harness in SUPPORTED_HARNESSES]

    def _status(self, harness: SupportedHarness) -> PluginStatus:
        entry = self._registry.entry_for(harness.principal_id)
        recorded = self._read_record().get(harness.harness_id)
        return PluginStatus(
            harness=harness,
            available=self._available(),
            command=harness.locate(),
            installed=entry is not None,
            needs_update=(
                entry is not None
                and harness.copies_plugin
                and recorded is not None
                and recorded != self._version
            ),
            allowed_agent_ids=entry.allowed_agent_ids if entry is not None else None,
            manual_commands=harness.manual_install(self._integrations),
        )

    def _available(self) -> bool:
        return self._environment is not None and (self._integrations / "hivememory").is_dir()

    # ---- 安装 ----

    async def install(
        self, harness_id: HarnessId, *, actor: WorkingActor, run_plugin_commands: bool = True
    ) -> PluginActionResult:
        """安装插件并登记接入；已经安装时重新安装，原有登记与凭据沿用。

        ``run_plugin_commands`` 为 ``False`` 用于用户已经按页面给出的命令手动装好插件：
        只登记接入、交付凭据，不运行插件命令。
        """
        harness = harness_by_id(harness_id)
        async with self._lock:
            environment = self._environment
            if environment is None or not self._available():
                return PluginActionResult(self._status(harness), PluginFailure("unavailable"))
            if run_plugin_commands:
                failure = await self._run_plugin_install(harness)
                if failure is not None:
                    return PluginActionResult(self._status(harness), failure)
            credential = self._register(harness)
            self._ensure_default_agent_record(harness, actor)
            # 写用户环境变量会广播一次设置变更，可能等待几秒，放到线程中执行。
            if credential is not None:
                await asyncio.to_thread(environment.set, harness.credential_env, credential)
            if harness.runs_python_hooks:
                await asyncio.to_thread(environment.set, PYTHON_ENV, self._python)
            self._write_record(harness.harness_id, self._version)
            logger.info("插件已安装: harness=%s", harness.harness_id)
            return PluginActionResult(self._status(harness))

    async def _run_plugin_install(self, harness: SupportedHarness) -> PluginFailure | None:
        command = harness.locate()
        if command is None:
            return PluginFailure("harness_missing")
        try:
            await asyncio.to_thread(harness.install, self._run, command, self._integrations)
        except PluginCommandError as exc:
            logger.warning(
                "插件命令失败: harness=%s reason=%s returncode=%s",
                harness.harness_id,
                exc.reason,
                exc.result.returncode,
            )
            return _failure_from(exc)
        except HarnessSettingsError as exc:
            return _settings_failure(harness, exc)
        return None

    def _register(self, harness: SupportedHarness) -> str | None:
        """登记 harness 的来源，返回要交给 harness 一侧的凭据。

        来源不声明凭据环境变量，后端只认生成的凭据，harness 一侧的环境变量不会覆盖
        它。已经登记过时沿用原有登记：生成的凭据原样交付；凭据来自后端环境变量的
        旧登记由用户自己管理，返回 ``None``；登记了却没有可用凭据时生成一个。
        """
        entry = self._registry.entry_for(harness.principal_id)
        if entry is None:
            _, credential = self._registry.register(
                harness.principal_id,
                adapters=EXTERNAL_ADAPTERS,
                allowed_agent_ids=frozenset({harness.default_agent_id}),
            )
            return credential
        if entry.credential_source == "env":
            return None
        if entry.credential_source == "generated":
            return entry.credential
        _, credential = self._registry.rotate_credential(harness.principal_id)
        return credential

    def _ensure_default_agent_record(self, harness: SupportedHarness, actor: WorkingActor) -> None:
        """缺省 Agent 没有精确记录时补一条；已有记录（可能被用户调整过）不改动。"""
        records = self._root_access.list_records(target_workspace=actor.workspace, actor=actor)
        if any(record.agent_id == harness.default_agent_id for record in records):
            return
        self._root_access.put_record(
            harness.default_agent_id,
            enabled=True,
            allowed_operations=PLUGIN_OPERATIONS,
            target_workspace=actor.workspace,
            actor=actor,
        )

    # ---- 卸载 ----

    async def uninstall(self, harness_id: HarnessId) -> PluginActionResult:
        """卸载插件并删除接入；插件命令失败时，接入照样删除。"""
        harness = harness_by_id(harness_id)
        async with self._lock:
            failure = None
            if self._available():
                failure = await self._run_plugin_uninstall(harness)
            entry = self._registry.entry_for(harness.principal_id)
            if entry is not None:
                self._registry.delete(harness.principal_id)
            if self._environment is not None:
                # 凭据来自后端环境变量的旧登记，harness 一侧的变量也由用户自己管理。
                if entry is not None and entry.credential_source != "env":
                    await asyncio.to_thread(self._environment.delete, harness.credential_env)
                if not self._python_hooks_still_installed():
                    await asyncio.to_thread(self._environment.delete, PYTHON_ENV)
            self._write_record(harness.harness_id, None)
            logger.info("插件已卸载: harness=%s", harness.harness_id)
            return PluginActionResult(self._status(harness), failure)

    async def _run_plugin_uninstall(self, harness: SupportedHarness) -> PluginFailure | None:
        command = harness.locate()
        if command is None:
            return PluginFailure("harness_missing")
        try:
            await asyncio.to_thread(harness.uninstall, self._run, command, self._integrations)
        except PluginCommandError as exc:
            logger.warning(
                "插件卸载命令失败: harness=%s reason=%s returncode=%s",
                harness.harness_id,
                exc.reason,
                exc.result.returncode,
            )
            return _failure_from(exc)
        except HarnessSettingsError as exc:
            return _settings_failure(harness, exc)
        return None

    def _python_hooks_still_installed(self) -> bool:
        return any(
            harness.runs_python_hooks and self._registry.entry_for(harness.principal_id) is not None
            for harness in SUPPORTED_HARNESSES
        )

    # ---- skill 的语言 ----

    def write_skill(self, language: Language) -> bool:
        """把 ``language`` 对应的 skill 正文写到 harness 读取的位置；返回是否改写。

        后端每次启动时调用，先于复制安装插件的自动更新：升级之后重新安装的插件带上的就是
        新的 skill，链接安装的插件重新打开即读到。插件目录不存在时（Docker）不写；内容相同
        时不改写。不依赖代为安装是否打开：开发实例也写 checkout 中的这一份，供测试插件使用。
        """
        plugin_dir = self._integrations / "hivememory"
        if not plugin_dir.is_dir():
            return False
        content = (plugin_dir / "skill-languages" / language.value / "SKILL.md").read_bytes()
        target = plugin_dir / SKILL_PATH
        if target.is_file() and target.read_bytes() == content:
            return False
        write_atomically(target, content)
        logger.info("插件 skill 已按语言设置写入: language=%s", language.value)
        return True

    # ---- 升级之后 ----

    async def update_copied_plugins(self) -> None:
        """重新安装版本落后的复制安装插件，让升级后的 harness 读到新版本。

        只处理经插件服务安装、记录了版本的 harness；链接安装的插件原地读取新文件，
        不需要处理。失败只写日志，插件页把它标为需要更新。
        """
        async with self._lock:
            if not self._available():
                return
            record = self._read_record()
            for harness in SUPPORTED_HARNESSES:
                recorded = record.get(harness.harness_id)
                if not harness.copies_plugin or recorded in (None, self._version):
                    continue
                if self._registry.entry_for(harness.principal_id) is None:
                    continue
                if await self._run_plugin_install(harness) is None:
                    self._write_record(harness.harness_id, self._version)
                    logger.info(
                        "插件已随升级更新: harness=%s from_version=%s to_version=%s",
                        harness.harness_id,
                        recorded,
                        self._version,
                    )

    # ---- 安装版本记录 ----

    def _read_record(self) -> dict[str, str]:
        try:
            data = json.loads(self._record_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        return {str(key): str(value) for key, value in data.items()}

    def _write_record(self, harness_id: str, version: str | None) -> None:
        record = self._read_record()
        if version is None:
            record.pop(harness_id, None)
        else:
            record[harness_id] = version
        content = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
        write_atomically(self._record_path, content.encode("utf-8"))


def hook_python_executable() -> str:
    """hook 用的解释器：运行后端的解释器目录中带控制台的 ``python.exe``。

    桌面端以不带控制台的 ``pythonw.exe`` 运行后端；hook 要从标准输出交回结果，
    所以取同目录的 ``python.exe``。
    """
    executable = Path(sys.executable)
    console = executable.with_name("python.exe")
    return str(console if console.is_file() else executable)


def _settings_failure(harness: SupportedHarness, exc: HarnessSettingsError) -> PluginFailure:
    """harness 的设置文件读不懂：插件命令已经执行，设置文件没有改动；页面给出文件位置。"""
    logger.warning(
        "harness 设置文件读不懂，没有改动: harness=%s reason=settings_unreadable path=%s",
        harness.harness_id,
        exc.path,
    )
    return PluginFailure("settings_unreadable", output_tail=(str(exc.path),))


def _failure_from(exc: PluginCommandError) -> PluginFailure:
    output = "\n".join(part for part in (exc.result.stdout, exc.result.stderr) if part)
    lines = [line for line in output.splitlines() if line.strip()]
    return PluginFailure(exc.reason, exc.result.argv, tuple(lines[-_OUTPUT_TAIL_LINES:]))


__all__ = [
    "PLUGIN_OPERATIONS",
    "PYTHON_ENV",
    "PluginActionResult",
    "PluginFailure",
    "PluginInstallationService",
    "PluginStatus",
    "hook_python_executable",
]
