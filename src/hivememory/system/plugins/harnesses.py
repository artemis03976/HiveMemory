"""HiveMemory 插件支持的外部 harness，以及在各个 harness 中安装、卸载插件的命令。

插件目录在仓库与安装版中布局相同：``integrations/`` 下是本地 marketplace 的声明，
``integrations/hivememory/`` 是插件本身。Claude Code 与 Codex 从本地 marketplace
把插件复制进自己的缓存；pi 与 dsh 桌面端链接插件目录、原地加载。

每个 harness 的安装都可以重复执行：已经装过时改为更新，指向别处的旧安装先移除，
同一个 harness 里不会同时留下两份 HiveMemory 插件。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

#: 插件的包名，也是 marketplace 中的插件名。
PLUGIN_NAME = "hivememory-connector"
#: ``integrations/.claude-plugin/marketplace.json`` 声明的本地 marketplace 名。
MARKETPLACE_NAME = "hivememory-local"
PLUGIN_ID = f"{PLUGIN_NAME}@{MARKETPLACE_NAME}"

#: 插件目录：与后端源码同在一个布局中，即仓库根或安装版的 ``resources/backend``。
INTEGRATIONS_DIR = Path(__file__).resolve().parents[4] / "integrations"

#: 单条插件命令的时限。dsh 经 pnpm 安装，最慢。
COMMAND_TIMEOUT_SECONDS = 180.0

type HarnessId = Literal["claude-code", "codex", "pi", "dsh"]
type CommandFailureReason = Literal["command_failed", "command_timeout", "unexpected_output"]


@dataclass(frozen=True)
class CommandResult:
    """一条插件命令的结果；``returncode`` 为 ``None`` 表示超时被结束。"""

    argv: tuple[str, ...]
    returncode: int | None
    stdout: str
    stderr: str


type CommandRunner = Callable[[Sequence[str]], CommandResult]


class PluginCommandError(Exception):
    """插件命令失败：退出码非零、超时，或输出不是预期的格式。"""

    def __init__(
        self, result: CommandResult, reason: CommandFailureReason = "command_failed"
    ) -> None:
        super().__init__(f"{reason}: {' '.join(result.argv)}")
        self.result = result
        self.reason = reason


def run_command(argv: Sequence[str]) -> CommandResult:
    """运行一条插件命令并取得全部输出，不弹出控制台窗口，也不读取标准输入。"""
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NO_WINDOW
    try:
        completed = subprocess.run(
            list(argv),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=COMMAND_TIMEOUT_SECONDS,
            creationflags=creationflags,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(tuple(argv), None, _text(exc.stdout), _text(exc.stderr))
    return CommandResult(tuple(argv), completed.returncode, completed.stdout, completed.stderr)


def _text(output: str | bytes | None) -> str:
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output or ""


def _checked(run: CommandRunner, argv: Sequence[str]) -> CommandResult:
    result = run(argv)
    if result.returncode is None:
        raise PluginCommandError(result, "command_timeout")
    if result.returncode != 0:
        raise PluginCommandError(result)
    return result


def _json_output(result: CommandResult) -> Any:
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise PluginCommandError(result, "unexpected_output") from exc


def _same_path(left: str | Path, right: str | Path) -> bool:
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))


# ---- Claude Code ----


def _install_claude_code(run: CommandRunner, claude: str, integrations: Path) -> None:
    listed = _json_output(_checked(run, [claude, "plugin", "marketplace", "list", "--json"]))
    existing = next((m for m in listed if m.get("name") == MARKETPLACE_NAME), None)
    if existing is not None and not _same_path(existing.get("path", ""), integrations):
        _checked(run, [claude, "plugin", "marketplace", "remove", MARKETPLACE_NAME])
        existing = None
    if existing is None:
        _checked(run, [claude, "plugin", "marketplace", "add", str(integrations)])
    else:
        _checked(run, [claude, "plugin", "marketplace", "update", MARKETPLACE_NAME])
    if _claude_code_has_plugin(run, claude):
        _checked(run, [claude, "plugin", "update", PLUGIN_ID])
    else:
        _checked(run, [claude, "plugin", "install", PLUGIN_ID])


def _uninstall_claude_code(run: CommandRunner, claude: str, integrations: Path) -> None:
    # 保留插件的数据目录：连接器的导入进度在其中，重新安装后不必从头补发。
    if _claude_code_has_plugin(run, claude):
        _checked(run, [claude, "plugin", "uninstall", PLUGIN_ID, "--keep-data"])
    listed = _json_output(_checked(run, [claude, "plugin", "marketplace", "list", "--json"]))
    if any(m.get("name") == MARKETPLACE_NAME for m in listed):
        _checked(run, [claude, "plugin", "marketplace", "remove", MARKETPLACE_NAME])


def _claude_code_has_plugin(run: CommandRunner, claude: str) -> bool:
    installed = _json_output(_checked(run, [claude, "plugin", "list", "--json"]))
    return any(p.get("id") == PLUGIN_ID for p in installed)


# ---- Codex ----


def _install_codex(run: CommandRunner, codex: str, integrations: Path) -> None:
    existing = _codex_marketplace_root(run, codex)
    if existing is not None and not _same_path(existing, integrations):
        _checked(run, [codex, "plugin", "marketplace", "remove", MARKETPLACE_NAME])
        existing = None
    if existing is None:
        _checked(run, [codex, "plugin", "marketplace", "add", str(integrations)])
    # 已经安装时再次执行会把新版本复制进 Codex 的缓存。
    _checked(run, [codex, "plugin", "add", PLUGIN_ID])


def _uninstall_codex(run: CommandRunner, codex: str, integrations: Path) -> None:
    if _codex_marketplace_root(run, codex) is None:
        return
    _checked(run, [codex, "plugin", "remove", PLUGIN_ID])
    _checked(run, [codex, "plugin", "marketplace", "remove", MARKETPLACE_NAME])


def _codex_marketplace_root(run: CommandRunner, codex: str) -> str | None:
    listed = _json_output(_checked(run, [codex, "plugin", "marketplace", "list", "--json"]))
    for marketplace in listed.get("marketplaces", []):
        if marketplace.get("name") == MARKETPLACE_NAME:
            return str(marketplace.get("root", ""))
    return None


# ---- pi ----


def _install_pi(run: CommandRunner, pi: str, integrations: Path) -> None:
    plugin_dir = integrations / "hivememory"
    installed_here = False
    for source in _pi_hivememory_sources(run, pi):
        if _same_path(source, plugin_dir):
            installed_here = True
        else:
            # 另一处的 HiveMemory 插件（例如 checkout）会让同名工具注册两次。
            _checked(run, [pi, "remove", source])
    if not installed_here:
        _checked(run, [pi, "install", str(plugin_dir)])


def _uninstall_pi(run: CommandRunner, pi: str, integrations: Path) -> None:
    plugin_dir = integrations / "hivememory"
    for source in _pi_hivememory_sources(run, pi):
        if _same_path(source, plugin_dir):
            _checked(run, [pi, "remove", source])


def _pi_hivememory_sources(run: CommandRunner, pi: str) -> list[str]:
    """``pi list`` 中指向 HiveMemory 插件目录的本地包。

    ``pi list`` 把每个包的来源缩进两格列出，下面再缩进四格写出解析后的位置。
    """
    sources = []
    for line in _checked(run, [pi, "list"]).stdout.splitlines():
        if not line.startswith("  ") or line.startswith("    "):
            continue
        source = line.strip()
        if _is_hivememory_plugin_dir(Path(source)):
            sources.append(source)
    return sources


def _is_hivememory_plugin_dir(path: Path) -> bool:
    try:
        manifest = json.loads((path / "package.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(manifest, dict) and manifest.get("name") == PLUGIN_NAME


# ---- dsh 桌面端 ----

#: dsh 桌面端独占的 profile，插件只能在桌面端退出时安装到这里。
_DSH_PROFILE = "desktop"


def _install_dsh(run: CommandRunner, dsh: str, integrations: Path) -> None:
    # dsh 把参数转交给 pnpm：按路径添加即链接安装，指向别处的旧链接随之替换。
    plugin_dir = integrations / "hivememory"
    _checked(run, [dsh, "plugin", "--profile", _DSH_PROFILE, "add", str(plugin_dir)])


def _uninstall_dsh(run: CommandRunner, dsh: str, integrations: Path) -> None:
    _checked(run, [dsh, "plugin", "--profile", _DSH_PROFILE, "remove", PLUGIN_NAME])


def _dsh_desktop_command() -> str | None:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        return None
    command = (
        Path(local_app_data)
        / "Programs"
        / "DeepSeek Harness"
        / "resources"
        / "runtime"
        / "cli"
        / "bin"
        / "dsh.cmd"
    )
    return str(command) if command.is_file() else None


# ---- 支持的 harness ----


@dataclass(frozen=True)
class SupportedHarness:
    """一个支持的外部 harness。

    ``principal_id`` 是它接入 HiveMemory 的来源；``default_agent_id`` 是插件在没有
    设置 ``HIVEMEMORY_AGENT_ID`` 时声明的 Agent。同一个 harness 可以以多个 Agent
    运行，来源能声明哪些 Agent 由接入登记决定，不由这张表决定。
    """

    harness_id: HarnessId
    name: str
    principal_id: str
    default_agent_id: str
    #: harness 一侧读取接入凭据的环境变量。
    credential_env: str
    #: 安装时复制插件（Claude Code、Codex）；否则链接插件目录、原地加载。
    copies_plugin: bool
    #: 插件的 hook 以 Python 运行连接器，需要知道用哪个解释器。
    runs_python_hooks: bool
    locate: Callable[[], str | None]
    install: Callable[[CommandRunner, str, Path], None]
    uninstall: Callable[[CommandRunner, str, Path], None]
    manual_install: Callable[[Path], list[str]]


SUPPORTED_HARNESSES: tuple[SupportedHarness, ...] = (
    SupportedHarness(
        harness_id="claude-code",
        name="Claude Code",
        principal_id="harness:claude-code",
        default_agent_id="claude_code",
        credential_env="HIVEMEMORY_CLAUDE_CODE_TOKEN",
        copies_plugin=True,
        runs_python_hooks=True,
        locate=lambda: shutil.which("claude"),
        install=_install_claude_code,
        uninstall=_uninstall_claude_code,
        manual_install=lambda integrations: [
            f'claude plugin marketplace add "{integrations}"',
            f"claude plugin install {PLUGIN_ID}",
        ],
    ),
    SupportedHarness(
        harness_id="codex",
        name="Codex",
        principal_id="harness:codex",
        default_agent_id="codex",
        credential_env="HIVEMEMORY_CODEX_TOKEN",
        copies_plugin=True,
        runs_python_hooks=True,
        locate=lambda: shutil.which("codex"),
        install=_install_codex,
        uninstall=_uninstall_codex,
        manual_install=lambda integrations: [
            f'codex plugin marketplace add "{integrations}"',
            f"codex plugin add {PLUGIN_ID}",
        ],
    ),
    SupportedHarness(
        harness_id="pi",
        name="pi",
        principal_id="harness:pi",
        default_agent_id="pi",
        credential_env="HIVEMEMORY_PI_TOKEN",
        copies_plugin=False,
        runs_python_hooks=False,
        locate=lambda: shutil.which("pi"),
        install=_install_pi,
        uninstall=_uninstall_pi,
        manual_install=lambda integrations: [f'pi install "{integrations / "hivememory"}"'],
    ),
    SupportedHarness(
        harness_id="dsh",
        name="DeepSeek Harness",
        principal_id="harness:dsh",
        default_agent_id="dsh",
        credential_env="HIVEMEMORY_DSH_TOKEN",
        copies_plugin=False,
        runs_python_hooks=False,
        locate=_dsh_desktop_command,
        install=_install_dsh,
        uninstall=_uninstall_dsh,
        # 第一条用于 dsh 桌面端（PowerShell，桌面端退出之后执行），第二条用于命令行版。
        manual_install=lambda integrations: [
            '& "$env:LOCALAPPDATA\\Programs\\DeepSeek Harness\\resources\\runtime\\cli\\bin'
            f'\\dsh.cmd" plugin --profile desktop add "{integrations / "hivememory"}"',
            f'dsh plugin --profile <profile> add "{integrations / "hivememory"}"',
        ],
    ),
)


def harness_by_id(harness_id: HarnessId) -> SupportedHarness:
    return next(h for h in SUPPORTED_HARNESSES if h.harness_id == harness_id)


__all__ = [
    "COMMAND_TIMEOUT_SECONDS",
    "INTEGRATIONS_DIR",
    "MARKETPLACE_NAME",
    "PLUGIN_ID",
    "PLUGIN_NAME",
    "SUPPORTED_HARNESSES",
    "CommandResult",
    "CommandFailureReason",
    "CommandRunner",
    "HarnessId",
    "PluginCommandError",
    "SupportedHarness",
    "harness_by_id",
    "run_command",
]
