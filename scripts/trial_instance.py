"""建立、运行与删除试验实例：与日常实例、开发实例都隔离的 HiveMemory 实例，供评测与对照使用。

一个试验实例是一次运行目录下的数据根 ``instance``，有自己的配置覆盖、访问登记、Qdrant 存储、
日志与端口，每次运行从空库开始。模型清单与提供商凭据取自开发 checkout 的个人配置
（``configs/models.yaml``、``configs/providers.secrets.yaml``）与 ``.env`` 中的
``HIVEMEMORY__PROVIDERS__*``，只复制或传给试验实例，不写进仓库；嵌入与重排模型直接用开发
checkout 的 ``data/model_cache``，不重复下载。Qdrant 取 ``.hivememory/dev.env`` 的
``QDRANT_BIN``，与开发时同一个版本。

试验实例只经 HTTP 使用，与日常的桌面安装版、开发实例的区别只在数据根与端口：

- 每个登记的用户有自己的默认 Workspace（请求头 ``X-User-ID``），记忆互不可见；
- 对话由一个导入来源经 ``POST /api/v1/import/turns`` 写入，凭据保存在数据根的
  ``configs/principals.secrets.yaml``，只允许声明登记的 Agent 与用户；
- 管理请求用后端首次启动时写到数据根的 ``configs/management.token``；
- 生成模型、结算方式与语言写在 ``configs/config.local.yaml``，Gateway 与其余配置沿用主配置。

子命令：

- ``create``：建立运行目录与数据根，写入配置覆盖与访问登记；运行目录已存在时拒绝；
- ``serve``：在前台运行 Qdrant 与后端，直到 Ctrl+C 或 ``stop``，退出时先请后端正常关闭；
- ``stop``：请正在 ``serve`` 的实例退出；
- ``status``：打印实例的规格，以及端口上是否有服务；
- ``delete``：删除整个运行目录；实例还在运行时拒绝。

评测脚本（例如 ``halumem_eval.py``）直接调用 ``create`` 与 ``run_instance``，一次运行内自己
启动与停止实例。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml
from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE = REPO_ROOT / "data" / "model_cache"
DEV_ENV = REPO_ROOT / ".hivememory" / "dev.env"

# Qdrant 不支持长路径：记忆文件位于数据根之下约 140 个字符处，数据根本身不能更长。
MAX_DATA_ROOT_CHARS = 100

# 日常实例（桌面安装版）的端口；试验实例占用它们时，harness 的真实流量会写进实验数据。
RESERVED_PORTS = frozenset({8769, 6333, 6334})

DEFAULT_PORTS = (18769, 16333, 16334)
IMPORT_PRINCIPAL_ID = "trial:importer"
MAIN_WORKSPACE_ID = "main_workspace"

QDRANT_READY_SECONDS = 60
# 后端就绪要等推理模型加载，首次启动可能要几分钟。
BACKEND_READY_SECONDS = 600
BACKEND_STOP_SECONDS = 180

SPEC_FILE = "instance.json"
STOP_FILE = "stop"


@dataclass(frozen=True)
class TrialSpec:
    """一个试验实例的规格；建立时写进 ``instance.json``，之后不再改变。"""

    users: tuple[str, ...]
    agent_id: str
    settle_mode: str
    generation_model: str
    language: str
    backend_port: int
    qdrant_http_port: int
    qdrant_grpc_port: int
    log_level: str


@dataclass(frozen=True)
class InstanceEndpoints:
    """运行中的试验实例：地址与两种凭据。"""

    base_url: str
    management_token: str
    import_token: str
    principal_id: str
    agent_id: str


class TrialInstanceError(RuntimeError):
    """试验实例无法建立、启动或停止；信息直接给出原因。"""


def data_root(run_dir: Path) -> Path:
    return run_dir.resolve() / "instance"


def load_spec(run_dir: Path) -> TrialSpec:
    raw = json.loads((run_dir / SPEC_FILE).read_text(encoding="utf-8"))
    return TrialSpec(**{**raw["spec"], "users": tuple(raw["spec"]["users"])})


def repo_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


# ---------------------------------------------------------------- create


def create(run_dir: Path, spec: TrialSpec) -> None:
    """建立运行目录与数据根；运行目录已存在时拒绝，保证每个实例从空库开始。"""
    if run_dir.exists():
        raise TrialInstanceError(f"运行目录已存在：{run_dir}")
    root = data_root(run_dir)
    if len(str(root)) > MAX_DATA_ROOT_CHARS:
        raise TrialInstanceError(
            f"数据根超过 {MAX_DATA_ROOT_CHARS} 个字符（Qdrant 的路径限制）：{root}"
        )
    for port in (spec.backend_port, spec.qdrant_http_port, spec.qdrant_grpc_port):
        if port in RESERVED_PORTS:
            raise TrialInstanceError(f"端口 {port} 属于日常实例，试验实例不能使用")
    if not MODEL_CACHE.is_dir():
        raise TrialInstanceError(f"没有找到模型缓存：{MODEL_CACHE}")
    qdrant_bin()

    configs = root / "configs"
    configs.mkdir(parents=True)
    (root / "logs").mkdir()
    _write_yaml(configs / "config.local.yaml", _config_overlay(root, spec))
    _write_yaml(configs / "system_principals.yaml", _principals(spec))
    _write_yaml(configs / "workspace_actors.yaml", _workspace_actors(spec))
    _write_yaml(
        configs / "principals.secrets.yaml",
        {"credentials": {IMPORT_PRINCIPAL_ID: secrets.token_urlsafe(32)}},
    )
    for name in ("models.yaml", "providers.secrets.yaml"):
        source = REPO_ROOT / "configs" / name
        if source.is_file():
            shutil.copyfile(source, configs / name)
    record = {
        "created_at": datetime.now(UTC).isoformat(),
        "repo_commit": repo_commit(),
        "spec": asdict(spec),
    }
    (run_dir / SPEC_FILE).write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_yaml(path: Path, document: object) -> None:
    path.write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _config_overlay(root: Path, spec: TrialSpec) -> dict[str, object]:
    """覆盖主配置中与试验有关的项；只改生成一侧，Gateway 与其余配置沿用主配置。"""
    logs = root / "logs"
    return {
        "i18n": {"language": spec.language},
        "logging": {
            "level": spec.log_level,
            "file_path": str(logs / "backend.log"),
            "event_file_path": str(logs / "runtime-events.jsonl"),
            "console_output": False,
        },
        "shared": {"embedding": {"default": {"cache_dir": str(MODEL_CACHE)}}},
        "patchouli": {
            "llm": {"model_id": spec.generation_model},
            "storage": {"port": spec.qdrant_http_port, "grpc_port": spec.qdrant_grpc_port},
            "generation": {"settle_mode": spec.settle_mode},
            "retrieval": {"reranker": {"cache_dir": str(MODEL_CACHE)}},
        },
    }


def _principals(spec: TrialSpec) -> dict[str, object]:
    """HTTP server 与系统自身两条来源与模板相同，另加一个只能导入的来源。"""
    return {
        "principals": [
            {
                "principal_id": "hivememory:http-server",
                "kind": "http-server",
                "enabled": True,
                "adapters": ["http"],
                "allow_system_actor": True,
            },
            {
                "principal_id": "hivememory:system",
                "kind": "local-process",
                "enabled": True,
                "adapters": ["local"],
                "allow_system_actor": True,
            },
            {
                "principal_id": IMPORT_PRINCIPAL_ID,
                "kind": "import",
                "enabled": True,
                "adapters": ["import"],
                "allowed_user_ids": list(spec.users),
                "allowed_agent_ids": [spec.agent_id],
            },
        ]
    }


def _workspace_actors(spec: TrialSpec) -> dict[str, object]:
    """每个用户一条用户级记录（导入只需要准入，不登记 actor 操作）与一条 system 记录。"""
    records: list[dict[str, object]] = []
    for user in spec.users:
        identity = {"owner_user_id": user, "workspace_id": MAIN_WORKSPACE_ID, "user_id": user}
        records.append({**identity, "allowed_operations": []})
        records.append({**identity, "agent_id": "system"})
    return {"workspace_actors": records}


# ---------------------------------------------------------------- run


def qdrant_bin() -> Path:
    path = dotenv_values(DEV_ENV).get("QDRANT_BIN") if DEV_ENV.is_file() else None
    if not path or not Path(path).is_file():
        raise TrialInstanceError(f"{DEV_ENV} 中没有可用的 QDRANT_BIN")
    return Path(path)


def ensure_ports_free(spec: TrialSpec) -> None:
    """端口被占用时直接失败：不连接、不复用别的进程。"""
    for port in (spec.backend_port, spec.qdrant_http_port, spec.qdrant_grpc_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise TrialInstanceError(f"端口 {port} 已被占用")


def _provider_env() -> dict[str, str]:
    """开发 checkout 的 ``.env`` 中的提供商凭据；试验实例以数据根为工作目录，读不到它。"""
    values = dotenv_values(REPO_ROOT / ".env") if (REPO_ROOT / ".env").is_file() else {}
    return {
        key: value
        for key, value in values.items()
        if key.startswith("HIVEMEMORY__PROVIDERS__") and value is not None
    }


def _spawn(command: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> subprocess.Popen:
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    with log.open("ab") as output:
        return subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            creationflags=flags,
            start_new_session=sys.platform != "win32",
        )


def _wait_ready(url: str, process: subprocess.Popen, label: str, seconds: int) -> None:
    deadline = time.monotonic() + seconds
    with httpx.Client(trust_env=False, timeout=5) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise TrialInstanceError(f"{label} 启动后退出，退出码 {process.returncode}")
            with contextlib.suppress(httpx.HTTPError):
                if client.get(url).status_code == 200:
                    return
            time.sleep(1)
    raise TrialInstanceError(f"{label} 在 {seconds} 秒内没有就绪：{url}")


def _stop_backend(process: subprocess.Popen) -> None:
    """请后端正常退出：它会结算话题并保存未完成的生成工作；超时才强制结束。"""
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=BACKEND_STOP_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _stop_qdrant(process: subprocess.Popen) -> None:
    """Qdrant 在确认写入前先写 WAL，直接结束也能在下次启动时恢复。"""
    if process.poll() is None:
        process.terminate()
        process.wait()


@contextlib.contextmanager
def run_instance(run_dir: Path) -> Iterator[InstanceEndpoints]:
    """启动试验实例，交出地址与凭据；退出时先停后端、再停 Qdrant。"""
    spec = load_spec(run_dir)
    root = data_root(run_dir)
    ensure_ports_free(spec)
    logs = root / "logs"
    qdrant_env = {
        **os.environ,
        "QDRANT__STORAGE__STORAGE_PATH": str(root / "qdrant"),
        "QDRANT__SERVICE__HOST": "127.0.0.1",
        "QDRANT__SERVICE__HTTP_PORT": str(spec.qdrant_http_port),
        "QDRANT__SERVICE__GRPC_PORT": str(spec.qdrant_grpc_port),
    }
    qdrant = _spawn([str(qdrant_bin())], cwd=root, env=qdrant_env, log=logs / "qdrant.log")
    backend: subprocess.Popen | None = None
    try:
        _wait_ready(
            f"http://127.0.0.1:{spec.qdrant_http_port}/readyz",
            qdrant,
            "Qdrant",
            QDRANT_READY_SECONDS,
        )
        backend_env = {
            **os.environ,
            **_provider_env(),
            "HIVEMEMORY_DATA_ROOT": str(root),
            "HIVEMEMORY_SERVE_FRONTEND": "false",
        }
        command = [
            sys.executable,
            "-m",
            "uvicorn",
            "hivememory.server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(spec.backend_port),
        ]
        backend = _spawn(command, cwd=root, env=backend_env, log=logs / "backend.stdout.log")
        base_url = f"http://127.0.0.1:{spec.backend_port}"
        _wait_ready(f"{base_url}/health/ready", backend, "后端", BACKEND_READY_SECONDS)
        configs = root / "configs"
        credentials = yaml.safe_load((configs / "principals.secrets.yaml").read_text("utf-8"))
        yield InstanceEndpoints(
            base_url=base_url,
            management_token=(configs / "management.token").read_text(encoding="utf-8").strip(),
            import_token=credentials["credentials"][IMPORT_PRINCIPAL_ID],
            principal_id=IMPORT_PRINCIPAL_ID,
            agent_id=spec.agent_id,
        )
    finally:
        if backend is not None:
            _stop_backend(backend)
        _stop_qdrant(qdrant)


def serve(run_dir: Path) -> None:
    """前台运行，直到 Ctrl+C 或 ``stop`` 写下停止文件。"""
    stop_file = run_dir / STOP_FILE
    stop_file.unlink(missing_ok=True)
    with run_instance(run_dir) as endpoints:
        print(f"ready base_url={endpoints.base_url}", flush=True)
        try:
            while not stop_file.exists():
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    stop_file.unlink(missing_ok=True)
    print("stopped", flush=True)


def is_running(spec: TrialSpec) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex(("127.0.0.1", spec.backend_port)) == 0


# ---------------------------------------------------------------- cli


def parse_ports(text: str) -> tuple[int, int, int]:
    parts = [int(part) for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("端口写成 后端,Qdrant HTTP,Qdrant gRPC")
    return parts[0], parts[1], parts[2]


def spec_from_args(args: argparse.Namespace, users: list[str]) -> TrialSpec:
    backend, qdrant_http, qdrant_grpc = args.ports
    return TrialSpec(
        users=tuple(users),
        agent_id=args.agent_id,
        settle_mode=args.settle_mode,
        generation_model=args.generation_model,
        language=args.language,
        backend_port=backend,
        qdrant_http_port=qdrant_http,
        qdrant_grpc_port=qdrant_grpc,
        log_level=args.log_level,
    )


def add_spec_arguments(parser: argparse.ArgumentParser) -> None:
    """试验实例规格的参数；评测脚本的 ``run`` 也用同一组参数。"""
    parser.add_argument("--agent-id", default="trial-assistant", help="导入时声明的 Agent")
    parser.add_argument("--settle-mode", choices=["one_shot", "librarian"], default="one_shot")
    parser.add_argument(
        "--generation-model", default="deepseek-chat", help="models.yaml 中的模型 id"
    )
    parser.add_argument("--language", choices=["zh", "en"], default="en")
    parser.add_argument(
        "--ports", type=parse_ports, default=DEFAULT_PORTS, help="后端,Qdrant HTTP,Qdrant gRPC"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="后端日志级别",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="建立、运行与删除隔离的试验实例")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("create", help="建立运行目录与数据根")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--users", required=True, help="逗号分隔的用户 id，每个用户一个 Workspace")
    add_spec_arguments(p)
    for name, text in (
        ("serve", "在前台运行实例"),
        ("stop", "请正在运行的实例退出"),
        ("status", "打印规格与运行状态"),
        ("delete", "删除运行目录"),
    ):
        commands.add_parser(name, help=text).add_argument("run_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "create":
            create(args.run_dir, spec_from_args(args, args.users.split(",")))
            print(f"created {data_root(args.run_dir)}")
        elif args.command == "serve":
            serve(args.run_dir)
        elif args.command == "stop":
            (args.run_dir / STOP_FILE).touch()
        elif args.command == "status":
            spec = load_spec(args.run_dir)
            print(json.dumps({**asdict(spec), "running": is_running(spec)}, indent=2))
        elif args.command == "delete":
            if is_running(load_spec(args.run_dir)):
                raise TrialInstanceError("实例还在运行，先 stop")
            shutil.rmtree(args.run_dir)
    except TrialInstanceError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
