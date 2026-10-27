#!/usr/bin/env bash

# HiveMemory project services for WSL/Linux, without Docker.
#
# Usage:
#   ./scripts/hivememory-dev.sh start   [all|qdrant|backend|frontend]
#   ./scripts/hivememory-dev.sh stop    [all|qdrant|backend|frontend]
#   ./scripts/hivememory-dev.sh restart [all|qdrant|backend|frontend]
#   ./scripts/hivememory-dev.sh serve   [--build]
#   ./scripts/hivememory-dev.sh status
#   ./scripts/hivememory-dev.sh logs    [qdrant|backend|frontend|all]
#
# Two backend modes share the same processes and ports:
#   - start: development. The backend reloads on code changes unless
#     HIVEMEMORY_BACKEND_RELOAD=0, and Vite serves the frontend.
#   - serve: long-running local use. Qdrant plus a backend that never reloads
#     and serves the built frontend itself, the same runtime as the Docker image
#     without the container. Reloading would settle in-memory topics and reset
#     in-process duplicate detection on every code change.
# Only one backend runs at a time; asking for the other mode while one runs is
# refused instead of starting a second backend.
#
# Copy scripts/hivememory-dev.env.example to .hivememory/dev.env and adjust
# QDRANT_BIN before the first start.

set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${HIVEMEMORY_ENV_FILE:-$PROJECT_ROOT/.hivememory/dev.env}"

if [[ -f "$ENV_FILE" ]]; then
    # Local, user-maintained settings. This file is ignored by Git.
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

RUN_DIR="${HIVEMEMORY_RUN_DIR:-$PROJECT_ROOT/.hivememory/run}"
LOG_DIR="${HIVEMEMORY_LOG_DIR:-$PROJECT_ROOT/.hivememory/logs}"
QDRANT_BIN="${QDRANT_BIN:-$HOME/services/qdrant/qdrant}"
QDRANT_STORAGE_PATH="${QDRANT_STORAGE_PATH:-$PROJECT_ROOT/.hivememory/qdrant}"
QDRANT_HOST="${QDRANT_HOST:-127.0.0.1}"
QDRANT_HTTP_PORT="${QDRANT_HTTP_PORT:-6333}"
QDRANT_GRPC_PORT="${QDRANT_GRPC_PORT:-6334}"
BACKEND_HOST="${HIVEMEMORY_BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${HIVEMEMORY_BACKEND_PORT:-8769}"
BACKEND_RELOAD="${HIVEMEMORY_BACKEND_RELOAD:-1}"
# Seconds a backend stop waits before SIGKILL. On shutdown the backend settles
# in-memory topics into memories and waits for that generation (up to
# patchouli.shutdown.generation_wait_timeout_seconds); killing it earlier loses
# turns that import connectors already consider delivered.
BACKEND_STOP_TIMEOUT="${HIVEMEMORY_BACKEND_STOP_TIMEOUT:-60}"
FRONTEND_HOST="${HIVEMEMORY_FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${HIVEMEMORY_FRONTEND_PORT:-5173}"
FRONTEND_DIST_DIR="$PROJECT_ROOT/frontend/dist"
NODE_VERSION="${HIVEMEMORY_NODE_VERSION:-22}"
QDRANT_READY_URL="http://$QDRANT_HOST:$QDRANT_HTTP_PORT/readyz"
BACKEND_HEALTH_URL="http://$BACKEND_HOST:$BACKEND_PORT/health"
FRONTEND_URL="http://$FRONTEND_HOST:$FRONTEND_PORT"

# Set by each start_* function: 1 when this invocation launched the service,
# 0 when it reused one that was already running. A failed start rolls back only
# what this invocation launched, never services the user started earlier.
QDRANT_STARTED_BY_SCRIPT=0
BACKEND_STARTED_BY_SCRIPT=0
FRONTEND_STARTED_BY_SCRIPT=0

if [[ ! "$BACKEND_STOP_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    echo "错误：HIVEMEMORY_BACKEND_STOP_TIMEOUT 必须是正整数秒数" >&2
    exit 1
fi

mkdir -p "$RUN_DIR" "$LOG_DIR" "$QDRANT_STORAGE_PATH"

require_command() {
    local command_name="$1"
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "错误：找不到命令 $command_name" >&2
        return 1
    fi
}

pid_file() {
    printf '%s/%s.pid' "$RUN_DIR" "$1"
}

# The backend's mode is recorded next to its PID so that status, restart and
# the mode conflict check know how the running backend was started.
backend_mode_file() {
    printf '%s/backend.mode' "$RUN_DIR"
}

log_file() {
    printf '%s/%s.log' "$LOG_DIR" "$1"
}

read_pid() {
    local service="$1"
    local file
    file="$(pid_file "$service")"
    if [[ -f "$file" ]]; then
        tr -d '[:space:]' < "$file"
    fi
}

pid_is_running() {
    local pid="$1"
    [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

service_matches_pid() {
    local service="$1"
    local pid="$2"
    local command_line
    command_line="$(ps -p "$pid" -o args= 2>/dev/null || true)"

    case "$service" in
        qdrant)  [[ "$command_line" == *qdrant* ]] ;;
        backend) [[ "$command_line" == *hivememory.server.app:app* ]] ;;
        # npm normally remains as the recorded parent process while Vite runs
        # as its child, so the command line may contain npm rather than vite.
        frontend) [[ "$command_line" == *vite* || "$command_line" == *"npm run dev"* || "$command_line" == *"npm-run-script"* ]] ;;
        *) return 1 ;;
    esac
}

clear_stale_pid() {
    local service="$1"
    rm -f -- "$(pid_file "$service")"
    if [[ "$service" == 'backend' ]]; then
        rm -f -- "$(backend_mode_file)"
    fi
}

managed_pid() {
    local service="$1"
    local pid
    pid="$(read_pid "$service")"
    if [[ -z "$pid" ]]; then
        return 1
    fi
    if ! pid_is_running "$pid" || ! service_matches_pid "$service" "$pid"; then
        clear_stale_pid "$service"
        return 1
    fi
    printf '%s' "$pid"
}

# Mode of the backend this script is running: dev-reload, dev or serve.
# Empty when no backend is managed by this script.
running_backend_mode() {
    if managed_pid backend >/dev/null; then
        cat "$(backend_mode_file)" 2>/dev/null || printf 'unknown'
    fi
}

describe_backend_mode() {
    case "$1" in
        dev-reload) printf '开发（自动重载）' ;;
        dev) printf '开发（不重载）' ;;
        serve) printf 'serve（不重载，后端提供前端构建产物）' ;;
        *) printf '未知' ;;
    esac
}

reload_enabled() {
    case "${BACKEND_RELOAD,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

start_process_group() {
    local service="$1"
    local work_dir="$2"
    local output_log="$3"
    shift 3

    local service_pid_file
    local launcher_pid
    service_pid_file="$(pid_file "$service")"
    rm -f -- "$service_pid_file"

    # The inner shell writes its own PID before exec. This remains correct even
    # when setsid has to fork because its caller is a process-group leader.
    setsid bash -c '
        pid_file="$1"
        work_dir="$2"
        shift 2
        printf "%s\\n" "$$" > "$pid_file"
        cd "$work_dir" || exit 1
        exec "$@"
    ' -- "$service_pid_file" "$work_dir" "$@" >> "$output_log" 2>&1 &
    launcher_pid="$!"

    for _ in {1..40}; do
        if [[ -s "$service_pid_file" ]]; then
            local service_pid
            service_pid="$(read_pid "$service")"
            if pid_is_running "$service_pid" && service_matches_pid "$service" "$service_pid"; then
                return 0
            fi
        fi
        if ! pid_is_running "$launcher_pid" && [[ ! -s "$service_pid_file" ]]; then
            break
        fi
        sleep 0.25
    done

    echo "$service: failed to record a running process group" >&2
    if [[ -s "$service_pid_file" ]]; then
        local failed_pid
        failed_pid="$(read_pid "$service")"
        if pid_is_running "$failed_pid"; then
            kill -KILL -- "-$failed_pid" 2>/dev/null || kill -KILL "$failed_pid" 2>/dev/null || true
        fi
    fi
    rm -f -- "$service_pid_file"
    return 1
}

stop_managed_service() {
    local service="$1"
    local pid
    local child
    local poll
    local stop_timeout=5
    if [[ "$service" == backend ]]; then
        stop_timeout="$BACKEND_STOP_TIMEOUT"
    fi
    pid="$(managed_pid "$service" || true)"

    if [[ -z "$pid" ]]; then
        echo "$service: 未由本脚本运行"
        return 0
    fi

    echo "$service: stopping (pid=$pid, waiting up to ${stop_timeout}s)"
    # Services are started with setsid above. Their recorded PID is therefore
    # also the process-group ID, allowing npm/Vite and uvicorn reload children
    # to be stopped together instead of leaving orphaned listeners behind.
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    while read -r child; do
        [[ -n "$child" ]] && kill -TERM "$child" 2>/dev/null || true
    done < <(pgrep -P "$pid" 2>/dev/null || true)

    for (( poll = 0; poll < stop_timeout * 4; poll++ )); do
        if ! pid_is_running "$pid"; then
            break
        fi
        sleep 0.25
    done

    if pid_is_running "$pid"; then
        echo "$service: graceful stop timed out; sending SIGKILL" >&2
        kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        while read -r child; do
            [[ -n "$child" ]] && kill -KILL "$child" 2>/dev/null || true
        done < <(pgrep -P "$pid" 2>/dev/null || true)
    fi
    clear_stale_pid "$service"
}

wait_for_url() {
    local url="$1"
    local label="$2"
    local timeout_seconds="${3:-60}"

    for ((second = 1; second <= timeout_seconds; second++)); do
        if curl --fail --silent --show-error --max-time 2 "$url" >/dev/null 2>&1; then
            echo "$label: ready"
            return 0
        fi
        sleep 1
    done

    echo "$label: 在 ${timeout_seconds} 秒内没有就绪" >&2
    return 1
}

url_is_ready() {
    curl --fail --silent --show-error --max-time 2 "$1" >/dev/null 2>&1
}

qdrant_is_ready() {
    url_is_ready "$QDRANT_READY_URL"
}

start_qdrant() {
    local existing_pid
    QDRANT_STARTED_BY_SCRIPT=0
    existing_pid="$(managed_pid qdrant || true)"
    if [[ -n "$existing_pid" ]]; then
        echo "qdrant: already managed (pid=$existing_pid)"
        wait_for_url "$QDRANT_READY_URL" "qdrant" 60
        return 0
    fi

    if qdrant_is_ready; then
        echo "qdrant: already ready outside this script; it will not be stopped by stop"
        return 0
    fi

    if [[ ! -x "$QDRANT_BIN" ]]; then
        echo "错误：Qdrant 可执行文件不存在或不可执行：$QDRANT_BIN" >&2
        echo "请在 $ENV_FILE 中设置 QDRANT_BIN" >&2
        return 1
    fi

    echo "qdrant: starting"
    if ! start_process_group qdrant "$RUN_DIR" "$(log_file qdrant)" \
        env \
        "QDRANT__STORAGE__STORAGE_PATH=$QDRANT_STORAGE_PATH" \
        "QDRANT__SERVICE__HOST=$QDRANT_HOST" \
        "QDRANT__SERVICE__HTTP_PORT=$QDRANT_HTTP_PORT" \
        "QDRANT__SERVICE__GRPC_PORT=$QDRANT_GRPC_PORT" \
        "$QDRANT_BIN"; then
        return 1
    fi
    QDRANT_STARTED_BY_SCRIPT=1

    if ! wait_for_url "$QDRANT_READY_URL" "qdrant" 60; then
        tail -n 80 "$(log_file qdrant)" >&2 || true
        stop_managed_service qdrant
        QDRANT_STARTED_BY_SCRIPT=0
        return 1
    fi
}

load_node() {
    local nvm_dir="${NVM_DIR:-$HOME/.nvm}"
    if [[ -s "$nvm_dir/nvm.sh" ]]; then
        # nvm.sh is not written for `set -u`; relax it while loading.
        set +u
        # shellcheck disable=SC1090
        source "$nvm_dir/nvm.sh"
        if ! nvm use "$NODE_VERSION" >/dev/null; then
            set -u
            echo "错误：nvm 无法切换到 Node $NODE_VERSION" >&2
            return 1
        fi
        set -u
    fi
    require_command npm
}

# Starts the backend in the requested mode: "dev" (reload per
# HIVEMEMORY_BACKEND_RELOAD) or "serve" (no reload, serves the built frontend).
start_backend() {
    local requested="$1"
    local mode
    local serve_frontend=false
    local -a reload_args=()
    if [[ "$requested" == 'serve' ]]; then
        mode=serve
        serve_frontend=true
    elif reload_enabled; then
        mode=dev-reload
        reload_args=(--reload)
    else
        mode=dev
    fi

    BACKEND_STARTED_BY_SCRIPT=0
    local existing_pid
    existing_pid="$(managed_pid backend || true)"
    if [[ -n "$existing_pid" ]]; then
        local current
        current="$(running_backend_mode)"
        if [[ "$current" != "$mode" ]]; then
            echo "错误：后端已以$(describe_backend_mode "$current")模式运行（pid=$existing_pid），与请求的$(describe_backend_mode "$mode")模式不同；请先执行 $0 stop backend" >&2
            return 1
        fi
        echo "backend: already managed (pid=$existing_pid, $(describe_backend_mode "$mode"))"
        wait_for_url "$BACKEND_HEALTH_URL" "backend" 120
        return 0
    fi

    if url_is_ready "$BACKEND_HEALTH_URL"; then
        echo "错误：$BACKEND_HEALTH_URL 上已有不由本脚本管理的后端；请先停止它再启动" >&2
        return 1
    fi

    require_command uv || return 1
    if ! qdrant_is_ready; then
        echo "backend: 警告：Qdrant 尚未就绪（$QDRANT_READY_URL），后端可能无法连接记忆存储" >&2
    fi
    echo "backend: starting ($(describe_backend_mode "$mode"))"
    if ! start_process_group backend "$PROJECT_ROOT" "$(log_file backend)" \
        env "HIVEMEMORY_SERVE_FRONTEND=$serve_frontend" \
        uv run uvicorn hivememory.server.app:app \
        --host "$BACKEND_HOST" \
        --port "$BACKEND_PORT" \
        "${reload_args[@]}"; then
        return 1
    fi
    printf '%s\n' "$mode" > "$(backend_mode_file)"
    BACKEND_STARTED_BY_SCRIPT=1

    if ! wait_for_url "$BACKEND_HEALTH_URL" "backend" 120; then
        tail -n 100 "$(log_file backend)" >&2 || true
        stop_managed_service backend
        BACKEND_STARTED_BY_SCRIPT=0
        return 1
    fi
}

start_frontend() {
    FRONTEND_STARTED_BY_SCRIPT=0
    local existing_pid
    existing_pid="$(managed_pid frontend || true)"
    if [[ -n "$existing_pid" ]]; then
        echo "frontend: already managed (pid=$existing_pid)"
        wait_for_url "$FRONTEND_URL" "frontend" 30
        return 0
    fi

    if url_is_ready "$FRONTEND_URL"; then
        echo "frontend: already ready outside this script; it will not be stopped by stop"
        return 0
    fi

    load_node || return 1
    echo "frontend: starting"
    if ! start_process_group frontend "$PROJECT_ROOT/frontend" "$(log_file frontend)" \
        npm run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT"; then
        return 1
    fi
    FRONTEND_STARTED_BY_SCRIPT=1

    if ! wait_for_url "$FRONTEND_URL" "frontend" 30; then
        tail -n 100 "$(log_file frontend)" >&2 || true
        stop_managed_service frontend
        FRONTEND_STARTED_BY_SCRIPT=0
        return 1
    fi
}

# Stops only the services this invocation launched; services that were already
# running before the failed command keep running.
rollback_started() {
    if [[ "$FRONTEND_STARTED_BY_SCRIPT" -eq 1 ]]; then
        stop_managed_service frontend
    fi
    if [[ "$BACKEND_STARTED_BY_SCRIPT" -eq 1 ]]; then
        stop_managed_service backend
    fi
    if [[ "$QDRANT_STARTED_BY_SCRIPT" -eq 1 ]]; then
        stop_managed_service qdrant
    fi
}

# Callers run the start_* and build functions inside conditions, where bash
# disables `set -e`; every step therefore checks its own failure explicitly.
build_frontend() {
    load_node || return 1
    echo "frontend: building $FRONTEND_DIST_DIR"
    (
        cd "$PROJECT_ROOT/frontend" || exit 1
        if [[ ! -d node_modules ]]; then
            npm ci || exit 1
        fi
        npm run build
    )
}

start_all() {
    if ! start_qdrant || ! start_backend dev || ! start_frontend; then
        rollback_started
        return 1
    fi

    echo
    echo "HiveMemory 已启动（$(describe_backend_mode "$(running_backend_mode)")）"
    echo "  前端: http://$FRONTEND_HOST:$FRONTEND_PORT"
    echo "  后端: http://$BACKEND_HOST:$BACKEND_PORT"
    echo "  Qdrant: http://$QDRANT_HOST:$QDRANT_HTTP_PORT"
    echo "  日志: $LOG_DIR"
}

serve_all() {
    local build=0
    local argument
    for argument in "$@"; do
        case "$argument" in
            --build) build=1 ;;
            *) echo "未知参数：$argument" >&2; usage >&2; return 2 ;;
        esac
    done

    if [[ "$build" -eq 1 ]]; then
        build_frontend || return 1
    elif [[ ! -f "$FRONTEND_DIST_DIR/index.html" ]]; then
        echo "错误：前端构建产物不存在：$FRONTEND_DIST_DIR；请执行 $0 serve --build" >&2
        return 1
    fi

    if ! start_qdrant || ! start_backend serve; then
        rollback_started
        return 1
    fi

    echo
    echo "HiveMemory 已以 serve 模式启动（后端不重载，由后端提供前端构建产物）"
    echo "  Web UI 与后端: http://$BACKEND_HOST:$BACKEND_PORT"
    echo "  Qdrant: http://$QDRANT_HOST:$QDRANT_HTTP_PORT"
    echo "  日志: $LOG_DIR"
    if [[ "$build" -eq 0 ]]; then
        echo "  前端构建产物可能早于当前代码；需要时执行 $0 serve --build（先停止后端）"
    fi
}

start_service() {
    local target="${1:-all}"
    case "$target" in
        all) start_all ;;
        qdrant) start_qdrant ;;
        backend) start_backend dev ;;
        frontend) start_frontend ;;
        *) echo "未知服务：$target" >&2; usage >&2; return 2 ;;
    esac
}

stop_service() {
    local target="${1:-all}"
    case "$target" in
        all) stop_all ;;
        qdrant|backend|frontend) stop_managed_service "$target" ;;
        *) echo "未知服务：$target" >&2; usage >&2; return 2 ;;
    esac
}

# Restarts keep the backend's current mode: a serve backend comes back as serve.
restart_service() {
    local target="${1:-all}"
    local mode
    mode="$(running_backend_mode)"
    case "$target" in
        all)
            stop_all
            if [[ "$mode" == 'serve' ]]; then
                serve_all
            else
                start_all
            fi
            ;;
        backend)
            stop_managed_service backend
            if [[ "$mode" == 'serve' ]]; then
                start_backend serve
            else
                start_backend dev
            fi
            ;;
        qdrant|frontend)
            stop_managed_service "$target"
            start_service "$target"
            ;;
        *) echo "未知服务：$target" >&2; usage >&2; return 2 ;;
    esac
}

service_status() {
    local service="$1"
    local pid
    local url
    pid="$(managed_pid "$service" || true)"

    case "$service" in
        qdrant) url="$QDRANT_READY_URL" ;;
        backend) url="$BACKEND_HEALTH_URL" ;;
        frontend) url="$FRONTEND_URL" ;;
    esac

    if [[ -n "$pid" && "$service" == 'backend' ]]; then
        printf '%-8s running (pid=%s, %s)\n' "$service" "$pid" \
            "$(describe_backend_mode "$(running_backend_mode)")"
    elif [[ -n "$pid" ]]; then
        printf '%-8s running (pid=%s)\n' "$service" "$pid"
    elif url_is_ready "$url"; then
        printf '%-8s external (health check passed; pid not owned by this script)\n' "$service"
    else
        printf '%-8s stopped\n' "$service"
    fi
}

status_all() {
    local service
    for service in qdrant backend frontend; do
        service_status "$service"
    done

    if qdrant_is_ready; then
        echo 'qdrant: health check passed'
    else
        echo 'qdrant: health check failed or not running'
    fi
    if url_is_ready "$BACKEND_HEALTH_URL"; then
        echo 'backend: health check passed'
    else
        echo 'backend: health check failed or not running'
    fi
    if url_is_ready "$FRONTEND_URL"; then
        echo 'frontend: health check passed'
    else
        echo 'frontend: health check failed or not running'
    fi
}

show_logs() {
    local service="${1:-all}"
    if [[ "$service" == 'all' ]]; then
        echo "日志目录：$LOG_DIR"
        for service in qdrant backend frontend; do
            echo
            echo "===== $service ====="
            tail -n 40 "$(log_file "$service")" 2>/dev/null || echo '(暂无日志)'
        done
        return 0
    fi
    case "$service" in
        qdrant|backend|frontend) tail -n 100 -f "$(log_file "$service")" ;;
        *) echo "用法：$0 logs [qdrant|backend|frontend|all]" >&2; return 2 ;;
    esac
}

stop_all() {
    # Stop only processes recorded by this script. An externally managed Qdrant
    # instance is deliberately left untouched.
    stop_managed_service frontend
    stop_managed_service backend
    stop_managed_service qdrant
    echo 'HiveMemory 已停止；数据和日志仍保留。'
}

usage() {
    cat <<'EOF'
用法：
  scripts/hivememory-dev.sh start   [all|qdrant|backend|frontend]   开发模式（缺省为全部）
  scripts/hivememory-dev.sh serve   [--build]                       Qdrant + 不重载的后端，由后端提供前端
  scripts/hivememory-dev.sh stop    [all|qdrant|backend|frontend]
  scripts/hivememory-dev.sh restart [all|qdrant|backend|frontend]   保持后端当前的模式
  scripts/hivememory-dev.sh status
  scripts/hivememory-dev.sh logs    [qdrant|backend|frontend|all]

开发模式的后端默认随代码改动自动重载；在 .hivememory/dev.env 中设置
HIVEMEMORY_BACKEND_RELOAD=0 可关闭。serve --build 先构建前端再启动。
同一时间只运行一个后端：已有后端以另一种模式运行时，先 stop backend。

首次使用：
  cp scripts/hivememory-dev.env.example .hivememory/dev.env
  编辑 .hivememory/dev.env 中的 QDRANT_BIN
EOF
}

require_command curl
require_command pgrep
require_command setsid
command_name="${1:-help}"
shift || true
case "$command_name" in
    start) start_service "${1:-all}" ;;
    serve) serve_all "$@" ;;
    stop) stop_service "${1:-all}" ;;
    restart) restart_service "${1:-all}" ;;
    status) status_all ;;
    logs) show_logs "${1:-all}" ;;
    help|-h|--help) usage ;;
    *) echo "未知命令：$command_name" >&2; usage >&2; exit 2 ;;
esac
