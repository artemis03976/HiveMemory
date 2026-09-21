#!/usr/bin/env bash

# HiveMemory project services for WSL/Linux development.
#
# Usage:
#   ./scripts/hivememory-dev.sh start
#   ./scripts/hivememory-dev.sh stop
#   ./scripts/hivememory-dev.sh restart
#   ./scripts/hivememory-dev.sh status
#   ./scripts/hivememory-dev.sh logs [qdrant|backend|frontend]
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
FRONTEND_HOST="${HIVEMEMORY_FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${HIVEMEMORY_FRONTEND_PORT:-5173}"
NODE_VERSION="${HIVEMEMORY_NODE_VERSION:-22}"
QDRANT_READY_URL="http://$QDRANT_HOST:$QDRANT_HTTP_PORT/readyz"
BACKEND_HEALTH_URL="http://$BACKEND_HOST:$BACKEND_PORT/health"
FRONTEND_URL="http://$FRONTEND_HOST:$FRONTEND_PORT"

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
    pid="$(managed_pid "$service" || true)"

    if [[ -z "$pid" ]]; then
        echo "$service: 未由本脚本运行"
        return 0
    fi

    echo "$service: stopping (pid=$pid)"
    # Services are started with setsid below. Their recorded PID is therefore
    # also the process-group ID, allowing npm/Vite and uvicorn reload children
    # to be stopped together instead of leaving orphaned listeners behind.
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    while read -r child; do
        [[ -n "$child" ]] && kill -TERM "$child" 2>/dev/null || true
    done < <(pgrep -P "$pid" 2>/dev/null || true)

    for _ in {1..20}; do
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

qdrant_is_ready() {
    curl --fail --silent --show-error --max-time 2 "$QDRANT_READY_URL" >/dev/null 2>&1
}

start_qdrant() {
    local existing_pid
    existing_pid="$(managed_pid qdrant || true)"
    if [[ -n "$existing_pid" ]]; then
        echo "qdrant: already managed (pid=$existing_pid)"
        wait_for_url "$QDRANT_READY_URL" "qdrant" 60
        QDRANT_STARTED_BY_SCRIPT=0
        return 0
    fi

    if qdrant_is_ready; then
        echo "qdrant: already ready outside this script; it will not be stopped by stop"
        QDRANT_STARTED_BY_SCRIPT=0
        return 0
    fi

    require_command curl
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
        return 1
    fi
}

load_node() {
    local nvm_dir="${NVM_DIR:-$HOME/.nvm}"
    if [[ -s "$nvm_dir/nvm.sh" ]]; then
        # shellcheck disable=SC1090
        source "$nvm_dir/nvm.sh"
        nvm use "$NODE_VERSION" >/dev/null
    fi
    require_command npm
}

start_backend() {
    local existing_pid
    existing_pid="$(managed_pid backend || true)"
    if [[ -n "$existing_pid" ]]; then
        echo "backend: already managed (pid=$existing_pid)"
        wait_for_url "$BACKEND_HEALTH_URL" "backend" 120
        return 0
    fi

    require_command uv
    echo "backend: starting"
    if ! start_process_group backend "$PROJECT_ROOT" "$(log_file backend)" \
        uv run uvicorn hivememory.server.app:app \
        --host "$BACKEND_HOST" \
        --port "$BACKEND_PORT" \
        --reload; then
        return 1
    fi

    if ! wait_for_url "$BACKEND_HEALTH_URL" "backend" 120; then
        tail -n 100 "$(log_file backend)" >&2 || true
        return 1
    fi
}

start_frontend() {
    local existing_pid
    existing_pid="$(managed_pid frontend || true)"
    if [[ -n "$existing_pid" ]]; then
        echo "frontend: already managed (pid=$existing_pid)"
        wait_for_url "$FRONTEND_URL" "frontend" 30
        return 0
    fi

    if curl --fail --silent --show-error --max-time 2 "$FRONTEND_URL" >/dev/null 2>&1; then
        echo "frontend: already ready outside this script; it will not be stopped by stop"
        return 0
    fi

    load_node
    echo "frontend: starting"
    if ! start_process_group frontend "$PROJECT_ROOT/frontend" "$(log_file frontend)" \
        npm run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT"; then
        return 1
    fi

    if ! wait_for_url "$FRONTEND_URL" "frontend" 30; then
        tail -n 100 "$(log_file frontend)" >&2 || true
        return 1
    fi
}

start_all() {
    local qdrant_started=0
    QDRANT_STARTED_BY_SCRIPT=0

    start_qdrant || return 1
    qdrant_started="$QDRANT_STARTED_BY_SCRIPT"

    if ! start_backend; then
        stop_managed_service backend
        if [[ "$qdrant_started" -eq 1 ]]; then
            stop_managed_service qdrant
        fi
        return 1
    fi

    if ! start_frontend; then
        stop_managed_service frontend
        stop_managed_service backend
        if [[ "$qdrant_started" -eq 1 ]]; then
            stop_managed_service qdrant
        fi
        return 1
    fi

    echo
    echo "HiveMemory 已启动"
    echo "  前端: http://$FRONTEND_HOST:$FRONTEND_PORT"
    echo "  后端: http://$BACKEND_HOST:$BACKEND_PORT"
    echo "  Qdrant: http://$QDRANT_HOST:$QDRANT_HTTP_PORT"
    echo "  日志: $LOG_DIR"
}

service_status() {
    local service="$1"
    local pid
    pid="$(managed_pid "$service" || true)"

    if [[ -n "$pid" ]]; then
        printf '%-8s running (pid=%s)\n' "$service" "$pid"
    elif [[ "$service" == 'qdrant' ]] && qdrant_is_ready; then
        printf '%-8s external (health check passed; pid not owned by this script)\n' "$service"
    elif [[ "$service" == 'backend' ]] && curl --fail --silent --show-error --max-time 2 "$BACKEND_HEALTH_URL" >/dev/null 2>&1; then
        printf '%-8s external (health check passed; pid not owned by this script)\n' "$service"
    elif [[ "$service" == 'frontend' ]] && curl --fail --silent --show-error --max-time 2 "$FRONTEND_URL" >/dev/null 2>&1; then
        printf '%-8s external (health check passed; pid not owned by this script)\n' "$service"
    else
        printf '%-8s stopped\n' "$service"
    fi
}

status_all() {
    service_status qdrant
    service_status backend
    service_status frontend

    if qdrant_is_ready; then
        echo 'qdrant: health check passed'
    else
        echo 'qdrant: health check failed or not running'
    fi
    if curl --fail --silent --show-error --max-time 2 "$BACKEND_HEALTH_URL" >/dev/null 2>&1; then
        echo 'backend: health check passed'
    else
        echo 'backend: health check failed or not running'
    fi
    if curl --fail --silent --show-error --max-time 2 "$FRONTEND_URL" >/dev/null 2>&1; then
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
  scripts/hivememory-dev.sh start
  scripts/hivememory-dev.sh stop
  scripts/hivememory-dev.sh restart
  scripts/hivememory-dev.sh status
  scripts/hivememory-dev.sh logs [qdrant|backend|frontend|all]

首次使用：
  cp scripts/hivememory-dev.env.example .hivememory/dev.env
  编辑 .hivememory/dev.env 中的 QDRANT_BIN
EOF
}

require_command curl
require_command pgrep
require_command setsid
command_name="${1:-help}"
case "$command_name" in
    start) start_all ;;
    stop) stop_all ;;
    restart) stop_all; start_all ;;
    status) status_all ;;
    logs) show_logs "${2:-all}" ;;
    help|-h|--help) usage ;;
    *) echo "未知命令：$command_name" >&2; usage >&2; exit 2 ;;
esac
