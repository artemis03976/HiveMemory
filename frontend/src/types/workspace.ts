/**
 * 后端日志与运行时事件的类型：日志 WebSocket、事件流、连接状态与多窗口同步。
 */

export type LogLevel = 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL';

export type WorkspaceConnectionStatus =
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'error'
  | 'reconnecting';

export type RuntimeEventConnectionStatus =
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'disabled'
  | 'error';

/** 与后端 `RuntimeEventType` 一致的事件目录。 */
export type RuntimeEventType =
  | 'chat.run.created'
  | 'chat.run.status'
  | 'chat.run.cancel_requested'
  | 'chat.run.cancelled'
  | 'chat.run.completed'
  | 'chat.run.failed'
  | 'import.turn.accepted'
  | 'import.turn.duplicate'
  | 'import.turn.skipped'
  | 'import.turn.deferred'
  | 'gateway.workflow.started'
  | 'gateway.workflow.completed'
  | 'gateway.workflow.failed'
  | 'gateway.analysis.capability.completed'
  | 'agent.run.started'
  | 'agent.run.completed'
  | 'agent.run.cancelled'
  | 'agent.run.failed'
  | 'agent.context.compaction'
  | 'agent.context.budget_exceeded'
  | 'actor.operation'
  | 'topic.created'
  | 'topic.settled'
  | 'topic.deleted'
  | 'memory.task.created'
  | 'memory.task.status'
  | 'memory.task.cancel_requested'
  | 'memory.task.cancelled'
  | 'memory.task.completed'
  | 'memory.task.failed'
  | 'work.queued'
  | 'work.started'
  | 'work.retry_scheduled'
  | 'work.succeeded'
  | 'work.failed'
  | 'work.dead_lettered'
  | 'work.cancel_requested'
  | 'work.cancelled'
  | 'work.rejected'
  | 'maintenance.task.completed'
  | 'maintenance.task.failed'
  | 'system.starting'
  | 'system.ready'
  | 'system.start_failed'
  | 'system.shutting_down'
  | 'system.stopped'
  | 'system.stop_failed'
  | 'subsystem.operation.started'
  | 'subsystem.operation.completed'
  | 'subsystem.operation.failed'
  | 'event.stream.gap';

/**
 * Log entry from backend WebSocket
 * Matches the JSON format from ws://localhost:8769/api/v1/ws/logs
 */
export interface LogEntry {
  id: string; // Generated client-side (UUID)
  timestamp: string; // ISO 8601 format from backend
  level: LogLevel;
  logger: string; // e.g., "hivememory.patchouli.runtime.core"
  message: string;
  module: string; // e.g., "core.py"
  function: string; // e.g., "handle_hot"
  line: number;
  thread: string; // e.g., "MainThread"
  process: number;
  // Trace context fields for distributed tracing
  trace_id: string;
  span_name: string;
  task_type: 'foreground' | 'background';
  exception?: {
    type: string;
    message: string;
    traceback: string;
  };
  extra?: Record<string, unknown>;
}

export interface RuntimeEvent {
  event_id: string;
  sequence: number;
  event_type: RuntimeEventType;
  timestamp: string;
  trace_id: string | null;
  span_name: string | null;
  task_type: 'foreground' | 'background' | null;
  source: string | null;
  subsystem: string | null;
  component: string | null;
  severity: 'debug' | 'info' | 'warning' | 'error';
  process_id: string | null;
  agent_run_id: string | null;
  task_id: string | null;
  agent_id: string | null;
  frame_id: string | null;
  topic_id: string | null;
  atom_id: string | null;
  workspace_id: string | null;
  status: string | null;
  reason: string | null;
  message: string | null;
  data: Record<string, unknown>;
}

/**
 * WebSocket connection state
 */
export interface ConnectionState {
  status: WorkspaceConnectionStatus;
  error: string | null;
  connectedAt: number | null; // Unix timestamp
  reconnectAttempts: number;
  lastPingTime: number | null;
}

export interface RuntimeEventConnectionState {
  status: RuntimeEventConnectionStatus;
  error: string | null;
  connectedAt: number | null;
  lastEventId: string | null;
}

/**
 * Log filtering state
 */
export interface FilterState {
  logLevel: LogLevel | null; // null = show all
  loggerNamespace: string; // "" = show all, e.g., "hivememory.patchouli"
  searchText: string; // "" = no search
}

/**
 * UI state for terminal panel
 */
export interface UIState {
  autoScroll: boolean; // Auto-scroll to bottom on new logs
  isPaused: boolean; // Pause log ingestion
  maxBufferSize: number; // Max logs to keep in memory
}

/**
 * Statistics about logs and connection
 */
export interface Statistics {
  totalLogs: number; // Total logs received
  filteredCount: number; // Logs after filtering
  connectionUptime: number; // milliseconds
  logsPerSecond: number; // Rolling average
}

/**
 * Main Zustand store interface
 */
export interface WorkspaceStore {
  // State
  logs: LogEntry[];
  runtimeEvents: RuntimeEvent[];
  connection: ConnectionState;
  runtimeEventConnection: RuntimeEventConnectionState;
  filters: FilterState;
  ui: UIState;
  stats: Statistics;

  // Computed selectors
  filteredLogs: () => LogEntry[];

  // Actions - Connection Management
  connect: () => void;
  disconnect: () => void;
  reconnect: () => void;
  connectRuntimeEvents: () => void;
  disconnectRuntimeEvents: () => void;

  // Actions - Log Management
  addLog: (log: Omit<LogEntry, 'id'>) => void;
  addLogs: (logs: Omit<LogEntry, 'id'>[]) => void;
  clearLogs: () => void;
  addRuntimeEvent: (event: RuntimeEvent) => void;
  clearRuntimeEvents: () => void;

  // Actions - Filter Management
  setLogLevel: (level: LogLevel | null) => void;
  setLoggerNamespace: (namespace: string) => void;
  setSearchText: (text: string) => void;
  clearFilters: () => void;

  // Actions - UI Management
  toggleAutoScroll: () => void;
  togglePause: () => void;
  setMaxBufferSize: (size: number) => void;

  // Internal state (private, prefixed with _)
  _ws: WebSocket | null;
  _eventSource: EventSource | null;
  _broadcastChannel: BroadcastChannel | null;
  _isPrimaryWindow: boolean;
  _reconnectTimer: ReturnType<typeof setTimeout> | null;
  _statsUpdateTimer: ReturnType<typeof setInterval> | null;
  _manualDisconnecting: boolean;
}

/**
 * BroadcastChannel message types for cross-window synchronization
 */
export type BroadcastMessage =
  | { type: 'NEW_LOG'; payload: Omit<LogEntry, 'id'> }
  | { type: 'NEW_RUNTIME_EVENT'; payload: RuntimeEvent }
  | { type: 'BATCH_LOGS'; payload: Omit<LogEntry, 'id'>[] }
  | { type: 'CLEAR_LOGS' }
  | { type: 'CLEAR_RUNTIME_EVENTS' }
  | { type: 'CONNECTION_STATE'; payload: ConnectionState }
  | { type: 'RUNTIME_EVENT_CONNECTION_STATE'; payload: RuntimeEventConnectionState }
  | { type: 'FILTER_UPDATE'; payload: FilterState }
  | { type: 'UI_UPDATE'; payload: Partial<UIState> }
  | { type: 'REQUEST_SYNC' }
  | {
      type: 'FULL_SYNC';
      payload: {
        logs: LogEntry[];
        runtimeEvents: RuntimeEvent[];
        filters: FilterState;
        ui: UIState;
      }
    }
  | { type: 'PRIMARY_HEARTBEAT'; timestamp: number };
