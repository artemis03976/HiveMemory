/**
 * Kernel Terminal Type Definitions
 *
 * Types for the kernel terminal panel with real-time log streaming,
 * WebSocket connection management, and cross-window synchronization.
 */

export type LogLevel = 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL';

export type ConnectionStatus =
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'error'
  | 'reconnecting';

/**
 * Log entry from backend WebSocket
 * Matches the JSON format from ws://localhost:8000/api/v1/ws/logs
 */
export interface LogEntry {
  id: string; // Generated client-side (UUID)
  timestamp: string; // ISO 8601 format from backend
  level: LogLevel;
  logger: string; // e.g., "hivememory.patchouli.kernel"
  message: string;
  module: string; // e.g., "kernel.py"
  function: string; // e.g., "handle_hot"
  line: number;
  thread: string; // e.g., "MainThread"
  process: number;
  exception?: {
    type: string;
    message: string;
    traceback: string;
  };
  extra?: Record<string, any>;
}

/**
 * WebSocket connection state
 */
export interface ConnectionState {
  status: ConnectionStatus;
  error: string | null;
  connectedAt: number | null; // Unix timestamp
  reconnectAttempts: number;
  lastPingTime: number | null;
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
export interface KernelStore {
  // State
  logs: LogEntry[];
  connection: ConnectionState;
  filters: FilterState;
  ui: UIState;
  stats: Statistics;

  // Computed selectors
  filteredLogs: () => LogEntry[];

  // Actions - Connection Management
  connect: () => void;
  disconnect: () => void;
  reconnect: () => void;

  // Actions - Log Management
  addLog: (log: Omit<LogEntry, 'id'>) => void;
  addLogs: (logs: Omit<LogEntry, 'id'>[]) => void;
  clearLogs: () => void;

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
  _broadcastChannel: BroadcastChannel | null;
  _isPrimaryWindow: boolean;
  _reconnectTimer: NodeJS.Timeout | null;
  _statsUpdateTimer: NodeJS.Timeout | null;
}

/**
 * BroadcastChannel message types for cross-window synchronization
 */
export type BroadcastMessage =
  | { type: 'NEW_LOG'; payload: Omit<LogEntry, 'id'> }
  | { type: 'BATCH_LOGS'; payload: Omit<LogEntry, 'id'>[] }
  | { type: 'CLEAR_LOGS' }
  | { type: 'CONNECTION_STATE'; payload: ConnectionState }
  | { type: 'FILTER_UPDATE'; payload: FilterState }
  | { type: 'UI_UPDATE'; payload: Partial<UIState> }
  | { type: 'REQUEST_SYNC' }
  | {
      type: 'FULL_SYNC';
      payload: {
        logs: LogEntry[];
        filters: FilterState;
        ui: UIState;
      }
    }
  | { type: 'PRIMARY_HEARTBEAT'; timestamp: number };
