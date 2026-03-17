/**
 * KernelTerminal - Independent kernel log terminal component
 *
 * Designed to be fully decoupled from any panel/layout.
 * Consumes useKernelStore directly — no props needed.
 * Can be rendered inside L4 panel, a standalone route, or an Electron detached window.
 */

import { useEffect, useRef, useCallback } from 'react';
import {
  Search,
  Trash2,
  Pause,
  Play,
  ArrowDownToLine,
  ArrowUpFromLine,
  RefreshCw,
  Circle,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useKernelStore } from '@/stores/kernelStore';
import type { LogLevel, LogEntry, ConnectionStatus } from '@/types/kernel';

// ─── Color Maps ───────────────────────────────────────────────

const LEVEL_STYLES: Record<LogLevel, string> = {
  DEBUG: 'text-muted-foreground',
  INFO: 'text-magic-water',
  WARNING: 'text-magic-metal',
  ERROR: 'text-magic-fire',
  CRITICAL: 'text-magic-fire font-bold',
};

const STATUS_DOT: Record<ConnectionStatus, string> = {
  disconnected: 'text-muted-foreground',
  connecting: 'text-magic-metal',
  connected: 'text-magic-wood',
  error: 'text-magic-fire',
  reconnecting: 'text-magic-metal animate-pulse',
};

const STATUS_LABEL: Record<ConnectionStatus, string> = {
  disconnected: 'Disconnected',
  connecting: 'Connecting...',
  connected: 'Connected',
  error: 'Error',
  reconnecting: 'Reconnecting...',
};

// ─── Sub-components ───────────────────────────────────────────

/** Single log row */
function LogRow({ log }: { log: LogEntry }) {
  const ts = new Date(log.timestamp).toLocaleTimeString();

  return (
    <div className="flex gap-2 px-3 py-0.5 hover:bg-white/5 rounded group leading-5">
      <span className="text-muted-foreground shrink-0 select-none">{ts}</span>
      <span className={cn('shrink-0 w-[70px] text-right', LEVEL_STYLES[log.level])}>
        {log.level}
      </span>
      <span className="text-primary/60 shrink-0">{log.logger.split('.').pop()}</span>
      <span className="text-foreground break-all">{log.message}</span>
      {log.exception && (
        <span className="text-magic-fire shrink-0 opacity-0 group-hover:opacity-100" title={log.exception.type}>
          !!
        </span>
      )}
    </div>
  );
}

/** Toolbar icon button */
function ToolbarButton({
  onClick,
  title,
  active,
  children,
}: {
  onClick: () => void;
  title: string;
  active?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      className={cn(
        'p-1 rounded transition-colors',
        active
          ? 'bg-primary/20 text-primary'
          : 'text-muted-foreground hover:bg-white/10 hover:text-foreground'
      )}
    >
      {children}
    </button>
  );
}

/** Empty state placeholder */
function EmptyState({ status }: { status: ConnectionStatus }) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-muted-foreground gap-2 select-none">
      <span className="text-lg opacity-40">{'>'}_</span>
      <span className="text-[11px]">
        {status === 'connected'
          ? 'Waiting for backend activity...'
          : status === 'error'
            ? 'Connection failed. Click reconnect to retry.'
            : 'Connecting to kernel...'}
      </span>
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────

export function KernelTerminal() {
  const scrollRef = useRef<HTMLDivElement>(null);

  // State selectors (granular to avoid unnecessary re-renders)
  const logs = useKernelStore((s) => s.filteredLogs());
  const connection = useKernelStore((s) => s.connection);
  const filters = useKernelStore((s) => s.filters);
  const ui = useKernelStore((s) => s.ui);

  // Actions
  const setLogLevel = useKernelStore((s) => s.setLogLevel);
  const setSearchText = useKernelStore((s) => s.setSearchText);
  const clearLogs = useKernelStore((s) => s.clearLogs);
  const toggleAutoScroll = useKernelStore((s) => s.toggleAutoScroll);
  const togglePause = useKernelStore((s) => s.togglePause);
  const reconnect = useKernelStore((s) => s.reconnect);

  // Auto-scroll
  useEffect(() => {
    if (ui.autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs.length, ui.autoScroll]);

  const handleLevelChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      setLogLevel((e.target.value as LogLevel) || null);
    },
    [setLogLevel]
  );

  const handleSearchChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setSearchText(e.target.value);
    },
    [setSearchText]
  );

  return (
    <div className="flex flex-col h-full bg-black/20 font-mono text-xs">
      {/* ── Toolbar ── */}
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-white/10 bg-black/30">
        {/* Connection status */}
        <Circle
          className={cn('w-2.5 h-2.5 fill-current', STATUS_DOT[connection.status])}
        />
        <span className="text-[11px] text-muted-foreground mr-1">
          {STATUS_LABEL[connection.status]}
        </span>

        {connection.status === 'error' && (
          <button
            onClick={reconnect}
            className="p-1 rounded hover:bg-white/10 transition-colors"
            title="Reconnect"
          >
            <RefreshCw className="w-3 h-3 text-muted-foreground" />
          </button>
        )}

        {/* Spacer */}
        <div className="flex-1" />

        {/* Action buttons */}
        <ToolbarButton
          onClick={togglePause}
          title={ui.isPaused ? 'Resume' : 'Pause'}
          active={ui.isPaused}
        >
          {ui.isPaused ? <Play className="w-3 h-3" /> : <Pause className="w-3 h-3" />}
        </ToolbarButton>

        <ToolbarButton
          onClick={toggleAutoScroll}
          title={ui.autoScroll ? 'Unlock scroll' : 'Lock to bottom'}
          active={ui.autoScroll}
        >
          {ui.autoScroll ? (
            <ArrowDownToLine className="w-3 h-3" />
          ) : (
            <ArrowUpFromLine className="w-3 h-3" />
          )}
        </ToolbarButton>

        <ToolbarButton onClick={clearLogs} title="Clear logs">
          <Trash2 className="w-3 h-3" />
        </ToolbarButton>
      </div>

      {/* ── Filter bar ── */}
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-white/10 bg-black/20">
        <select
          value={filters.logLevel || ''}
          onChange={handleLevelChange}
          className="text-[11px] bg-black/30 border border-white/10 rounded px-1.5 py-0.5 text-foreground outline-none focus:border-primary/50"
        >
          <option value="">ALL</option>
          <option value="DEBUG">DEBUG</option>
          <option value="INFO">INFO</option>
          <option value="WARNING">WARN</option>
          <option value="ERROR">ERROR</option>
          <option value="CRITICAL">CRIT</option>
        </select>

        <div className="flex-1 relative">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3 h-3 text-muted-foreground pointer-events-none" />
          <input
            type="text"
            placeholder="Filter logs..."
            value={filters.searchText}
            onChange={handleSearchChange}
            className="w-full text-[11px] bg-black/30 border border-white/10 rounded pl-6 pr-2 py-0.5 text-foreground placeholder:text-muted-foreground outline-none focus:border-primary/50"
          />
        </div>
      </div>

      {/* ── Log viewport ── */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto custom-scrollbar py-1"
      >
        {logs.length === 0 ? (
          <EmptyState status={connection.status} />
        ) : (
          logs.map((log) => <LogRow key={log.id} log={log} />)
        )}
      </div>

      {/* ── Status bar ── */}
      <div className="flex items-center justify-between px-3 py-1 border-t border-white/10 bg-black/30 text-[11px] text-muted-foreground select-none">
        <span>
          {logs.length} entries
          {(filters.logLevel || filters.searchText) && ' (filtered)'}
        </span>
        <div className="flex items-center gap-3">
          {ui.isPaused && <span className="text-magic-metal">PAUSED</span>}
          {ui.autoScroll && <span>AUTO-SCROLL</span>}
        </div>
      </div>
    </div>
  );
}
