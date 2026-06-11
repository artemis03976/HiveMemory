import { useEffect, useRef, useCallback, useMemo, useState } from 'react';
import { Circle, Play, Pause, ArrowDownToLine, ArrowUpFromLine, Trash2, Search, RefreshCw, ChevronRight, User, Activity, Layers, Radio } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { useKernelStore } from '@/stores/kernelStore';
import type { LogLevel, LogEntry, KernelConnectionStatus, RuntimeEventConnectionStatus, RuntimeEvent, SpanGroup, TraceGroup } from '@/types/kernel';

const STATUS_DOT: Record<KernelConnectionStatus, string> = {
  disconnected: 'text-slate-500',
  connecting: 'text-magic-metal',
  connected: 'text-magic-wood',
  error: 'text-magic-fire',
  reconnecting: 'text-magic-metal animate-pulse',
};

const STATUS_LABEL: Record<KernelConnectionStatus, string> = {
  disconnected: 'Disconnected',
  connecting: 'Connecting...',
  connected: 'Connected',
  error: 'Error',
  reconnecting: 'Reconnecting...',
};

const EVENT_STATUS_DOT: Record<RuntimeEventConnectionStatus, string> = {
  disconnected: 'text-slate-500',
  connecting: 'text-magic-metal animate-pulse',
  connected: 'text-magic-wood',
  disabled: 'text-slate-500',
  error: 'text-magic-fire',
};

const EVENT_STATUS_LABEL: Record<RuntimeEventConnectionStatus, string> = {
  disconnected: 'Events disconnected',
  connecting: 'Events connecting...',
  connected: 'Events connected',
  disabled: 'Events disabled',
  error: 'Events error',
};

type KernelView = 'logs' | 'events';

const LEVEL_STYLES: Record<LogLevel, string> = {
  DEBUG: 'text-slate-500 bg-slate-500/10',
  INFO: 'text-magic-water bg-magic-water/10',
  WARNING: 'text-magic-metal bg-magic-metal/10',
  ERROR: 'text-magic-fire bg-magic-fire/10',
  CRITICAL: 'text-magic-fire bg-magic-fire/20 font-bold',
};

function LogRow({ log }: { log: LogEntry }) {
  const date = new Date(log.timestamp);
  const ts = `${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}:${date.getSeconds().toString().padStart(2, '0')}.${date.getMilliseconds().toString().padStart(3, '0')}`;

  return (
    <div className="flex flex-col gap-1.5 px-3 py-2 hover:bg-white/5 rounded-lg group transition-colors border border-transparent hover:border-white/5 my-0.5">
      {/* Metadata Row */}
      <div className="flex items-center gap-2.5 select-none flex-wrap">
        <span className={`px-1.5 py-0.5 rounded text-[9px] font-bold tracking-wider leading-none uppercase shrink-0 ${LEVEL_STYLES[log.level]}`}>
          {log.level}
        </span>
        <span className="text-slate-500/60 font-mono text-[10px] shrink-0">{ts}</span>
        <span className="text-primary/60 text-[10px] font-mono px-1.5 py-0.5 rounded bg-primary/5 truncate max-w-[200px] ml-auto" title={log.logger}>
          {log.logger.split('.').pop()}
        </span>
      </div>

      {/* Message Row */}
      <div className="text-slate-300 font-mono text-[11px] whitespace-pre-wrap wrap-break-words leading-relaxed pl-0.5">
        {log.message}
      </div>
      
      {/* Exception */}
      {log.exception && (
        <div className="mt-1 p-2 rounded bg-magic-fire/10 border border-magic-fire/20">
          <div className="text-magic-fire font-bold text-[10px] mb-1">{log.exception.type}</div>
          <div className="text-magic-fire/80 text-[10px] font-mono whitespace-pre-wrap wrap-break-words">
            {log.exception.message}
          </div>
        </div>
      )}
    </div>
  );
}

function ToolbarButton({ onClick, title, active, children }: { onClick: () => void, title: string, active?: boolean, children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      title={title}
      className={`p-1.5 rounded-lg transition-colors ${
        active
          ? 'bg-primary/20 text-primary ghost-border'
          : 'text-slate-400 hover:bg-white/10 hover:text-white'
      }`}
    >
      {children}
    </button>
  );
}

function EventRow({ event }: { event: RuntimeEvent }) {
  const date = new Date(event.timestamp);
  const ts = `${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}:${date.getSeconds().toString().padStart(2, '0')}.${date.getMilliseconds().toString().padStart(3, '0')}`;
  const scope = event.task_id || event.generation_id || event.agent_run_id || event.trace_id || 'runtime';
  const severityClass =
    event.severity === 'error'
      ? 'text-magic-fire bg-magic-fire/10'
      : event.severity === 'warning'
        ? 'text-magic-metal bg-magic-metal/10'
        : 'text-primary bg-primary/10';

  return (
    <div className="grid grid-cols-[auto_1fr] gap-x-2 gap-y-1 px-3 py-2 hover:bg-white/5 rounded-lg border border-transparent hover:border-white/5 my-0.5">
      <span className="text-slate-500/60 font-mono text-[10px] pt-0.5">{event.sequence}</span>
      <div className="min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className={`px-1.5 py-0.5 rounded text-[9px] font-bold uppercase shrink-0 ${severityClass}`}>
            {event.event_type}
          </span>
          {event.status && (
            <span className="px-1.5 py-0.5 rounded bg-white/5 text-slate-300 text-[9px] uppercase">
              {event.status}
            </span>
          )}
          <span className="text-slate-500/60 font-mono text-[10px] shrink-0">{ts}</span>
          <span className="ml-auto text-slate-500 text-[10px] font-mono truncate max-w-[220px]" title={scope}>
            {scope}
          </span>
        </div>
        {(event.message || event.reason) && (
          <div className="mt-1 text-slate-300 font-mono text-[11px] whitespace-pre-wrap wrap-break-words">
            {event.message || event.reason}
          </div>
        )}
        <div className="mt-1 flex items-center gap-2 text-[10px] text-slate-500 font-mono flex-wrap">
          {event.subsystem && <span>{event.subsystem}</span>}
          {event.component && <span>{event.component}</span>}
          {event.topic_id && <span>topic:{event.topic_id}</span>}
        </div>
      </div>
    </div>
  );
}

/** Empty state placeholder */
function EmptyState({ status }: { status: KernelConnectionStatus }) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-slate-500 gap-2 select-none">
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

function SpanBlock({ span, onToggle }: { span: SpanGroup; onToggle: () => void }) {
  const duration = new Date(span.last_timestamp).getTime() - new Date(span.first_timestamp).getTime();

  return (
    <div className="relative ml-1 my-1 group/span">
      {/* Thread line */}
      <div className="absolute left-[11px] top-7 bottom-0 w-px bg-white/5 group-hover/span:bg-primary/20 transition-colors" />

      <button
        onClick={onToggle}
        className="flex items-center gap-2 px-2 py-1 hover:bg-white/5 rounded w-full text-left transition-colors relative z-10"
      >
        <motion.div
          animate={{ rotate: span.collapsed ? 0 : 90 }}
          transition={{ duration: 0.2 }}
          className="text-slate-500 group-hover/span:text-primary transition-colors"
        >
          <ChevronRight className="w-3 h-3" />
        </motion.div>
        
        <Activity className="w-3 h-3 text-primary/60 shrink-0" />

        <span className="text-primary/80 font-mono text-[11px] font-medium tracking-tight truncate">
          {span.span_name}
        </span>
        
        <span className="ml-auto text-slate-500 text-[10px] flex items-center gap-2 shrink-0 font-mono">
          <span>{span.logs.length} log{span.logs.length > 1 ? 's' : ''}</span>
          <span className="w-0.5 h-0.5 rounded-full bg-slate-600" />
          <span>{duration}ms</span>
        </span>
      </button>

      <AnimatePresence initial={false}>
        {!span.collapsed && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <div className="ml-6 mt-0.5 space-y-px pb-1">
              {span.logs.map(log => (
                <LogRow key={log.id} log={log} />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function TraceBlock({
  trace,
  onToggleTrace,
  onToggleSpan
}: {
  trace: TraceGroup;
  onToggleTrace: () => void;
  onToggleSpan: (span_name: string) => void;
}) {
  const isForeground = trace.task_type === 'foreground';
  const TaskIcon = isForeground ? User : RefreshCw;

  return (
    <div className="my-2 border border-white/5 bg-black/20 rounded-md overflow-hidden transition-colors hover:border-white/10">
      <button
        onClick={onToggleTrace}
        className={`flex items-center gap-2.5 px-3 py-1.5 w-full text-left transition-colors ${
          isForeground ? 'bg-primary/5 hover:bg-primary/10' : 'bg-white/2 hover:bg-white/4'
        }`}
      >
        <motion.div
          animate={{ rotate: trace.collapsed ? 0 : 90 }}
          transition={{ duration: 0.2 }}
        >
          <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
        </motion.div>
        
        <div className={`p-1 rounded ${isForeground ? 'bg-primary/20 text-primary' : 'bg-white/10 text-slate-300'}`}>
          <TaskIcon className="w-3 h-3" />
        </div>
        
        <div className="flex flex-col">
          <span className="text-slate-300 font-mono text-[11px] font-medium">
            {trace.trace_id}
          </span>
          <span className="text-[9px] text-slate-500 uppercase tracking-wider leading-none mt-0.5">
            {trace.task_type} TASK
          </span>
        </div>

        <div className="ml-auto text-[10px] text-slate-500 flex items-center gap-1.5 font-mono">
          <Layers className="w-3 h-3 opacity-70" />
          {trace.spans.size} span{trace.spans.size > 1 ? 's' : ''}
        </div>
      </button>

      <AnimatePresence initial={false}>
        {!trace.collapsed && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <div className="p-1.5 pl-3 border-t border-white/5 bg-black/10">
              {Array.from(trace.spans.values()).map(span => (
                <SpanBlock
                  key={span.span_name}
                  span={span}
                  onToggle={() => onToggleSpan(span.span_name)}
                />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function KernelTerminalTab() {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [view, setView] = useState<KernelView>('logs');

  // State selectors
  const connection = useKernelStore((s) => s.connection);
  const runtimeEventConnection = useKernelStore((s) => s.runtimeEventConnection);
  const filters = useKernelStore((s) => s.filters);
  const ui = useKernelStore((s) => s.ui);
  const traceGroups = useKernelStore((s) => s.traceGroups);
  const runtimeEvents = useKernelStore((s) => s.runtimeEvents);

  const allLogs = useKernelStore((s) => s.logs);
  const logs = useMemo(() => {
    return allLogs.filter((log) => {
      if (filters.logLevel && log.level !== filters.logLevel) return false;
      if (filters.loggerNamespace && !log.logger.startsWith(filters.loggerNamespace)) return false;

      if (filters.searchText) {
        const search = filters.searchText.toLowerCase();
        return (
          log.message.toLowerCase().includes(search) ||
          log.logger.toLowerCase().includes(search)
        );
      }
      return true;
    });
  }, [allLogs, filters.logLevel, filters.loggerNamespace, filters.searchText]);

  // Actions
  const connect = useKernelStore((s) => s.connect);
  const disconnect = useKernelStore((s) => s.disconnect);
  const setLogLevel = useKernelStore((s) => s.setLogLevel);
  const setSearchText = useKernelStore((s) => s.setSearchText);
  const clearLogs = useKernelStore((s) => s.clearLogs);
  const clearRuntimeEvents = useKernelStore((s) => s.clearRuntimeEvents);
  const toggleAutoScroll = useKernelStore((s) => s.toggleAutoScroll);
  const togglePause = useKernelStore((s) => s.togglePause);
  const reconnect = useKernelStore((s) => s.reconnect);
  const toggleTraceCollapse = useKernelStore((s) => s.toggleTraceCollapse);
  const toggleSpanCollapse = useKernelStore((s) => s.toggleSpanCollapse);

  // Auto-scroll
  useEffect(() => {
    if (ui.autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs.length, runtimeEvents.length, ui.autoScroll]);

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
    <div className="flex flex-col h-full bg-black/20 font-mono text-[11px] rounded-xl ghost-border overflow-hidden">
      {/* ── Toolbar ── */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-white/5 bg-black/40">
        <Circle className={`w-2.5 h-2.5 fill-current ${view === 'logs' ? STATUS_DOT[connection.status] : EVENT_STATUS_DOT[runtimeEventConnection.status]}`} />
        <span className="text-slate-400 mr-1">
          {view === 'logs' ? STATUS_LABEL[connection.status] : EVENT_STATUS_LABEL[runtimeEventConnection.status]}
        </span>
        
        {view === 'logs' && connection.error && (
          <span className="text-[11px] text-magic-fire/80 truncate max-w-[200px]" title={connection.error}>
            {connection.error}
          </span>
        )}

        {view === 'events' && runtimeEventConnection.error && (
          <span className="text-[11px] text-magic-fire/80 truncate max-w-[200px]" title={runtimeEventConnection.error}>
            {runtimeEventConnection.error}
          </span>
        )}

        {connection.status === 'disconnected' && (
          <button onClick={connect} className="px-2 py-0.5 rounded bg-primary/20 text-primary hover:bg-primary/30 transition-colors ghost-border">
            Connect
          </button>
        )}
        
        {connection.status === 'error' && (
          <button onClick={reconnect} className="p-1 rounded hover:bg-white/10 transition-colors" title="Reconnect">
            <RefreshCw className="w-3 h-3 text-slate-500" />
          </button>
        )}
        
        {connection.status !== 'disconnected' && (
          <button onClick={disconnect} className="px-2 py-0.5 rounded bg-magic-fire/20 text-magic-fire hover:bg-magic-fire/30 transition-colors ghost-border">
            Disconnect
          </button>
        )}

        <div className="flex-1" />

        <div className="flex items-center rounded-lg border border-white/10 bg-black/30 p-0.5">
          <button
            onClick={() => setView('logs')}
            className={`px-2 py-1 rounded-md text-[10px] transition-colors ${view === 'logs' ? 'bg-primary/20 text-primary' : 'text-slate-500 hover:text-slate-300'}`}
          >
            Logs
          </button>
          <button
            onClick={() => setView('events')}
            className={`px-2 py-1 rounded-md text-[10px] transition-colors flex items-center gap-1 ${view === 'events' ? 'bg-primary/20 text-primary' : 'text-slate-500 hover:text-slate-300'}`}
          >
            <Radio className="w-3 h-3" />
            Events
          </button>
        </div>

        <ToolbarButton onClick={togglePause} title={ui.isPaused ? 'Resume' : 'Pause'} active={ui.isPaused}>
          {ui.isPaused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
        </ToolbarButton>
        <ToolbarButton onClick={toggleAutoScroll} title={ui.autoScroll ? 'Unlock scroll' : 'Lock to bottom'} active={ui.autoScroll}>
          {ui.autoScroll ? <ArrowDownToLine className="w-3.5 h-3.5" /> : <ArrowUpFromLine className="w-3.5 h-3.5" />}
        </ToolbarButton>
        <ToolbarButton onClick={view === 'logs' ? clearLogs : clearRuntimeEvents} title={view === 'logs' ? 'Clear logs' : 'Clear events'}>
          <Trash2 className="w-3.5 h-3.5" />
        </ToolbarButton>
      </div>

      {/* ── Filter bar ── */}
      {view === 'logs' && <div className="flex items-center gap-2 px-3 py-2 border-b border-white/5 bg-black/20">
        <select
          value={filters.logLevel || ''}
          onChange={handleLevelChange}
          className="bg-black/40 border border-white/10 rounded-md px-2 py-1 text-slate-300 outline-none focus:border-primary/50 cursor-pointer appearance-none"
        >
          <option value="">ALL</option>
          <option value="DEBUG">DEBUG</option>
          <option value="INFO">INFO</option>
          <option value="WARNING">WARN</option>
          <option value="ERROR">ERROR</option>
          <option value="CRITICAL">CRIT</option>
        </select>

        <div className="flex-1 relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500 pointer-events-none" />
          <input
            type="text"
            placeholder="Filter logs..."
            value={filters.searchText}
            onChange={handleSearchChange}
            className="w-full bg-black/40 border border-white/10 rounded-md pl-8 pr-3 py-1 text-slate-300 placeholder:text-slate-600 outline-none focus:border-primary/50 transition-colors"
          />
        </div>
      </div>}

      {/* ── Log viewport ── */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto scrollbar-hide py-2">
        {view === 'logs' ? (traceGroups.size === 0 ? (
          <EmptyState status={connection.status} />
        ) : (
          <div className="py-2">
            {Array.from(traceGroups.values()).map(trace => (
              <TraceBlock
                key={trace.trace_id}
                trace={trace}
                onToggleTrace={() => toggleTraceCollapse(trace.trace_id)}
                onToggleSpan={(span) => toggleSpanCollapse(trace.trace_id, span)}
              />
            ))}
          </div>
        )) : runtimeEvents.length === 0 ? (
          <EmptyState status={runtimeEventConnection.status === 'connected' ? 'connected' : runtimeEventConnection.status === 'error' ? 'error' : 'connecting'} />
        ) : (
          <div className="py-2">
            {runtimeEvents.map((event) => (
              <EventRow key={event.event_id} event={event} />
            ))}
          </div>
        )}
      </div>

      {/* ── Status bar ── */}
      <div className="flex items-center justify-between px-3 py-1.5 border-t border-white/5 bg-black/40 text-[10px] text-slate-500 select-none">
        <span>
          {view === 'logs'
            ? `${logs.length} entries${(filters.logLevel || filters.searchText) ? ' (filtered)' : ''}`
            : `${runtimeEvents.length} events`}
        </span>
        <div className="flex items-center gap-3">
          {ui.isPaused && <span className="text-magic-metal">PAUSED</span>}
          {ui.autoScroll && <span className="text-primary/70">AUTO-SCROLL</span>}
        </div>
      </div>
    </div>
  );
}
