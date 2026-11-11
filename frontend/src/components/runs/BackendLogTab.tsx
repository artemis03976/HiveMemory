import { useLayoutEffect, useMemo, useRef, useState } from 'react';
import { ArrowDown, X } from 'lucide-react';
import { useWorkspaceStore } from '@/stores';
import { filterLogs } from '@/stores/workspace/selectors';
import type { LogEntry, LogLevel } from '@/types/workspace';
import { formatClock } from './runVisuals';

const LEVELS: (LogLevel | null)[] = [null, 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'];

const LEVEL_STYLE: Record<LogLevel, string> = {
  DEBUG: 'text-fg-3 bg-fg-3/10',
  INFO: 'text-magic-water bg-magic-water/10',
  WARNING: 'text-magic-metal bg-magic-metal/10',
  ERROR: 'text-magic-fire bg-magic-fire/10',
  CRITICAL: 'text-magic-fire bg-magic-fire/20',
};

/** 离底部多近算"在底部"：在底部时跟随新日志，否则提示"N 条新日志"。 */
const BOTTOM_THRESHOLD_PX = 24;

interface BackendLogTabProps {
  traceId: string | null;
  onClearTrace: () => void;
}

/** 平铺的后端日志尾部；按级别、logger 前缀、文本与 trace 筛选。 */
export default function BackendLogTab({ traceId, onClearTrace }: BackendLogTabProps) {
  const allLogs = useWorkspaceStore((state) => state.logs);
  const filters = useWorkspaceStore((state) => state.filters);
  const setLogLevel = useWorkspaceStore((state) => state.setLogLevel);
  const setLoggerNamespace = useWorkspaceStore((state) => state.setLoggerNamespace);
  const setSearchText = useWorkspaceStore((state) => state.setSearchText);

  const logs = useMemo(
    () => filterLogs(allLogs, filters).filter((log) => !traceId || log.trace_id === traceId),
    [allLogs, filters, traceId],
  );

  const scrollRef = useRef<HTMLDivElement>(null);
  // 滚离底部时记下当时最后一条日志，它之后的都是新日志。
  const [anchorId, setAnchorId] = useState<string | null>(null);
  const anchorIndex = anchorId ? logs.findIndex((log) => log.id === anchorId) : -1;
  const newLogCount = anchorId && anchorIndex !== -1 ? logs.length - 1 - anchorIndex : 0;

  useLayoutEffect(() => {
    if (anchorId === null && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, anchorId]);

  const handleScroll = () => {
    const element = scrollRef.current;
    if (!element) return;
    const atBottom = element.scrollHeight - element.scrollTop - element.clientHeight <= BOTTOM_THRESHOLD_PX;
    if (atBottom) setAnchorId(null);
    else if (anchorId === null) setAnchorId(logs[logs.length - 1]?.id ?? null);
  };

  return (
    <section aria-label="后端日志" className="flex min-h-0 flex-1 flex-col">
      <div className="flex flex-wrap items-center gap-2.5 border-b border-line bg-surface-container-low px-6 py-3">
        <div role="group" aria-label="日志级别" className="flex gap-1">
          {LEVELS.map((level) => {
            const active = filters.logLevel === level;
            return (
              <button
                key={level ?? 'all'}
                type="button"
                aria-pressed={active}
                onClick={() => setLogLevel(level)}
                className={`rounded-md border px-2.5 py-1 text-xs font-semibold ${
                  active
                    ? 'border-primary/50 bg-primary/15 text-fg'
                    : 'border-line bg-surface-container text-fg-2 hover:bg-surface-container-high'
                }`}
              >
                {level ?? '全部'}
              </button>
            );
          })}
        </div>
        <label className="flex items-center gap-1.5 text-[12px] text-fg-3">
          logger
          <input
            type="text"
            value={filters.loggerNamespace}
            onChange={(event) => setLoggerNamespace(event.target.value)}
            placeholder="hivememory.patchouli"
            className="w-44 rounded-md border border-line bg-surface-container-lowest px-2 py-1 font-mono text-[12px] text-fg placeholder:text-fg-3 focus:border-primary/50 focus:outline-none"
          />
        </label>
        <label className="flex min-w-[180px] flex-1 items-center gap-1.5 text-[12px] text-fg-3">
          文本
          <input
            type="search"
            value={filters.searchText}
            onChange={(event) => setSearchText(event.target.value)}
            placeholder="在消息与 logger 中查找"
            className="min-w-0 flex-1 rounded-md border border-line bg-surface-container-lowest px-2 py-1 text-[12px] text-fg placeholder:text-fg-3 focus:border-primary/50 focus:outline-none"
          />
        </label>
        {traceId && (
          <span className="flex items-center gap-1.5 rounded-full border border-primary/35 bg-primary/10 py-0.5 pl-2.5 pr-1 text-[12px] text-primary-light">
            trace <span className="font-mono">{traceId}</span>
            <button
              type="button"
              onClick={onClearTrace}
              aria-label="移除 trace 筛选"
              className="flex h-5 w-5 items-center justify-center rounded-full hover:bg-ink/10"
            >
              <X className="h-3 w-3" aria-hidden="true" />
            </button>
          </span>
        )}
      </div>

      <div className="relative min-h-0 flex-1">
        {newLogCount > 0 && (
          <button
            type="button"
            onClick={() => setAnchorId(null)}
            className="absolute bottom-3 left-1/2 z-10 flex -translate-x-1/2 items-center gap-1 rounded-full border border-primary/40 bg-surface-container-highest px-3 py-1 text-xs font-semibold text-primary-light shadow-lg"
          >
            <ArrowDown className="h-3 w-3" aria-hidden="true" />
            {newLogCount} 条新日志
          </button>
        )}
        <div ref={scrollRef} onScroll={handleScroll} className="h-full overflow-auto py-2 font-mono text-[12px]">
          {logs.map((log) => (
            <LogLine key={log.id} log={log} />
          ))}
          {logs.length === 0 && (
            <p className="px-6 py-8 text-center font-sans text-[12px] text-fg-3">
              {allLogs.length === 0 ? '还没有收到后端日志' : '没有符合筛选条件的日志'}
            </p>
          )}
        </div>
      </div>
    </section>
  );
}

function LogLine({ log }: { log: LogEntry }) {
  const loggerShort = log.logger.split('.').slice(-2).join('.');
  const isSystemTrace = log.trace_id === 'system';
  return (
    <div className={`px-6 py-1 ${log.level === 'ERROR' || log.level === 'CRITICAL' ? 'bg-magic-fire/5' : ''}`}>
      <div className="flex items-baseline gap-3 whitespace-nowrap">
        <span className="text-fg-3" title={log.timestamp}>{formatClock(Date.parse(log.timestamp), true)}</span>
        <span className={`w-16 shrink-0 rounded py-px text-center text-xs font-semibold ${LEVEL_STYLE[log.level]}`}>
          {log.level}
        </span>
        <span className="w-48 shrink-0 truncate text-fg-3" title={log.logger}>{loggerShort}</span>
        <span className={`shrink-0 ${isSystemTrace ? 'text-fg-3' : 'text-primary/80'}`}>
          [{log.trace_id}/{log.span_name}]
        </span>
        <span className="whitespace-pre-wrap break-all text-fg">{log.message}</span>
      </div>
      {log.exception && (
        <div className="ml-[19.5rem] mt-1 rounded border border-magic-fire/20 bg-magic-fire/10 px-2 py-1 text-magic-fire">
          {log.exception.type}: {log.exception.message}
        </div>
      )}
    </div>
  );
}
