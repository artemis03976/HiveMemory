import { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, Pause, Play, Terminal, Trash2 } from 'lucide-react';
import { useChatUiStore, useWorkspaceStore } from '@/stores';
import { filterRuns, groupRuns } from '@/stores/runs';
import type { RuntimeEvent } from '@/types/workspace';
import RunList from './RunList';
import RunDetail from './RunDetail';
import RunOverview from './RunOverview';
import BackendLogTab from './BackendLogTab';
import { formatClock } from './runVisuals';

type RunViewTab = 'runs' | 'logs';

/** 进行中的耗时与"未见终态"的判定随时间变化，页面按这个间隔重算。 */
const CLOCK_TICK_MS = 15_000;

function useNow(intervalMs: number): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(timer);
  }, [intervalMs]);
  return now;
}

/** 全局运行页：主视图把运行时事件按一次运行展示，第二个标签是后端日志。 */
export default function RunView() {
  const [tab, setTab] = useState<RunViewTab>('runs');
  const [logTraceId, setLogTraceId] = useState<string | null>(null);
  const connect = useWorkspaceStore((state) => state.connect);
  const connectRuntimeEvents = useWorkspaceStore((state) => state.connectRuntimeEvents);

  // 离开运行页不断开：其他页面的 Memory Runtime 仍需要事件流。
  useEffect(() => {
    connect();
    connectRuntimeEvents();
  }, [connect, connectRuntimeEvents]);

  const openRunLogs = (traceId: string) => {
    setLogTraceId(traceId);
    setTab('logs');
  };

  return (
    <div className="flex h-full min-w-0 flex-1 flex-col bg-background font-inter text-on-surface">
      <RunViewHeader tab={tab} onTabChange={setTab} />
      {tab === 'runs' ? (
        <RunsTab onOpenLogs={openRunLogs} />
      ) : (
        <BackendLogTab traceId={logTraceId} onClearTrace={() => setLogTraceId(null)} />
      )}
    </div>
  );
}

function RunViewHeader({ tab, onTabChange }: { tab: RunViewTab; onTabChange: (tab: RunViewTab) => void }) {
  const runtimeEvents = useWorkspaceStore((state) => state.runtimeEvents);
  const eventConnection = useWorkspaceStore((state) => state.runtimeEventConnection);
  const logConnection = useWorkspaceStore((state) => state.connection);
  const ui = useWorkspaceStore((state) => state.ui);
  const togglePause = useWorkspaceStore((state) => state.togglePause);
  const clearRuntimeEvents = useWorkspaceStore((state) => state.clearRuntimeEvents);
  const clearLogs = useWorkspaceStore((state) => state.clearLogs);

  const connected = tab === 'runs' ? eventConnection.status === 'connected' : logConnection.status === 'connected';
  const connectionLabel =
    tab === 'runs'
      ? { connected: '事件流已连接', connecting: '事件流连接中', disconnected: '事件流未连接', disabled: '事件流未启用', error: '事件流连接失败' }[eventConnection.status]
      : { connected: '日志已连接', connecting: '日志连接中', reconnecting: '日志重连中', disconnected: '日志未连接', error: '日志连接失败' }[logConnection.status];
  const earliest = runtimeEvents[0];

  const tabButton = (value: RunViewTab, label: string) => (
    <button
      type="button"
      role="tab"
      aria-selected={tab === value}
      onClick={() => onTabChange(value)}
      className={`rounded-md px-3.5 py-1.5 text-[13px] font-semibold transition-colors ${
        tab === value ? 'bg-primary/20 text-on-surface' : 'text-slate-400 hover:text-slate-200'
      }`}
    >
      {label}
    </button>
  );

  return (
    <header className="flex flex-wrap items-center gap-4 border-b border-white/5 bg-surface-container-low px-6 py-3.5">
      <h1 className="flex items-center gap-3 text-2xl font-black tracking-tighter text-primary drop-shadow-[0_0_12px_rgba(197,154,255,0.3)]">
        <Terminal className="h-6 w-6" aria-hidden="true" />
        运行
      </h1>
      <div role="tablist" aria-label="运行页标签" className="flex gap-0.5 rounded-lg border border-white/10 bg-background p-0.5">
        {tabButton('runs', '运行')}
        {tabButton('logs', '后端日志')}
      </div>
      <span className="flex-1" />
      <div className="flex flex-wrap items-center gap-2 text-[12px] text-slate-400">
        <span className={`h-2 w-2 rounded-full ${connected ? 'bg-magic-wood' : 'bg-slate-500'}`} aria-hidden="true" />
        <span>{connectionLabel}</span>
        {tab === 'runs' && (
          <>
            <span className="text-slate-600">·</span>
            <span>
              缓冲 {runtimeEvents.length} / {ui.maxBufferSize}
            </span>
            {earliest && (
              <>
                <span className="text-slate-600">·</span>
                <span>最早 {formatClock(Date.parse(earliest.timestamp))}</span>
              </>
            )}
          </>
        )}
        {ui.isPaused && (
          <span className="rounded-full bg-magic-metal/10 px-2 py-0.5 font-semibold text-magic-metal">已暂停</span>
        )}
      </div>
      <div className="flex gap-1.5">
        <button
          type="button"
          onClick={togglePause}
          className="flex h-8 items-center gap-1.5 rounded-lg border border-white/10 bg-surface-container px-3 text-[12px] text-slate-300 hover:bg-surface-container-high"
        >
          {ui.isPaused ? <Play className="h-3.5 w-3.5" aria-hidden="true" /> : <Pause className="h-3.5 w-3.5" aria-hidden="true" />}
          {ui.isPaused ? '继续' : '暂停'}
        </button>
        <button
          type="button"
          onClick={tab === 'runs' ? clearRuntimeEvents : clearLogs}
          title="只清空本页显示，不影响后端缓冲"
          className="flex h-8 items-center gap-1.5 rounded-lg border border-white/10 bg-surface-container px-3 text-[12px] text-slate-300 hover:bg-surface-container-high"
        >
          <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
          清空
        </button>
      </div>
    </header>
  );
}

/** 断档提示：事件流续传时找不到回放点，回放点之后、缓冲最早一条之前的事件已经丢失。 */
function gapNotice(events: RuntimeEvent[]): string | null {
  const gap = events.find((event) => event.event_type === 'event.stream.gap');
  if (!gap) return null;
  const earliestId = gap.data?.earliest_event_id;
  const earliest = events.find((event) => event.event_id === earliestId) ?? gap;
  return `事件流曾中断，${formatClock(Date.parse(earliest.timestamp))} 之前的部分事件已经不在缓冲中`;
}

function RunsTab({ onOpenLogs }: { onOpenLogs: (traceId: string) => void }) {
  const events = useWorkspaceStore((state) => state.runtimeEvents);
  const connection = useWorkspaceStore((state) => state.runtimeEventConnection);
  const connectRuntimeEvents = useWorkspaceStore((state) => state.connectRuntimeEvents);
  const filters = useChatUiStore((state) => state.runFilters);
  const setFilters = useChatUiStore((state) => state.setRunFilters);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const now = useNow(CLOCK_TICK_MS);

  const runs = useMemo(() => groupRuns(events, now), [events, now]);
  const runsByKey = useMemo(() => new Map(runs.map((run) => [run.key, run])), [runs]);
  const visibleRuns = useMemo(() => filterRuns(runs, filters), [runs, filters]);
  const selected = selectedKey ? runsByKey.get(selectedKey) : undefined;
  const notice = useMemo(() => gapNotice(events), [events]);

  if (connection.status === 'disabled') {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-2 text-center text-slate-400">
        <p className="text-[14px] font-semibold text-slate-200">运行时事件未启用</p>
        <p className="text-[12px]">
          在 <code className="font-mono text-primary-light">configs/config.yaml</code> 中打开{' '}
          <code className="font-mono text-primary-light">runtime_events.enabled</code>；后端日志标签照常可用。
        </p>
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {connection.status === 'error' && (
        <div className="flex items-center gap-2 border-b border-magic-fire/20 bg-magic-fire/10 px-6 py-2 text-[12px] text-magic-fire">
          <AlertTriangle className="h-3.5 w-3.5" aria-hidden="true" />
          {connection.error ?? '事件流连接失败'}
          <button type="button" onClick={connectRuntimeEvents} className="ml-2 rounded px-2 py-0.5 underline hover:bg-magic-fire/10">
            重试
          </button>
        </div>
      )}
      {notice && (
        <div className="flex items-center gap-2 border-b border-magic-metal/20 bg-magic-metal/10 px-6 py-2 text-[12px] text-magic-metal">
          <AlertTriangle className="h-3.5 w-3.5" aria-hidden="true" />
          {notice}
        </div>
      )}
      <div className="flex min-h-0 flex-1">
        <section
          aria-label="运行列表"
          className={`min-h-0 w-[380px] shrink-0 border-r border-white/5 bg-surface-container-low max-[900px]:w-full max-[900px]:border-r-0 ${
            selected ? 'max-[900px]:hidden' : ''
          }`}
        >
          <RunList
            runs={visibleRuns}
            allRuns={runs}
            filters={filters}
            onFiltersChange={setFilters}
            selectedKey={selectedKey}
            onSelect={setSelectedKey}
            now={now}
          />
        </section>
        <section aria-label="运行详情" className={`min-h-0 min-w-0 flex-1 ${selected ? '' : 'max-[900px]:hidden'}`}>
          {selected ? (
            <RunDetail
              key={selected.key}
              run={selected}
              runsByKey={runsByKey}
              now={now}
              onSelectRun={setSelectedKey}
              onFilterTopic={(topicId) => setFilters({ query: topicId })}
              onOpenLogs={onOpenLogs}
              onBack={() => setSelectedKey(null)}
            />
          ) : (
            <RunOverview runs={runs} now={now} onSelect={setSelectedKey} />
          )}
        </section>
      </div>
    </div>
  );
}
