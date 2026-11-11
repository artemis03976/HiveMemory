import { useRef, useState } from 'react';
import type { KeyboardEvent } from 'react';
import { ArrowUp, List, Search } from 'lucide-react';
import { formatDuration } from '@/stores/runs';
import type { RunFilters } from '@/stores/runs';
import type { Run } from '@/types/runs';
import { KIND_META, KIND_ORDER, STATUS_META, formatClock, formatFullTime } from './runVisuals';

interface RunListProps {
  runs: Run[];
  allRuns: Run[];
  filters: RunFilters;
  onFiltersChange: (filters: Partial<RunFilters>) => void;
  selectedKey: string | null;
  onSelect: (key: string | null) => void;
  now: number;
}

/** 列表离开顶部多远之后不再跟随新运行，改为提示"N 个新运行"。 */
const TOP_THRESHOLD_PX = 8;

function runDuration(run: Run, now: number): string {
  if (run.status === 'running') return `已进行 ${formatDuration(now - run.startedAt)}`;
  return run.durationMs !== null ? formatDuration(run.durationMs) : '—';
}

export default function RunList({ runs, allRuns, filters, onFiltersChange, selectedKey, onSelect, now }: RunListProps) {
  const listRef = useRef<HTMLDivElement>(null);
  // 滚离顶部时记下当时最上面的运行，它之上的都是之后出现的新运行。
  const [anchorKey, setAnchorKey] = useState<string | null>(null);
  const anchorIndex = anchorKey ? runs.findIndex((run) => run.key === anchorKey) : -1;
  const newRunCount = Math.max(anchorIndex, 0);

  const handleScroll = () => {
    const scrollTop = listRef.current?.scrollTop ?? 0;
    if (scrollTop <= TOP_THRESHOLD_PX) setAnchorKey(null);
    else if (anchorKey === null) setAnchorKey(runs[0]?.key ?? null);
  };

  const backToTop = () => {
    listRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
    setAnchorKey(null);
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape') {
      onSelect(null);
      return;
    }
    if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
    event.preventDefault();
    const index = runs.findIndex((run) => run.key === selectedKey);
    const next = event.key === 'ArrowDown' ? Math.min(index + 1, runs.length - 1) : Math.max(index - 1, 0);
    const run = runs[index === -1 ? 0 : next];
    if (!run) return;
    onSelect(run.key);
    listRef.current?.querySelector(`[data-run-key="${CSS.escape(run.key)}"]`)?.scrollIntoView({ block: 'nearest' });
  };

  const chip = (key: RunFilters['kind'], label: string, Icon: typeof List, count: number) => {
    const active = filters.kind === key;
    return (
      <button
        key={key}
        type="button"
        aria-pressed={active}
        onClick={() => onFiltersChange({ kind: key })}
        className={`flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-semibold transition-colors ${
          active
            ? 'border-primary/50 bg-primary/15 text-on-surface'
            : 'border-line bg-surface-container text-fg-2 hover:bg-surface-container-high'
        }`}
      >
        <Icon className="h-3 w-3" aria-hidden="true" />
        <span>{label}</span>
        <span className="font-normal text-fg-3">{count}</span>
      </button>
    );
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-col gap-2.5 border-b border-line px-4 pb-3 pt-4">
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="按种类筛选">
          {chip('all', '全部', List, allRuns.length)}
          {KIND_ORDER.map((kind) =>
            chip(kind, KIND_META[kind].label, KIND_META[kind].icon, allRuns.filter((run) => run.kind === kind).length),
          )}
        </div>
        <div className="flex items-center gap-4 text-[12px] text-fg-3">
          <label className="flex cursor-pointer items-center gap-1.5">
            <input
              type="checkbox"
              className="accent-primary"
              checked={filters.onlyActive}
              onChange={(event) => onFiltersChange({ onlyActive: event.target.checked })}
            />
            只看进行中
          </label>
          <label className="flex cursor-pointer items-center gap-1.5">
            <input
              type="checkbox"
              className="accent-primary"
              checked={filters.onlyIssues}
              onChange={(event) => onFiltersChange({ onlyIssues: event.target.checked })}
            />
            只看异常
          </label>
        </div>
        <label className="relative block">
          <span className="sr-only">按 trace、task、话题或 agent 筛选</span>
          <Search className="pointer-events-none absolute left-2.5 top-2 h-3.5 w-3.5 text-fg-3" aria-hidden="true" />
          <input
            type="search"
            value={filters.query}
            onChange={(event) => onFiltersChange({ query: event.target.value })}
            placeholder="按 trace、task、话题或 agent 筛选"
            className="w-full rounded-lg border border-line bg-surface-container-lowest py-1.5 pl-8 pr-2 text-[12px] text-on-surface placeholder:text-fg-3 focus:border-primary/50 focus:outline-none"
          />
        </label>
      </div>

      <div className="relative min-h-0 flex-1">
        {newRunCount > 0 && (
          <button
            type="button"
            onClick={backToTop}
            className="absolute left-1/2 top-2 z-10 flex -translate-x-1/2 items-center gap-1 rounded-full border border-primary/40 bg-surface-container-highest px-3 py-1 text-xs font-semibold text-primary-light shadow-lg"
          >
            <ArrowUp className="h-3 w-3" aria-hidden="true" />
            {newRunCount} 个新运行
          </button>
        )}
        <div
          ref={listRef}
          role="listbox"
          aria-label="运行列表"
          tabIndex={0}
          onScroll={handleScroll}
          onKeyDown={handleKeyDown}
          className="h-full overflow-y-auto px-2 py-2 focus:outline-none"
        >
          {runs.map((run) => (
            <RunRow
              key={run.key}
              run={run}
              selected={run.key === selectedKey}
              onSelect={() => onSelect(run.key)}
              now={now}
            />
          ))}
          {runs.length === 0 && (
            <p className="px-3 py-8 text-center text-[12px] text-fg-3">
              {allRuns.length === 0 ? '缓冲中还没有事件' : '没有符合筛选条件的运行'}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function RunRow({ run, selected, onSelect, now }: { run: Run; selected: boolean; onSelect: () => void; now: number }) {
  const kind = KIND_META[run.kind];
  const status = STATUS_META[run.status];
  const KindIcon = kind.icon;
  return (
    <button
      type="button"
      role="option"
      aria-selected={selected}
      data-run-key={run.key}
      tabIndex={-1}
      onClick={onSelect}
      className={`mb-0.5 flex w-full gap-2.5 rounded-lg px-2 py-2.5 text-left transition-colors ${
        selected
          ? 'bg-surface-container-highest ring-1 ring-inset ring-primary/40'
          : run.status === 'failed'
            ? 'bg-magic-fire/5 hover:bg-surface-container-high'
            : 'hover:bg-surface-container-high'
      }`}
    >
      <span
        className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border border-line bg-surface-container-high"
        title={kind.label}
      >
        <KindIcon className="h-3.5 w-3.5 text-fg-2" aria-hidden="true" />
      </span>
      <span className="flex min-w-0 flex-1 flex-col gap-1">
        <span className="flex items-center gap-2">
          <span className="truncate text-[13px] font-semibold text-on-surface">{run.title}</span>
          <span className="flex-1" />
          <span className={`flex shrink-0 items-center gap-1.5 text-xs font-semibold ${status.text}`}>
            <span className={`h-1.5 w-1.5 rounded-full ${status.dot}`} aria-hidden="true" />
            {status.label}
          </span>
        </span>
        {run.subtitle && <span className="truncate text-[12px] text-fg-3">{run.subtitle}</span>}
        <span className="flex gap-2.5 font-mono text-xs text-fg-3">
          <span title={formatFullTime(run.startedAt)}>{formatClock(run.startedAt)}</span>
          <span>{runDuration(run, now)}</span>
          <span>{run.events.length} 条事件</span>
        </span>
      </span>
    </button>
  );
}
