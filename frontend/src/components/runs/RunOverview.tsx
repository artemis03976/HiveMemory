import { ISSUE_STATUSES } from '@/stores/runs';
import type { Run } from '@/types/runs';
import { KIND_META, KIND_ORDER, STATUS_META, formatClock } from './runVisuals';

const OVERVIEW_WINDOW_MS = 60 * 60 * 1000;
const MAX_ISSUES = 8;

/** 没有选中运行时的概览：最近一小时各类运行的次数，以及缓冲内需要注意的运行。 */
export default function RunOverview({ runs, now, onSelect }: { runs: Run[]; now: number; onSelect: (key: string) => void }) {
  const recent = runs.filter((run) => now - run.startedAt <= OVERVIEW_WINDOW_MS);
  const issues = runs.filter((run) => ISSUE_STATUSES.has(run.status)).slice(0, MAX_ISSUES);

  return (
    <div className="flex flex-col gap-6 overflow-y-auto px-7 py-7">
      <div>
        <h2 className="font-manrope text-[18px] font-bold text-fg">最近一小时</h2>
        <p className="mt-1 text-[12px] text-fg-3">选中左侧的一次运行查看它的事件时间线</p>
      </div>
      <dl className="flex flex-wrap items-center gap-y-2 rounded-xl border border-line bg-surface-container-low px-1.5 py-2.5">
        {KIND_ORDER.map((kind) => {
          const Icon = KIND_META[kind].icon;
          return (
            <div key={kind} className="flex items-center gap-2 border-r border-line px-3.5 last:border-r-0">
              <dt className="flex items-center gap-1.5 text-[12px] text-fg-3">
                <Icon className="h-3.5 w-3.5" aria-hidden="true" />
                {KIND_META[kind].label}
              </dt>
              <dd className="font-manrope text-[17px] font-extrabold tabular-nums text-fg">
                {recent.filter((run) => run.kind === kind).length}
              </dd>
            </div>
          );
        })}
      </dl>
      <div className="flex flex-col gap-2">
        <h3 className="text-[13px] font-semibold text-fg">需要注意</h3>
        {issues.length === 0 && <p className="text-[12px] text-fg-3">缓冲内没有失败、推迟或未见终态的运行</p>}
        {issues.map((run) => {
          const status = STATUS_META[run.status];
          return (
            <button
              key={run.key}
              type="button"
              onClick={() => onSelect(run.key)}
              className="flex items-center gap-2.5 rounded-lg border border-line bg-surface-container-low px-3 py-2.5 text-left hover:bg-surface-container"
            >
              <span className={`h-1.5 w-1.5 rounded-full ${status.dot}`} aria-hidden="true" />
              <span className="text-[13px] font-semibold text-fg">{run.title}</span>
              <span className={`text-[12px] ${status.text}`}>{status.label}</span>
              {run.reason && <span className="truncate font-mono text-xs text-fg-3">{run.reason}</span>}
              <span className="flex-1" />
              <span className="font-mono text-xs text-fg-3">{formatClock(run.startedAt)}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
