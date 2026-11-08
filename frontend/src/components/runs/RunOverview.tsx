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
        <h2 className="font-manrope text-[18px] font-bold text-on-surface">最近一小时</h2>
        <p className="mt-1 text-[12px] text-slate-400">选中左侧的一次运行查看它的事件时间线</p>
      </div>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-2.5">
        {KIND_ORDER.map((kind) => {
          const Icon = KIND_META[kind].icon;
          return (
            <div key={kind} className="flex flex-col gap-1.5 rounded-xl border border-white/5 bg-surface-container-low p-3.5">
              <span className="flex items-center gap-1.5 text-[12px] text-slate-400">
                <Icon className="h-3.5 w-3.5" aria-hidden="true" />
                {KIND_META[kind].label}
              </span>
              <span className="font-manrope text-[24px] font-extrabold text-on-surface">
                {recent.filter((run) => run.kind === kind).length}
              </span>
            </div>
          );
        })}
      </div>
      <div className="flex flex-col gap-2">
        <h3 className="text-[13px] font-semibold text-slate-200">需要注意</h3>
        {issues.length === 0 && <p className="text-[12px] text-slate-500">缓冲内没有失败、推迟或未见终态的运行</p>}
        {issues.map((run) => {
          const status = STATUS_META[run.status];
          return (
            <button
              key={run.key}
              type="button"
              onClick={() => onSelect(run.key)}
              className="flex items-center gap-2.5 rounded-lg border border-white/5 bg-surface-container-low px-3 py-2.5 text-left hover:bg-surface-container"
            >
              <span className={`h-1.5 w-1.5 rounded-full ${status.dot}`} aria-hidden="true" />
              <span className="text-[13px] font-semibold text-on-surface">{run.title}</span>
              <span className={`text-[12px] ${status.text}`}>{status.label}</span>
              {run.reason && <span className="truncate font-mono text-[11px] text-slate-500">{run.reason}</span>}
              <span className="flex-1" />
              <span className="font-mono text-[11px] text-slate-500">{formatClock(run.startedAt)}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
