import { useState } from 'react';
import { ArrowLeft, Copy, Hash, ScrollText, Sparkles } from 'lucide-react';
import { describeEvent, formatDuration, gatewayFallbacks, gatewaySteps } from '@/stores/runs';
import { useToastStore } from '@/stores';
import type { Run } from '@/types/runs';
import type { RuntimeEvent } from '@/types/workspace';
import { KIND_META, STATUS_META, TONE_META, formatClock, formatFullTime } from './runVisuals';

interface RunDetailProps {
  run: Run;
  runsByKey: Map<string, Run>;
  now: number;
  onSelectRun: (key: string) => void;
  onFilterTopic: (topicId: string) => void;
  onOpenLogs: (traceId: string) => void;
  onBack: () => void;
}

export default function RunDetail({ run, runsByKey, now, onSelectRun, onFilterTopic, onOpenLogs, onBack }: RunDetailProps) {
  const addToast = useToastStore((state) => state.addToast);
  const kind = KIND_META[run.kind];
  const status = STATUS_META[run.status];
  const KindIcon = kind.icon;
  const traceId = run.traceId;
  const identifier = traceId ?? run.taskId ?? run.topicIds[0] ?? run.key;
  const origin = run.originKey ? runsByKey.get(run.originKey) : undefined;
  const generations = run.generationKeys.flatMap((key) => runsByKey.get(key) ?? []);
  const duration =
    run.status === 'running'
      ? `已进行 ${formatDuration(now - run.startedAt)}`
      : run.durationMs !== null
        ? formatDuration(run.durationMs)
        : '—';

  const copyIdentifier = () => {
    void navigator.clipboard
      .writeText(identifier)
      .then(() => addToast('已复制标识', 'success', 1500))
      .catch(() => addToast('复制失败', 'error'));
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-col gap-2.5 border-b border-white/5 px-7 pb-4 pt-5">
        <button
          type="button"
          onClick={onBack}
          className="flex w-fit items-center gap-1 text-[12px] text-slate-400 hover:text-on-surface min-[900px]:hidden"
        >
          <ArrowLeft className="h-3.5 w-3.5" aria-hidden="true" /> 返回列表
        </button>
        <div className="flex flex-wrap items-center gap-2.5">
          <span className="flex items-center gap-1.5 rounded-md border border-white/5 bg-surface-container-highest px-2 py-0.5 text-[11px] font-bold tracking-wide text-slate-300">
            <KindIcon className="h-3 w-3" aria-hidden="true" />
            {kind.label}
          </span>
          <h2 className="font-manrope text-[18px] font-bold text-on-surface">{run.title}</h2>
          <span className={`flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[12px] font-semibold ${status.text} ${status.badge}`}>
            <span className={`h-1.5 w-1.5 rounded-full ${status.dot}`} aria-hidden="true" />
            {status.label}
            {run.reason && <span className="font-mono font-normal">· {run.reason}</span>}
          </span>
          <span className="flex-1" />
          <span className="font-mono text-[13px] text-slate-300">{duration}</span>
        </div>
        <div className="flex flex-wrap gap-x-5 gap-y-1.5 text-[12px] text-slate-400">
          <span className="flex items-center gap-1.5">
            {traceId ? 'trace' : run.taskId ? 'task' : '标识'}
            <button
              type="button"
              onClick={copyIdentifier}
              aria-label="复制标识"
              className="flex items-center gap-1 rounded bg-surface-container px-1.5 py-0.5 font-mono text-primary-light hover:bg-surface-container-high"
            >
              {identifier}
              <Copy className="h-3 w-3" aria-hidden="true" />
            </button>
          </span>
          {run.agentId && (
            <span>
              Agent <span className="text-on-surface">{run.agentId}</span>
            </span>
          )}
          {run.workspaceId && (
            <span>
              Workspace <span className="text-on-surface">{run.workspaceId}</span>
            </span>
          )}
          <span>
            开始 <span className="font-mono text-on-surface" title={formatFullTime(run.startedAt)}>{formatClock(run.startedAt, true)}</span>
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {run.topicIds.map((topicId) => (
            <LinkChip key={topicId} icon={Hash} label="话题" value={topicId.slice(0, 8)} title={`筛出带话题 ${topicId} 的运行`} onClick={() => onFilterTopic(topicId)} />
          ))}
          {origin && (
            <LinkChip icon={KIND_META[origin.kind].icon} label="来自" value={origin.title} onClick={() => onSelectRun(origin.key)} />
          )}
          {generations.map((generation) => (
            <LinkChip key={generation.key} icon={Sparkles} label="生成任务" value={generation.subtitle || generation.title} onClick={() => onSelectRun(generation.key)} />
          ))}
          <span className="flex-1" />
          {traceId && (
            <button
              type="button"
              onClick={() => onOpenLogs(traceId)}
              className="flex items-center gap-1.5 rounded-lg border border-primary/35 bg-primary/10 px-3 py-1.5 text-[12px] font-semibold text-primary-light hover:bg-primary/15"
            >
              <ScrollText className="h-3.5 w-3.5" aria-hidden="true" /> 查看这次运行的日志
            </button>
          )}
        </div>
      </div>

      <ol aria-label="事件时间线" className="min-h-0 flex-1 overflow-y-auto px-7 pb-7 pt-3">
        {run.events.map((event) => (
          <TimelineItem key={event.event_id} event={event} startedAt={run.startedAt} />
        ))}
      </ol>
    </div>
  );
}

function LinkChip({
  icon: Icon,
  label,
  value,
  title,
  onClick,
}: {
  icon: typeof Hash;
  label: string;
  value: string;
  title?: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      className="flex max-w-[320px] items-center gap-1.5 rounded-full border border-white/10 bg-surface-container px-2.5 py-1 text-[12px] text-slate-200 hover:bg-surface-container-high"
    >
      <Icon className="h-3 w-3 shrink-0 text-slate-400" aria-hidden="true" />
      <span className="shrink-0 text-slate-400">{label}</span>
      <span className="truncate font-mono">{value}</span>
    </button>
  );
}

function TimelineItem({ event, startedAt }: { event: RuntimeEvent; startedAt: number }) {
  const [expanded, setExpanded] = useState(false);
  const description = describeEvent(event);
  const tone = TONE_META[description.tone];
  const offset = Math.max(0, Date.parse(event.timestamp) - startedAt);
  const isGatewayTerminal = event.event_type === 'gateway.workflow.completed' || event.event_type === 'gateway.workflow.failed';

  return (
    <li className="flex">
      <span className="w-[72px] shrink-0 pr-3.5 pt-2.5 text-right font-mono text-[12px] text-slate-500" title={formatFullTime(Date.parse(event.timestamp))}>
        +{formatDuration(offset)}
      </span>
      <span className="relative flex w-4 shrink-0 justify-center" aria-hidden="true">
        <span className="absolute bottom-0 top-0 w-px bg-white/10" />
        <span className={`relative mt-3.5 h-2 w-2 rounded-full ring-4 ring-background ${tone.dot}`} />
      </span>
      <div className="min-w-0 flex-1 py-1 pl-2.5">
        <button
          type="button"
          onClick={() => setExpanded((value) => !value)}
          aria-expanded={expanded}
          className={`flex w-full flex-col gap-0.5 rounded-lg px-2.5 py-1.5 text-left hover:bg-surface-container ${expanded ? 'bg-surface-container' : tone.row}`}
        >
          <span className="flex items-baseline gap-2.5">
            <span className={`text-[13px] font-semibold ${tone.text}`}>{description.text}</span>
            <span className="flex-1" />
            <span className="whitespace-nowrap font-mono text-[11px] text-slate-500">{event.event_type}</span>
          </span>
          {description.detail && <span className="break-all text-[12px] text-slate-400">{description.detail}</span>}
        </button>
        {isGatewayTerminal && <GatewaySteps event={event} />}
        {expanded && (
          <pre className="mx-2.5 my-1.5 whitespace-pre-wrap break-all rounded-lg border border-white/5 bg-surface-container-lowest p-3 font-mono text-[11px] leading-relaxed text-slate-300">
            {JSON.stringify(event, null, 2)}
          </pre>
        )}
      </div>
    </li>
  );
}

function GatewaySteps({ event }: { event: RuntimeEvent }) {
  const steps = gatewaySteps(event);
  const fallbacks = gatewayFallbacks(event);
  if (steps.length === 0 && fallbacks.length === 0) return null;
  const slowest = Math.max(...steps.map((step) => step.durationMs), 1);

  return (
    <div className="mx-2.5 my-1.5 flex flex-col gap-1.5 rounded-lg border border-white/5 bg-surface-container-low px-3 py-2.5">
      {steps.map((step) => {
        // 最慢的一步（通常是话题路由）用 primary 标出，其余为中性色。
        const highlighted = step.durationMs === slowest && slowest > 10;
        return (
          <div key={step.stepId} className="flex items-center gap-2.5 text-[12px]">
            <span className={`w-24 shrink-0 ${highlighted ? 'text-on-surface' : 'text-slate-400'}`} title={step.stepId}>
              {step.label}
            </span>
            <span className="h-2 flex-1 overflow-hidden rounded bg-white/5">
              <span
                className={`block h-full rounded ${highlighted ? 'bg-primary' : 'bg-slate-500/50'}`}
                style={{ width: `${Math.max(2, Math.round((step.durationMs / slowest) * 100))}%` }}
              />
            </span>
            <span className={`w-16 shrink-0 text-right font-mono ${highlighted ? 'text-on-surface' : 'text-slate-400'}`}>
              {formatDuration(step.durationMs)}
            </span>
          </div>
        );
      })}
      {fallbacks.length > 0 && (
        <div className="text-[12px] text-magic-metal">降级：{fallbacks.join('，')}</div>
      )}
    </div>
  );
}
