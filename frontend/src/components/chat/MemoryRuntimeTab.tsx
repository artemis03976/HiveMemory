import { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, Loader2, RefreshCw, Square } from 'lucide-react';
import { useWorkspaceStore, useMemoryTaskStore } from '@/stores';
import { sortTasks } from '@/stores/memory/taskReducers';
import type { MemoryGenerationTask, MemoryGenerationTaskStatus } from '@/types';
import type { RuntimeEventConnectionStatus } from '@/types/workspace';
import { formatDateTime } from '@/utils/dateTime';

const TERMINAL_STATUSES = new Set<MemoryGenerationTaskStatus>(['completed', 'cancelled', 'failed']);

const EVENT_CONNECTION: Record<RuntimeEventConnectionStatus, { label: string; dot: string }> = {
  disconnected: { label: '事件流未连接', dot: 'bg-fg-3' },
  connecting: { label: '事件流连接中', dot: 'bg-magic-metal animate-pulse' },
  connected: { label: '事件流已连接', dot: 'bg-magic-wood' },
  disabled: { label: '事件流未启用', dot: 'bg-fg-3' },
  error: { label: '事件流连接失败', dot: 'bg-magic-fire' },
};

/** 状态色：水为进行中，木为完成，火为失败，金为等待。 */
const STATUS: Record<MemoryGenerationTaskStatus, { label: string; text: string; dot: string; pill: string }> = {
  pending: { label: '等待中', text: 'text-magic-metal', dot: 'bg-magic-metal', pill: 'bg-magic-metal/12' },
  running: { label: '生成中', text: 'text-magic-water', dot: 'bg-magic-water animate-pulse', pill: 'bg-magic-water/12' },
  completed: { label: '完成', text: 'text-magic-wood', dot: 'bg-magic-wood', pill: 'bg-magic-wood/12' },
  cancelled: { label: '已取消', text: 'text-fg-3', dot: 'bg-fg-3', pill: 'bg-fg-3/12' },
  failed: { label: '失败', text: 'text-magic-fire', dot: 'bg-magic-fire', pill: 'bg-magic-fire/12' },
};

function time(value: string | null): string | null {
  if (!value) return null;
  const ms = Date.parse(value);
  return Number.isNaN(ms) ? value : formatDateTime(ms);
}

/** 卡片底部的一行：何时开始或结束，以及用时或失败原因。 */
function taskMeta(task: MemoryGenerationTask): string {
  switch (task.status) {
    case 'pending':
      return `${time(task.created_at)} 进入队列`;
    case 'running':
      return `${time(task.started_at ?? task.created_at)} 开始`;
    case 'completed': {
      const seconds = task.started_at && task.finished_at
        ? (Date.parse(task.finished_at) - Date.parse(task.started_at)) / 1000
        : null;
      return seconds === null ? `${time(task.finished_at ?? task.created_at)} 完成` : `${time(task.finished_at)} · 用时 ${seconds.toFixed(1)} 秒`;
    }
    case 'failed':
      return [time(task.finished_at ?? task.created_at), [task.reason, task.error].filter(Boolean).join('：')].filter(Boolean).join(' · ');
    case 'cancelled':
      return `${time(task.finished_at ?? task.created_at)} 已取消`;
  }
}

function DetailRow({ label, value }: { label: string; value: string | null | undefined }) {
  return (
    <div className="grid grid-cols-[96px_1fr] gap-3 text-xs leading-relaxed">
      <span className="text-fg-3">{label}</span>
      <span className="min-w-0 break-all font-mono text-fg-2">{value || '—'}</span>
    </div>
  );
}

function TaskCard({ task, expanded, onToggle, onCancel }: {
  task: MemoryGenerationTask;
  expanded: boolean;
  onToggle: () => void;
  onCancel: () => void;
}) {
  const status = STATUS[task.status];
  const alias = task.canonical_alias || task.pending_alias || task.task_id.slice(0, 8);
  const cancelling = task.cancel_requested === true && !TERMINAL_STATUSES.has(task.status);
  const cancellable = (task.status === 'pending' || task.status === 'running') && !task.cancel_requested;

  return (
    <div className="overflow-hidden rounded-[14px] border border-line bg-surface-container">
      <button type="button" onClick={onToggle} aria-expanded={expanded} className="flex w-full flex-col gap-1.5 p-3 text-left">
        <span className="flex w-full items-center gap-2">
          <span className="min-w-0 flex-1 truncate text-[13px] font-semibold text-fg">{task.label || alias}</span>
          <span className={`inline-flex shrink-0 items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-semibold ${status.text} ${status.pill}`}>
            <span aria-hidden="true" className={`h-1.5 w-1.5 rounded-full ${status.dot}`} />
            {cancelling ? '取消中' : status.label}
          </span>
        </span>
        <span className="w-full truncate font-mono text-xs text-fg-3">{alias}</span>
        <span className="text-xs text-fg-3">{taskMeta(task)}</span>
      </button>

      {expanded && (
        <div className="flex flex-col gap-2 border-t border-line bg-field p-3">
          <DetailRow label="task_id" value={task.task_id} />
          <DetailRow label="topic_id" value={task.topic_id} />
          <DetailRow label="source" value={task.source} />
          <DetailRow label="pending_alias" value={task.pending_alias} />
          <DetailRow label="canonical_alias" value={task.canonical_alias} />
          <DetailRow label="reason" value={task.reason} />
          <DetailRow label="error" value={task.error} />
          <DetailRow label="created_at" value={time(task.created_at)} />
          <DetailRow label="started_at" value={time(task.started_at)} />
          <DetailRow label="finished_at" value={time(task.finished_at)} />
          {(cancellable || cancelling) && (
            <button
              type="button"
              onClick={onCancel}
              disabled={cancelling}
              className="mt-1 inline-flex h-8 w-fit items-center gap-2 rounded-lg border border-magic-fire/30 bg-magic-fire/10 px-3 text-xs font-semibold text-magic-fire hover:bg-magic-fire/15 disabled:cursor-wait disabled:opacity-60"
            >
              {cancelling ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Square className="h-3.5 w-3.5" />}
              {cancelling ? '正在取消' : '取消任务'}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

/** 记忆生成任务：默认只列进行中与等待中的任务，可以选择连已结束的一起显示。 */
export default function MemoryRuntimeTab() {
  const eventConnection = EVENT_CONNECTION[useWorkspaceStore((state) => state.runtimeEventConnection.status)];
  const tasks = useMemoryTaskStore((state) => state.tasks);
  const connection = useMemoryTaskStore((state) => state.connection);
  const showTerminalTasks = useMemoryTaskStore((state) => state.showTerminalTasks);
  const fetchTasks = useMemoryTaskStore((state) => state.fetchTasks);
  const cancelTask = useMemoryTaskStore((state) => state.cancelTask);
  const setShowTerminalTasks = useMemoryTaskStore((state) => state.setShowTerminalTasks);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  useEffect(() => {
    void fetchTasks();
  }, [fetchTasks]);

  // 任务没有进程关联，统一按创建时间倒序展示
  const visibleTasks = useMemo(
    () => sortTasks(showTerminalTasks ? tasks : tasks.filter((task) => !TERMINAL_STATUSES.has(task.status))),
    [showTerminalTasks, tasks],
  );
  const isLoading = connection.status === 'loading';

  return (
    <div className="flex flex-col gap-2.5">
      <div className="flex items-center gap-2 px-0.5 text-xs text-fg-3">
        <span aria-hidden="true" className={`h-1.5 w-1.5 rounded-full ${eventConnection.dot}`} />
        <span className="flex-1">{eventConnection.label}</span>
        <label className="flex cursor-pointer items-center gap-1.5">
          <input
            type="checkbox"
            checked={showTerminalTasks}
            onChange={(event) => setShowTerminalTasks(event.target.checked)}
            className="h-3.5 w-3.5 accent-primary"
          />
          显示已结束
        </label>
        <button
          type="button"
          onClick={() => void fetchTasks()}
          disabled={isLoading}
          aria-label="刷新任务"
          title="刷新任务"
          className="flex h-[26px] w-[26px] items-center justify-center rounded-[7px] text-fg-3 transition-colors hover:bg-surface-container-high hover:text-fg disabled:cursor-wait disabled:opacity-60"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${isLoading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {connection.status === 'error' && (
        <div className="flex items-start gap-2 rounded-xl border border-magic-fire/20 bg-magic-fire/10 p-3 text-xs text-magic-fire">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
          <span className="min-w-0 wrap-break-word">任务读取失败：{connection.error}</span>
        </div>
      )}

      {isLoading && tasks.length === 0 && (
        <p className="mt-10 flex items-center justify-center gap-2 text-[13px] text-fg-3">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
          正在读取记忆任务
        </p>
      )}
      {!isLoading && visibleTasks.length === 0 && (
        <p className="mt-10 text-center text-[13px] leading-7 text-fg-3">
          {showTerminalTasks ? '还没有记忆生成任务' : '没有进行中的记忆生成任务'}
          <br />
          写入、更新与话题结算会在这里显示进度
        </p>
      )}

      {visibleTasks.map((task) => (
        <TaskCard
          key={task.task_id}
          task={task}
          expanded={expanded[task.task_id] === true}
          onToggle={() => setExpanded((prev) => ({ ...prev, [task.task_id]: !prev[task.task_id] }))}
          onCancel={() => void cancelTask(task.task_id)}
        />
      ))}
    </div>
  );
}
