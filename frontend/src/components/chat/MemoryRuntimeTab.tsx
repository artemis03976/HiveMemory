import { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  Ban,
  CheckCircle2,
  ChevronDown,
  Clock3,
  Loader2,
  RefreshCw,
  Square,
  XCircle,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { useChatStore, useMemoryTaskStore } from '@/stores';
import type { MemoryGenerationTask, MemoryGenerationTaskStatus } from '@/types';

const TERMINAL_STATUSES = new Set<MemoryGenerationTaskStatus>([
  'completed',
  'cancelled',
  'failed',
]);

const STATUS_CONFIG: Record<
  MemoryGenerationTaskStatus,
  {
    label: string;
    icon: typeof Clock3;
    iconClass: string;
    dotClass: string;
    textClass: string;
  }
> = {
  pending: {
    label: '等待中',
    icon: Clock3,
    iconClass: 'text-magic-metal',
    dotClass: 'bg-magic-metal',
    textClass: 'text-magic-metal',
  },
  running: {
    label: '运行中',
    icon: Loader2,
    iconClass: 'text-magic-water animate-spin',
    dotClass: 'bg-magic-water',
    textClass: 'text-magic-water',
  },
  completed: {
    label: '已完成',
    icon: CheckCircle2,
    iconClass: 'text-magic-wood',
    dotClass: 'bg-magic-wood',
    textClass: 'text-magic-wood',
  },
  cancelled: {
    label: '已取消',
    icon: Ban,
    iconClass: 'text-slate-400',
    dotClass: 'bg-slate-500',
    textClass: 'text-slate-400',
  },
  failed: {
    label: '失败',
    icon: XCircle,
    iconClass: 'text-magic-fire',
    dotClass: 'bg-magic-fire',
    textClass: 'text-magic-fire',
  },
};

function formatTime(value: string | null): string {
  if (!value) return '未记录';
  const time = new Date(value);
  if (Number.isNaN(time.getTime())) return value;
  return time.toLocaleString(undefined, {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

function getTaskAlias(task: MemoryGenerationTask): string {
  return task.canonical_alias || task.pending_alias || task.task_id.slice(0, 8);
}

function canCancelTask(task: MemoryGenerationTask): boolean {
  return (task.status === 'pending' || task.status === 'running') && !task.cancel_requested;
}

function DetailRow({ label, value, danger = false }: { label: string; value: string | null | undefined; danger?: boolean }) {
  return (
    <div className="grid grid-cols-[104px_1fr] gap-3 text-[11px] leading-relaxed">
      <span className="text-slate-500 select-none">{label}</span>
      <span className={`min-w-0 break-all font-mono ${danger ? 'text-magic-fire/90' : 'text-slate-300/85'}`}>
        {value || '—'}
      </span>
    </div>
  );
}

function TaskItem({
  task,
  isCurrentRunTask,
  isExpanded,
  onToggle,
  onCancel,
}: {
  task: MemoryGenerationTask;
  isCurrentRunTask: boolean;
  isExpanded: boolean;
  onToggle: () => void;
  onCancel: () => void;
}) {
  const config = STATUS_CONFIG[task.status];
  const StatusIcon = config.icon;
  const alias = getTaskAlias(task);

  return (
    <motion.div
      layout
      className="overflow-hidden rounded-lg border border-white/10 bg-surface-container-high/80 hover:bg-surface-container-highest transition-colors"
    >
      <button
        type="button"
        onClick={onToggle}
        className="w-full flex items-start justify-between gap-3 px-3 py-3 text-left"
        aria-expanded={isExpanded}
      >
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 min-w-0">
            <span className={`h-2 w-2 rounded-full shrink-0 ${config.dotClass}`} />
            <span className="truncate text-[13px] font-semibold text-slate-200">
              {task.label || alias}
            </span>
            {isCurrentRunTask && (
              <span className="shrink-0 rounded-md border border-primary/20 bg-primary/10 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider text-primary">
                当前
              </span>
            )}
          </div>

          <div className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-[10px] text-slate-500">
            <span className="rounded bg-white/5 px-1.5 py-0.5 font-mono text-slate-400">{task.source}</span>
            <span className="font-mono text-slate-400">{alias}</span>
            <span>{formatTime(task.created_at)}</span>
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-2">
          <span className={`flex items-center gap-1 text-[10px] font-semibold ${config.textClass}`}>
            <StatusIcon className={`h-3.5 w-3.5 ${config.iconClass}`} />
            {task.cancel_requested ? '取消中' : config.label}
          </span>
          <ChevronDown className={`h-4 w-4 text-slate-500 transition-transform ${isExpanded ? 'rotate-180' : ''}`} />
        </div>
      </button>

      <AnimatePresence initial={false}>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: 'easeInOut' }}
            className="overflow-hidden border-t border-white/5 bg-black/20"
          >
            <div className="space-y-2.5 p-3">
              <DetailRow label="task_id" value={task.task_id} />
              <DetailRow label="topic_id" value={task.topic_id} />
              <DetailRow label="source" value={task.source} />
              <DetailRow label="pending_alias" value={task.pending_alias} />
              <DetailRow label="canonical_alias" value={task.canonical_alias} />
              <DetailRow label="status" value={task.status} />
              <DetailRow label="reason" value={task.reason} />
              <DetailRow label="error" value={task.error} danger />
              <DetailRow label="created_at" value={formatTime(task.created_at)} />
              <DetailRow label="started_at" value={formatTime(task.started_at)} />
              <DetailRow label="finished_at" value={formatTime(task.finished_at)} />

              {(canCancelTask(task) || task.cancel_requested) && (
                <div className="pt-2">
                  <button
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation();
                      onCancel();
                    }}
                    disabled={task.cancel_requested}
                    className="inline-flex h-8 items-center gap-2 rounded-md border border-magic-fire/30 bg-magic-fire/10 px-3 text-[11px] font-semibold text-magic-fire hover:bg-magic-fire/15 focus:outline-none focus:ring-1 focus:ring-magic-fire/50 disabled:cursor-wait disabled:opacity-60"
                  >
                    {task.cancel_requested ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    ) : (
                      <Square className="h-3.5 w-3.5" />
                    )}
                    {task.cancel_requested ? '正在取消' : '取消任务'}
                  </button>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

export default function MemoryRuntimeTab() {
  const currentMemoryTaskIds = useChatStore((state) => state.currentMemoryTaskIds);
  const tasks = useMemoryTaskStore((state) => state.tasks);
  const connection = useMemoryTaskStore((state) => state.connection);
  const showTerminalTasks = useMemoryTaskStore((state) => state.showTerminalTasks);
  const fetchTasks = useMemoryTaskStore((state) => state.fetchTasks);
  const cancelTask = useMemoryTaskStore((state) => state.cancelTask);
  const setShowTerminalTasks = useMemoryTaskStore((state) => state.setShowTerminalTasks);
  const [expandedTaskIds, setExpandedTaskIds] = useState<Record<string, boolean>>({});

  useEffect(() => {
    void fetchTasks();
  }, [fetchTasks]);

  const currentTaskIdSet = useMemo(() => new Set(currentMemoryTaskIds), [currentMemoryTaskIds]);

  const visibleTasks = useMemo(() => {
    const filtered = showTerminalTasks
      ? tasks
      : tasks.filter((task) => !TERMINAL_STATUSES.has(task.status));

    return [...filtered].sort((a, b) => {
      const aCurrent = currentTaskIdSet.has(a.task_id);
      const bCurrent = currentTaskIdSet.has(b.task_id);
      if (aCurrent !== bCurrent) return aCurrent ? -1 : 1;
      return 0;
    });
  }, [currentTaskIdSet, showTerminalTasks, tasks]);

  const activeCount = tasks.filter((task) => !TERMINAL_STATUSES.has(task.status)).length;
  const isLoading = connection.status === 'loading';
  const isEmpty = !isLoading && visibleTasks.length === 0;

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <h4 className="text-[10px] font-semibold uppercase tracking-widest text-slate-500">
            Memory Tasks
          </h4>
          <p className="mt-1 text-[11px] text-slate-500">
            {activeCount} 个活跃任务 · {tasks.length} 个总任务
          </p>
        </div>

        <button
          type="button"
          onClick={() => void fetchTasks()}
          disabled={isLoading}
          className="flex h-8 w-8 items-center justify-center rounded-md border border-white/10 bg-white/5 text-slate-400 hover:text-primary focus:outline-none focus:ring-1 focus:ring-primary/40 disabled:cursor-wait disabled:opacity-60"
          title="刷新任务"
          aria-label="刷新任务"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${isLoading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      <label className="mb-3 flex items-center justify-between rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-[12px] text-slate-300">
        <span>显示已结束任务</span>
        <input
          type="checkbox"
          checked={showTerminalTasks}
          onChange={(event) => setShowTerminalTasks(event.target.checked)}
          className="h-4 w-4 accent-primary"
        />
      </label>

      {connection.status === 'error' && (
        <div className="mb-3 flex items-start gap-2 rounded-lg border border-magic-fire/20 bg-magic-fire/10 p-3 text-[12px] text-magic-fire/90">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          <div className="min-w-0">
            <p className="font-semibold">任务加载失败</p>
            <p className="mt-1 wrap-break-word text-magic-fire/75">{connection.error}</p>
          </div>
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto pr-1 scrollbar-hide">
        {isLoading && tasks.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-14 text-slate-500">
            <Loader2 className="mb-3 h-7 w-7 animate-spin text-primary/70" />
            <p className="text-sm">正在加载记忆任务</p>
          </div>
        ) : isEmpty ? (
          <div className="flex flex-col items-center justify-center py-14 text-center text-slate-500">
            <Clock3 className="mb-3 h-8 w-8 opacity-40" />
            <p className="text-sm">暂无记忆任务</p>
            <p className="mt-1 max-w-[260px] text-xs leading-relaxed text-slate-600">
              发送消息后，记忆生成、更新和归档任务会显示在这里。
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            <AnimatePresence initial={false}>
              {visibleTasks.map((task) => (
                <TaskItem
                  key={task.task_id}
                  task={task}
                  isCurrentRunTask={currentTaskIdSet.has(task.task_id)}
                  isExpanded={expandedTaskIds[task.task_id] === true}
                  onToggle={() =>
                    setExpandedTaskIds((prev) => ({
                      ...prev,
                      [task.task_id]: !prev[task.task_id],
                    }))
                  }
                  onCancel={() => void cancelTask(task.task_id)}
                />
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>
    </div>
  );
}
