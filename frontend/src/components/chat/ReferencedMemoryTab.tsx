import { useCallback, useState } from 'react';
import { ThumbsDown, ThumbsUp } from 'lucide-react';
import { useToastStore } from '@/stores';
import { recordMemoryFeedback } from '@/services/memoryApi';
import type { MemoryAtom } from '@/types';
import MemoryTypeTile from '@/components/memory/MemoryTypeTile';
import { memoryTypeLabels } from '@/types/memory';

interface ReferencedMemoryTabProps {
  memories: MemoryAtom[];
}

const feedbackButtonCls =
  'flex h-[26px] w-[26px] items-center justify-center rounded-[7px] transition-colors disabled:cursor-wait disabled:opacity-60';

/** 本轮预检索命中的记忆；悬停或聚焦时可逐条反馈有没有帮助。 */
export default function ReferencedMemoryTab({ memories }: ReferencedMemoryTabProps) {
  const addToast = useToastStore((state) => state.addToast);
  const [feedbackByMemory, setFeedbackByMemory] = useState<Record<string, boolean>>({});
  const [pendingFeedback, setPendingFeedback] = useState<Record<string, boolean>>({});

  const handleFeedback = useCallback(async (memory: MemoryAtom, positive: boolean) => {
    setPendingFeedback((prev) => ({ ...prev, [memory.id]: true }));
    try {
      await recordMemoryFeedback(memory.id, positive);
      setFeedbackByMemory((prev) => ({ ...prev, [memory.id]: positive }));
      addToast(positive ? '已记录：这条记忆有帮助' : '已记录：这条记忆不准确', 'success');
    } catch (error) {
      console.warn('Failed to record memory feedback:', error);
      addToast('记忆反馈记录失败', 'error');
    } finally {
      setPendingFeedback((prev) => ({ ...prev, [memory.id]: false }));
    }
  }, [addToast]);

  if (memories.length === 0) {
    return (
      <p className="mt-10 text-center text-[13px] leading-7 text-fg-3">
        发送消息后，这里显示本轮
        <br />
        预检索命中的记忆
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-2.5">
      <span className="px-0.5 text-xs text-fg-3">本轮预检索命中 {memories.length} 条，悬停可逐条反馈</span>
      {memories.map((memory) => {
        const feedback = feedbackByMemory[memory.id];
        const isPending = pendingFeedback[memory.id] === true;
        const name = memory.alias || memory.id.slice(0, 8);
        return (
          <div key={memory.id} className="group flex flex-col gap-2 rounded-[14px] border border-line bg-surface-container p-3">
            <div className="flex items-center gap-2.5">
              <MemoryTypeTile type={memory.memory_type} />
              <span className="flex min-w-0 flex-1 flex-col">
                <span className="text-xs font-semibold text-fg-2">{memoryTypeLabels[memory.memory_type] ?? memory.memory_type}</span>
                <span className="truncate font-mono text-xs text-fg-3">{name}</span>
              </span>
              <span className={`flex gap-0.5 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100 ${feedback === undefined ? 'opacity-0' : ''}`}>
                <button
                  type="button"
                  onClick={() => handleFeedback(memory, true)}
                  disabled={isPending}
                  aria-pressed={feedback === true}
                  aria-label={`记忆 ${name} 有帮助`}
                  title="这条记忆有帮助"
                  className={`${feedbackButtonCls} ${feedback === true ? 'bg-primary/15 text-primary' : 'text-fg-3 hover:bg-surface-container-high hover:text-fg'}`}
                >
                  <ThumbsUp className="h-[13px] w-[13px]" />
                </button>
                <button
                  type="button"
                  onClick={() => handleFeedback(memory, false)}
                  disabled={isPending}
                  aria-pressed={feedback === false}
                  aria-label={`记忆 ${name} 不准确`}
                  title="这条记忆不准确"
                  className={`${feedbackButtonCls} ${feedback === false ? 'bg-primary/15 text-primary' : 'text-fg-3 hover:bg-surface-container-high hover:text-fg'}`}
                >
                  <ThumbsDown className="h-[13px] w-[13px]" />
                </button>
              </span>
            </div>
            {memory.title && <span className="text-sm font-semibold leading-snug text-fg">{memory.title}</span>}
            {memory.summary && <p className="line-clamp-3 text-[13px] leading-relaxed text-fg-2">{memory.summary}</p>}
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-fg-3">
              {memory.vitality_score > 0 && (
                <span title="生命力：被使用越多越高，长期不用会衰减">生命力 <b className="font-semibold text-fg-2">{Math.round(memory.vitality_score)}</b></span>
              )}
              {memory.confidence_score > 0 && (
                <span title="置信度：生成时对内容可靠程度的判断">置信度 <b className="font-semibold text-fg-2">{(memory.confidence_score * 100).toFixed(0)}%</b></span>
              )}
              {memory.tags.slice(0, 3).map((tag) => (
                <span key={tag} className="rounded-full bg-surface-container-high px-2 py-0.5 text-fg-2">#{tag}</span>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
