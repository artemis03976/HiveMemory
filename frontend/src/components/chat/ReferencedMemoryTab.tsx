import { useCallback, useState } from 'react';
import { Brain, ThumbsDown, ThumbsUp } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { useToastStore } from '@/stores';
import { recordMemoryFeedback } from '@/services/memoryApi';
import type { MemoryAtom } from '@/types';
import MemoryTypeTile from '@/components/memory/MemoryTypeTile';
import { memoryTypeLabels } from '@/types/memory';

interface ReferencedMemoryTabProps {
  memories: MemoryAtom[];
}

export default function ReferencedMemoryTab({ memories }: ReferencedMemoryTabProps) {
  const addToast = useToastStore((state) => state.addToast);
  const [feedbackByMemory, setFeedbackByMemory] = useState<Record<string, boolean>>({});
  const [pendingFeedback, setPendingFeedback] = useState<Record<string, boolean>>({});
  const safeMemories = Array.isArray(memories) ? memories : [];

  const handleFeedback = useCallback(async (memory: MemoryAtom, positive: boolean) => {
    setPendingFeedback((prev) => ({ ...prev, [memory.id]: true }));

    try {
      await recordMemoryFeedback(memory.id, positive);
      setFeedbackByMemory((prev) => ({ ...prev, [memory.id]: positive }));
      addToast(positive ? '已记录正向记忆反馈' : '已记录负向记忆反馈', 'success', 1800);
    } catch (error) {
      console.warn('Failed to record memory feedback:', error);
      addToast('记忆反馈记录失败', 'error', 2400);
    } finally {
      setPendingFeedback((prev) => ({ ...prev, [memory.id]: false }));
    }
  }, [addToast]);

  return (
    <div className="space-y-4">
      {safeMemories.length > 0 && (
        <h4 className="px-1 text-xs text-fg-3">本轮预检索命中 {safeMemories.length} 条，悬停可逐条反馈</h4>
      )}

      {safeMemories.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-12 gap-3 text-fg-3">
          <Brain className="w-8 h-8 opacity-30" />
          <p className="text-sm">暂无引用记忆</p>
          <p className="text-xs opacity-60">发送消息后，检索到的记忆将显示在此处</p>
        </div>
      ) : (
        <AnimatePresence mode="popLayout">
          {safeMemories.map((memory) => {
            const feedback = feedbackByMemory[memory.id];
            const isPending = pendingFeedback[memory.id] === true;

            return (
              <motion.div
                key={memory.id}
                layout
                initial={{ opacity: 0, y: -20, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9, transition: { duration: 0.2 } }}
                transition={{ duration: 0.4, type: 'spring', bounce: 0.2 }}
                className="group relative mb-3 overflow-hidden rounded-2xl border border-line bg-surface-container p-3 transition-colors duration-300 hover:border-line-strong focus-within:border-line-strong"
              >
                <div className="absolute right-3 top-3 z-20 flex items-center gap-1 opacity-0 translate-y-1 group-hover:opacity-100 group-hover:translate-y-0 group-focus-within:opacity-100 group-focus-within:translate-y-0 transition-all duration-200 ease-out">
                  <button
                    type="button"
                    onClick={() => handleFeedback(memory, true)}
                    disabled={isPending}
                    className={`p-1.5 rounded-md bg-field border border-line text-fg-3 shadow-sm backdrop-blur-sm hover:bg-magic-wood/15 hover:text-magic-wood focus:outline-none focus:ring-1 focus:ring-magic-wood/50 disabled:cursor-wait disabled:opacity-60 transition-all ${
                      feedback === true ? 'text-magic-wood bg-magic-wood/15 border-magic-wood/20' : ''
                    }`}
                    title="这条记忆有帮助"
                    aria-label={`标记记忆 ${memory.alias || memory.id} 有帮助`}
                  >
                    <ThumbsUp className="w-3.5 h-3.5" />
                  </button>
                  <button
                    type="button"
                    onClick={() => handleFeedback(memory, false)}
                    disabled={isPending}
                    className={`p-1.5 rounded-md bg-field border border-line text-fg-3 shadow-sm backdrop-blur-sm hover:bg-magic-fire/15 hover:text-magic-fire focus:outline-none focus:ring-1 focus:ring-magic-fire/50 disabled:cursor-wait disabled:opacity-60 transition-all ${
                      feedback === false ? 'text-magic-fire bg-magic-fire/15 border-magic-fire/20' : ''
                    }`}
                    title="这条记忆不准确"
                    aria-label={`标记记忆 ${memory.alias || memory.id} 不准确`}
                  >
                    <ThumbsDown className="w-3.5 h-3.5" />
                  </button>
                </div>

                <div className="relative z-10 mb-2 flex items-center gap-2.5 pr-16">
                  <MemoryTypeTile type={memory.memory_type} />
                  <span className="flex min-w-0 flex-col">
                    <span className="text-xs font-semibold text-fg-2">{memoryTypeLabels[memory.memory_type] ?? memory.memory_type}</span>
                    <span className="truncate font-mono text-xs text-fg-3">{memory.alias || memory.id.slice(0, 8)}</span>
                  </span>
                </div>
                {memory.title && <p className="relative z-10 mb-1 text-sm font-semibold leading-snug text-fg">{memory.title}</p>}
                <p className="relative z-10 mb-2.5 line-clamp-3 text-[13px] leading-relaxed text-fg-2">
                  {memory.summary}
                </p>
                <div className="relative z-10 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-fg-3">
                  {memory.confidence_score > 0 && (
                    <span title="生成时对内容可靠程度的判断">置信度 <b className="font-semibold text-fg-2">{(memory.confidence_score * 100).toFixed(0)}%</b></span>
                  )}
                  {memory.vitality_score > 0 && (
                    <span title="被使用越多越高，长期不用会衰减">生命力 <b className="font-semibold text-fg-2">{Math.round(memory.vitality_score)}</b></span>
                  )}
                  {(Array.isArray(memory.tags) ? memory.tags : []).slice(0, 3).map((tag) => (
                    <span key={tag} className="rounded-full bg-surface-container-high px-2 py-0.5 text-fg-2">#{tag}</span>
                  ))}
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>
      )}
    </div>
  );
}
