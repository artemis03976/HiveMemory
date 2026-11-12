import { AnimatePresence, motion } from 'motion/react';
import { useToastStore } from '@/stores';
import { Check, X, Info, AlertTriangle } from 'lucide-react';

/** 右上角、页面头部之下的提示，不压住页面标题；错误提示带关闭按钮。 */
export default function DynamicToast() {
  const { toasts, removeToast } = useToastStore();

  return (
    <div className="pointer-events-none fixed right-6 top-20 z-100 flex flex-col items-end gap-2" role="status" aria-live="polite">
      <AnimatePresence>
        {toasts.map((toast) => {
          const Icon =
            toast.type === 'success' ? Check :
            toast.type === 'error' ? X :
            toast.type === 'warning' ? AlertTriangle :
            Info;

          const tone =
            toast.type === 'success' ? 'bg-magic-wood/15 text-magic-wood' :
            toast.type === 'error' ? 'bg-magic-fire/15 text-magic-fire' :
            toast.type === 'warning' ? 'bg-magic-metal/15 text-magic-metal' :
            'bg-magic-water/15 text-magic-water';

          return (
            <motion.div
              key={toast.id}
              layout
              initial={{ opacity: 0, x: 24 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 24 }}
              transition={{ type: 'spring', stiffness: 500, damping: 36 }}
              className="pointer-events-auto flex max-w-md min-w-72 items-center gap-2.5 rounded-xl border border-line-strong bg-surface-container px-3 py-2.5 shadow-pop"
            >
              <span className={`flex h-[22px] w-[22px] shrink-0 items-center justify-center rounded-full ${tone}`}>
                <Icon className="h-3.5 w-3.5" strokeWidth={2.6} />
              </span>
              <span className="line-clamp-2 flex-1 text-sm text-fg">{toast.message}</span>
              {toast.type === 'error' && (
                <button
                  onClick={() => removeToast(toast.id)}
                  className="shrink-0 rounded-md p-1 text-fg-3 transition-colors hover:bg-ink/10 hover:text-fg"
                  aria-label="关闭提示"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              )}
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
