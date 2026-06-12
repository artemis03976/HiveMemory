import { AnimatePresence, motion } from 'motion/react';
import { useToastStore } from '@/stores';
import { CheckCircle, AlertCircle, Info, AlertTriangle, X } from 'lucide-react';

export default function DynamicToast() {
  const { toasts, removeToast } = useToastStore();

  return (
    <div className="fixed top-6 left-1/2 -translate-x-1/2 z-100 flex flex-col items-center gap-3 pointer-events-none">
      <AnimatePresence>
        {toasts.map((toast) => {
          const Icon = 
            toast.type === 'success' ? CheckCircle :
            toast.type === 'error' ? AlertCircle :
            toast.type === 'warning' ? AlertTriangle : 
            Info;
          
          const iconColor = 
            toast.type === 'success' ? 'text-magic-wood' :
            toast.type === 'error' ? 'text-magic-fire' :
            toast.type === 'warning' ? 'text-magic-metal' :
            'text-magic-water';

          return (
            <motion.div
              key={toast.id}
              layout
              initial={{ opacity: 0, y: -40, scale: 0.8 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -20, scale: 0.9 }}
              transition={{ 
                type: 'spring', 
                stiffness: 500, 
                damping: 30, 
                mass: 2 
              }}
              className="pointer-events-auto flex items-center gap-3 px-4 py-3 min-w-[300px] max-w-md rounded-full shadow-2xl backdrop-blur-xl bg-surface/90 border border-white/10 group cursor-default"
            >
              <div className={`shrink-0 ${iconColor} bg-white/5 p-1.5 rounded-full`}>
                <Icon className="w-4 h-4" />
              </div>
              
              <span className="flex-1 text-sm font-medium tracking-wide text-on-surface line-clamp-2">
                {toast.message}
              </span>

              <button
                onClick={() => removeToast(toast.id)}
                className="shrink-0 p-1.5 rounded-full text-slate-400 opacity-0 group-hover:opacity-100 hover:bg-white/10 hover:text-white transition-all duration-200"
                aria-label="关闭提示"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
