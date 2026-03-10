import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Loader2, CheckCircle, XCircle, ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { MtpAction } from '@/types';

interface MtpActionCardProps {
  action: MtpAction;
}

export function MtpActionCard({ action }: MtpActionCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const statusConfig = {
    pending: {
      icon: Loader2,
      text: '帕秋莉正在解析指令...',
      className: 'text-blue-400',
      iconClassName: 'animate-spin',
    },
    executing: {
      icon: Loader2,
      text: `帕秋莉正在执行 ${action.type}...`,
      className: 'text-purple-400 animate-pulse-glow',
      iconClassName: 'animate-spin',
    },
    success: {
      icon: CheckCircle,
      text: `${action.type} 执行成功`,
      className: 'text-green-400',
      iconClassName: '',
    },
    error: {
      icon: XCircle,
      text: `${action.type} 执行失败`,
      className: 'text-red-400',
      iconClassName: '',
    },
  };

  const config = statusConfig[action.status];
  const Icon = config.icon;

  return (
    <motion.div
      layout
      className={cn(
        'glass-card rounded-lg p-3 my-2 cursor-pointer',
        'transition-all duration-200',
        config.className
      )}
      onClick={() => setIsExpanded(!isExpanded)}
    >
      <motion.div layout className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Icon className={cn('w-4 h-4', config.iconClassName)} />
          <span className="text-sm font-medium">{config.text}</span>
        </div>

        {action.response && (
          <ChevronDown
            className={cn(
              'w-4 h-4 transition-transform duration-200',
              isExpanded && 'rotate-180'
            )}
          />
        )}
      </motion.div>

      <AnimatePresence>
        {isExpanded && action.response && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="mt-3 pt-3 border-t border-white/10">
              <pre className="text-xs font-mono text-muted-foreground overflow-x-auto custom-scrollbar p-2 bg-black/20 rounded">
                {action.response}
              </pre>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
