import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Loader2, CheckCircle2, XCircle, ChevronDown, TerminalSquare, Wand2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { MtpAction } from '@/types';

interface MtpActionCardProps {
  action: MtpAction;
  // 预留给未来打通 L4 面板的接口
  onOpenKernelVision?: (actionId: string) => void; 
}

export function MtpActionCard({ action, onOpenKernelVision }: MtpActionCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  // 重新设计的状态配置：根据“五行”色彩体系进行重塑
  const statusConfig = {
    pending: {
      icon: Loader2,
      text: '正在解析指令...',
      bgClass: 'bg-magic-metal/10 border-magic-metal/20 shadow-[inset_0_0_12px_hsla(40,90%,55%,0.1)]',
      textClass: 'text-magic-metal',
      iconClass: 'animate-spin',
    },
    executing: {
      icon: Wand2, // 使用魔杖图标代表帕秋莉正在施法/执行
      text: `正在执行 ${action.type}...`,
      bgClass: 'bg-magic-water/10 border-magic-water/30 shadow-[inset_0_0_12px_hsla(190,90%,50%,0.15)]',
      textClass: 'text-magic-water',
      iconClass: 'animate-pulse drop-shadow-[0_0_8px_hsla(190,90%,50%,0.6)]',
    },
    success: {
      icon: CheckCircle2,
      text: `${action.type} 完毕`,
      bgClass: 'bg-magic-wood/10 border-magic-wood/20 shadow-[inset_0_0_12px_hsla(150,80%,45%,0.1)]',
      textClass: 'text-magic-wood',
      iconClass: '',
    },
    error: {
      icon: XCircle,
      text: `${action.type} 中断`,
      bgClass: 'bg-magic-fire/10 border-magic-fire/20 shadow-[inset_0_0_12px_hsla(340,80%,60%,0.1)]',
      textClass: 'text-magic-fire',
      iconClass: '',
    },
  };

  const config = statusConfig[action.status];
  const Icon = config.icon;

  return (
    <motion.div
      layout
      className={cn(
        'relative overflow-hidden rounded-xl border backdrop-blur-md',
        'transition-colors duration-300',
        config.bgClass
      )}
    >
      {/* 头部：状态胶囊 */}
      <motion.div 
        layout 
        className="flex items-center justify-between px-3 py-2 group"
      >
        {/* 左侧：点击展开简略参数 */}
        <div 
          className="flex items-center gap-2.5 cursor-pointer flex-1"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          <Icon className={cn('w-4 h-4', config.textClass, config.iconClass)} />
          <span className={cn("text-[13px] font-medium tracking-wide", config.textClass)}>
            {config.text}
          </span>
          <ChevronDown
            className={cn(
              'w-3.5 h-3.5 opacity-50 transition-transform duration-200',
              config.textClass,
              isExpanded && 'rotate-180'
            )}
          />
        </div>

        {/* 右侧：L4 透视镜触发按钮 */}
        {/* 只有在执行完毕或报错时，才允许去 L4 看日志 */}
        {(action.status === 'success' || action.status === 'error') && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onOpenKernelVision?.(action.id);
            }}
            className={cn(
              "flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] font-semibold transition-all duration-200",
              "opacity-0 -translate-x-2 group-hover:opacity-100 group-hover:translate-x-0",
              "bg-black/20 hover:bg-black/40",
              config.textClass
            )}
            title="在内核日志中查看完整结果"
          >
            <TerminalSquare className="w-3.5 h-3.5" />
            <span>日志</span>
          </button>
        )}
      </motion.div>

      {/* 展开区：仅显示输入参数 (Target & Args)，绝不显示长篇 Response */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="border-t border-black/10 bg-black/30"
          >
            <div className="p-3 text-[12px] font-mono text-muted-foreground/80 space-y-1.5">
              <div className="flex">
                <span className="w-10 shrink-0 text-emerald-500/40 select-none">TGT </span>
                <span className="text-primary-300 font-semibold">{action.target || '*'}</span>
              </div>
              {action.args && (
                <div className="flex">
                  <span className="w-12 shrink-0 opacity-50">ARG |</span>
                  <span className="text-emerald-300/70 line-clamp-3">
                    {/* 如果参数太长，这里也做了最多3行的截断 */}
                    {typeof action.args === 'string' ? action.args : JSON.stringify(action.args)}
                  </span>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}