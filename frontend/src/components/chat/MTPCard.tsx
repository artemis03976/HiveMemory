import { useState } from 'react';
import { 
  Terminal, Search, BookOpen, Edit3, RefreshCw, Box,
  ChevronDown, CheckCircle2, Loader2, XCircle, Wand2 
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import type { MtpAction } from '@/types';

interface MTPCardProps {
  action: MtpAction;
}

export default function MTPCard({ action }: MTPCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  // 重新设计的状态配置：根据“五行”色彩体系进行重塑
  const statusConfig = {
    pending: {
      icon: Loader2,
      text: '正在解析指令...',
      bgClass: 'bg-magic-metal/10 border-magic-metal/30 shadow-[inset_0_0_12px_hsla(40,90%,55%,0.1)]',
      textClass: 'text-magic-metal',
      iconClass: 'animate-spin',
    },
    executing: {
      icon: Wand2, // 使用魔杖图标代表施法/执行
      text: `正在执行 ${action.type}...`,
      bgClass: 'bg-magic-water/10 border-magic-water/30 shadow-[inset_0_0_12px_hsla(190,90%,50%,0.15)]',
      textClass: 'text-magic-water',
      iconClass: 'animate-pulse drop-shadow-[0_0_8px_hsla(190,90%,50%,0.6)]',
    },
    success: {
      icon: CheckCircle2,
      text: `${action.type} 执行完毕`,
      bgClass: 'bg-magic-wood/10 border-magic-wood/30 shadow-[inset_0_0_12px_hsla(150,80%,45%,0.1)]',
      textClass: 'text-magic-wood',
      iconClass: '',
    },
    error: {
      icon: XCircle,
      text: `${action.type} 中断`,
      bgClass: 'bg-magic-fire/12 border-magic-fire/30 shadow-[inset_0_0_12px_hsla(340,80%,60%,0.1)]',
      textClass: 'text-magic-fire',
      iconClass: '',
    },
  };

  const getInstructionIcon = () => {
    switch (action.type) {
      case 'RUN': return <Terminal className="w-3.5 h-3.5 text-primary" />;
      case 'SEARCH': return <Search className="w-3.5 h-3.5 text-primary" />;
      case 'READ': return <BookOpen className="w-3.5 h-3.5 text-primary" />;
      case 'WRITE': return <Edit3 className="w-3.5 h-3.5 text-primary" />;
      case 'UPDATE': return <RefreshCw className="w-3.5 h-3.5 text-primary" />;
      default: return <Box className="w-3.5 h-3.5 text-primary" />;
    }
  };

  const config = statusConfig[action.status] || statusConfig.pending;
  const StatusIcon = config.icon;

  return (
    <motion.div
      layout
      className={`mt-3 w-full max-w-md overflow-hidden rounded-xl border backdrop-blur-md transition-colors duration-300 ${config.bgClass}`}
    >
      {/* 点击展开简略参数 */}
      <div
        className="flex items-center justify-between px-3 py-2 cursor-pointer select-none group"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-3">
          {/* 动态指令图标 */}
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-primary/10 border border-primary/20">
            {getInstructionIcon()}
          </div>
          
          <div className="flex flex-col">
            {/* 指令名 & 工具名 */}
            <span className="text-[11px] font-bold tracking-widest uppercase text-slate-300 font-manrope">
              {action.type} <span className="opacity-60">[{action.command}]</span>
            </span>

            {/* 状态胶囊 */}
            <div className="flex items-center gap-1.5 mt-0.5">
              <StatusIcon className={`w-3 h-3 ${config.textClass} ${config.iconClass}`} />
              <span className={`text-[9px] font-medium uppercase tracking-tighter ${config.textClass}`}>
                {config.text}
              </span>
            </div>
          </div>
        </div>

        {/* 下拉展开箭头 */}
        <button className={`transition-transform duration-200 ${config.textClass} opacity-50 ${isExpanded ? 'rotate-180' : ''}`}>
          <ChevronDown className="w-4 h-4" />
        </button>
      </div>

      {/* 展开区：仅显示输入参数 (Target & Args)，不显示长篇 Response */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: 'easeInOut' }}
            className="overflow-hidden border-t border-black/10 bg-black/30"
          >
            <div className="p-3 text-[12px] font-mono text-slate-400/80 space-y-1.5">
              <div className="flex">
                <span className="w-10 shrink-0 text-emerald-500/40 select-none">CMD </span>
                <span className="text-primary/80 font-semibold">{action.command}</span>
              </div>
              {action.params && Object.keys(action.params).length > 0 && (
                <div className="flex">
                  <span className="w-12 shrink-0 opacity-50">PARAMS |</span>
                  <span className="text-emerald-300/70 line-clamp-3">
                    {typeof action.params === 'string' ? action.params : JSON.stringify(action.params)}
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