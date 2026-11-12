import { useState } from 'react';
import { ChevronDown, CheckCircle2, Loader2, XCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import type { MtpAction } from '@/types';

interface MTPCardProps {
  action: MtpAction;
}

/** 动词徽标保留协议名，说明用中文。 */
const VERB_LABEL: Record<string, string> = {
  SEARCH: '检索', READ: '读取', WRITE: '写入', UPDATE: '更新', RUN: '运行', CALL: '调用',
};

export default function MTPCard({ action }: MTPCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const verb = VERB_LABEL[action.type] ?? action.type;
  const target = action.target && action.target !== '*' ? action.target : null;

  // 颜色只表示状态：进行中为水，完成为木，没有完成为火；没有完成时写出原因
  const statusConfig = {
    pending: { icon: Loader2, label: '准备中', text: `正在解析${verb}指令`, tone: 'text-magic-metal', row: 'bg-surface-container', iconClass: 'animate-spin' },
    executing: { icon: Loader2, label: '进行中', text: `正在${verb}`, tone: 'text-magic-water', row: 'bg-surface-container', iconClass: 'animate-spin' },
    success: { icon: CheckCircle2, label: '完成', text: action.resultMessage ? `${verb}完成，${action.resultMessage}` : `${verb}完成`, tone: 'text-magic-wood', row: 'bg-surface-container', iconClass: '' },
    error: { icon: XCircle, label: '未完成', text: action.resultMessage ? `${action.type} 未完成：${action.resultMessage}` : `${action.type} 未完成`, tone: 'text-magic-fire', row: 'bg-magic-fire/10', iconClass: '' },
  };
  const config = statusConfig[action.status] || statusConfig.pending;
  const StatusIcon = config.icon;

  return (
    <motion.div layout className={`w-full overflow-hidden rounded-xl border border-line ${config.row}`}>
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        aria-expanded={isExpanded}
        className="flex w-full items-center gap-2.5 px-3 py-2 text-left"
      >
        <span className="shrink-0 rounded-md bg-primary/15 px-1.5 py-0.5 font-mono text-xs font-semibold text-primary">{action.type}</span>
        <span className="min-w-0 flex-1 truncate text-[13px] text-fg">
          {config.text}
          {target && <span className="ml-1.5 font-mono text-xs text-fg-3">{target}</span>}
        </span>
        <span className={`flex shrink-0 items-center gap-1 text-xs ${config.tone}`}>
          <StatusIcon className={`h-3 w-3 ${config.iconClass}`} aria-hidden="true" />
          {config.label}
        </span>
        <ChevronDown className={`h-4 w-4 shrink-0 text-fg-3 transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`} aria-hidden="true" />
      </button>

      {/* 展开区：显示详细信息 */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: 'easeInOut' }}
            className="overflow-hidden border-t border-line bg-field"
          >
            <div className="p-3 text-xs font-mono text-fg-2 space-y-2">
              <div className="flex">
                <span className="w-12 shrink-0 text-fg-3 select-none">目标</span>
                <span className="text-primary/80 font-semibold">{action.target || '*'}</span>
              </div>
              
              {action.params && Object.keys(action.params).length > 0 && (
                <div className="flex">
                  <span className="w-12 shrink-0 text-fg-3 select-none">参数</span>
                  <div className="text-magic-wood/70 flex flex-col gap-0.5">
                    {typeof action.params === 'string' ? (
                      <span className="line-clamp-3">{action.params}</span>
                    ) : (
                      Object.entries(action.params).map(([key, val]) => (
                        <span key={key} className="flex gap-2">
                          <span className="text-magic-wood/50 shrink-0">{key}:</span>
                          <span className="line-clamp-3 break-all">{String(val)}</span>
                        </span>
                      ))
                    )}
                  </div>
                </div>
              )}

              {action.status !== 'pending' && action.status !== 'executing' && (
                <div className="flex">
                  <span className="w-12 shrink-0 text-fg-3 select-none">结果</span>
                  <span className={action.status === 'success' ? 'text-magic-wood/80' : 'text-magic-fire/80'}>
                    {action.resultMessage || (action.status === 'success' ? '执行成功' : '执行失败')}
                  </span>
                </div>
              )}

              {action.stats && Object.keys(action.stats).length > 0 && (
                <div className="flex">
                  <span className="w-12 shrink-0 text-fg-3 select-none">统计</span>
                  <div className="text-fg-3/70 flex flex-col gap-0.5">
                    {Object.entries(action.stats).map(([key, val]) => (
                      <span key={key} className="flex gap-2">
                        <span className="text-fg-3 shrink-0">{key}:</span>
                        <span>{String(val)}</span>
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}