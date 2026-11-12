import { useState } from 'react';
import {
  ChevronDown, CheckCircle2, Loader2, XCircle,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import type { SubAgentBlock } from '@/types';
import type { AgentProfile } from '@/hooks/useAgents';
import InlineBlockList from './InlineBlockList';
import { activateOnKey } from '@/utils/activateOnKey';

interface SubAgentCardProps {
  block: SubAgentBlock;
  /** 被调用的 Agent，名称与图标取自 Profile 列表。 */
  agent: AgentProfile;
}

const STATUS_CONFIG = {
  running: {
    icon: Loader2,
    text: '执行中...',
    colorClass: 'text-magic-water',
    iconClass: 'animate-spin',
    borderClass: 'border-magic-water/30',
    bgClass: 'bg-magic-water/5',
  },
  completed: {
    icon: CheckCircle2,
    text: '已完成',
    colorClass: 'text-magic-wood',
    iconClass: '',
    borderClass: 'border-magic-wood/30',
    bgClass: 'bg-magic-wood/5',
  },
  error: {
    icon: XCircle,
    text: '执行失败',
    colorClass: 'text-magic-fire',
    iconClass: '',
    borderClass: 'border-magic-fire/30',
    bgClass: 'bg-magic-fire/5',
  },
};

export default function SubAgentCard({ block, agent }: SubAgentCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const AgentIcon = agent.avatarIcon;
  const agentName = agent.name;
  const agentColor = 'text-primary';

  const config = STATUS_CONFIG[block.status];
  const StatusIcon = config.icon;

  return (
    <motion.div
      layout
      className={`w-full overflow-hidden rounded-xl border backdrop-blur-md transition-colors duration-300 ${config.borderClass} ${config.bgClass}`}
    >
      {/* Header */}
      <div
        role="button"
        tabIndex={0}
        aria-expanded={isExpanded}
        className="flex items-center justify-between px-4 py-3 cursor-pointer select-none"
        onClick={() => setIsExpanded(!isExpanded)}
        onKeyDown={activateOnKey(() => setIsExpanded(!isExpanded))}
      >
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center bg-primary/10 border border-primary/20">
            <AgentIcon className={`w-4 h-4 ${agentColor}`} />
          </div>
          <div className="flex flex-col">
            <span className="text-sm font-semibold text-fg font-manrope">
              {agentName}
            </span>
            <div className="flex items-center gap-1.5 mt-0.5">
              <StatusIcon className={`w-3 h-3 ${config.colorClass} ${config.iconClass}`} />
              <span className={`text-xs font-medium ${config.colorClass}`}>
                {config.text}
              </span>
            </div>
          </div>
        </div>

        <span aria-hidden="true" className={`transition-transform duration-200 text-fg-3 opacity-60 ${isExpanded ? 'rotate-180' : ''}`}>
          <ChevronDown className="w-4 h-4" />
        </span>
      </div>

      {/* Task description */}
      <div className="px-4 pb-2.5">
        <p className="text-xs text-fg-3/80 line-clamp-2 leading-relaxed">
          <span className="text-fg-3 font-medium">Task: </span>
          {block.task}
        </p>
      </div>

      {/* Expandable body */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: 'easeInOut' }}
            className="overflow-hidden border-t border-line"
          >
            <div className="p-4 space-y-3 max-h-[400px] overflow-y-auto scrollbar-thin">
              {block.contentBlocks.length === 0 && block.status === 'running' && (
                <div className="flex items-center gap-2 text-magic-water/60 text-sm">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>子代理正在处理中...</span>
                </div>
              )}

              <InlineBlockList
                blocks={block.contentBlocks}
                textClassName="text-sm leading-relaxed text-fg-2"
              />

              {/* 恢复历史只有根帧回填结果，保留原文显示，不能伪造子帧流。 */}
              {block.contentBlocks.length === 0 && block.finalText && (
                <div className="text-sm text-fg-2 whitespace-pre-wrap break-words">{block.finalText}</div>
              )}
              {block.contentBlocks.length === 0 && !block.finalText && block.status === 'error' && (
                <div className="flex items-center gap-2 text-magic-fire/70 text-sm">
                  <XCircle className="w-4 h-4" />
                  <span>子代理执行过程中出现错误</span>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
