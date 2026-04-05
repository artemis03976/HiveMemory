import { useState } from 'react';
import { Archive, Trash2 } from 'lucide-react';
import type { Topic } from '@/types';

interface TopicCardProps {
  topic: Topic;
  isActive: boolean;
  onClick: () => void;
  onArchive?: (id: string) => void;
  onDelete?: (id: string) => void;
}

export default function TopicCard({
  topic,
  isActive,
  onClick,
  onArchive,
  onDelete
}: TopicCardProps) {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className={`group px-4 py-3 rounded-xl cursor-pointer transition-all relative overflow-hidden ${
        isActive ? 'bg-white/5 ghost-border shadow-[0_4px_20px_rgba(139,92,246,0.1)]' : 'hover:bg-white/5 border border-transparent'
      }`}
    >
      {/* 活跃状态指示灯条 */}
      {/* {isActive && (
        <div className="absolute left-0 top-1/2 -translate-y-1/2 h-4/5 w-[3px] rounded-r-full bg-primary shadow-[0_0_10px_rgba(139,92,246,0.9)]" />
      )} */}
      
      <div className="flex flex-col relative z-10">
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1 min-w-0">
            {/* 卡片顶部区域 */}
            <div className="flex items-center gap-2 mb-1.5">
              {/* 状态指示灯 */}
              <div className={`w-2 h-2 rounded-full transition-all duration-500 shrink-0 ${
                isActive
                  ? 'bg-magic-wood shadow-[0_0_8px_rgba(16,185,129,0.5)] animate-pulse'
                  : topic.willEvict
                    ? 'bg-magic-earth shadow-[0_0_6px_rgba(245,158,11,0.4)]'
                    : 'bg-magic-metal'
              }`} />
              
              {/* 标题 */}
              <h3 className={`text-sm font-semibold truncate transition-colors ${
                isActive ? 'text-white' : 'text-slate-300 group-hover:text-white/95'
              }`}>
                {topic.title}
              </h3>
            </div>
          </div>

          {/* hover 状态操作按钮 */}
          <div className={`flex gap-1 transition-all duration-200 ${
            isHovered ? "opacity-100 translate-x-0" : "opacity-0 translate-x-2 pointer-events-none"
          }`}>
            {/* 话题归档按钮 */}
            <button
              className="p-1.5 rounded-md hover:bg-white/10 hover:text-white text-slate-500 transition-colors"
              aria-label="Archive topic"
              onClick={(e) => { 
                e.stopPropagation(); 
                if (onArchive) onArchive(topic.id);
              }}
            >
              <Archive className="w-3.5 h-3.5" />
            </button>

            {/* 话题删除按钮 */}
            <button
              className="p-1.5 rounded-md hover:bg-magic-fire/20 text-slate-500 hover:text-magic-fire transition-colors"
              aria-label="Delete topic"
              onClick={(e) => { 
                e.stopPropagation(); 
                if (onDelete) onDelete(topic.id);
              }}
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
        
        {/* 话题内容摘要 - 现在占满全宽 */}
        {topic.summary && (
          <p className={`text-xs line-clamp-2 leading-relaxed transition-colors w-full mt-1 ${
            isActive ? 'text-white/70' : 'text-slate-500/80'
          }`}>
            {topic.summary}
          </p>
        )}
        
        {/* Meta 信息 */}
        <div className={`flex items-center gap-2 mt-2.5 text-[10px] font-medium transition-colors uppercase tracking-tighter ${
          isActive ? 'text-white/40' : 'text-slate-500'
        }`}>
          <span>{topic.messageCount || 0} msgs</span>
          <span>•</span>
          <span>{topic.activeNow ? '在线' : topic.timeAgo}</span>
          <span>•</span>
          <span>{topic.model}</span>
        </div>
      </div>
    </div>
  );
}