import { useState } from 'react';
import { BookmarkCheck, Trash2 } from 'lucide-react';
import type { Topic } from '@/types';
import { formatTimeAgo } from '@/utils/timeAgo';

interface TopicCardProps {
  topic: Topic;
  isActive: boolean;
  onClick: () => void;
  onSettle?: (id: string) => void;
  onDelete?: (id: string) => void;
}

export default function TopicCard({
  topic,
  isActive,
  onClick,
  onSettle,
  onDelete
}: TopicCardProps) {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className={`group px-4 py-3 rounded-xl cursor-pointer transition-all relative overflow-hidden ${
        isActive ? 'bg-ink/5 border border-line' : 'hover:bg-ink/5 border border-transparent'
      }`}
    >
      {/* 活跃状态指示灯条 */}
      {/* {isActive && (
        <div className="absolute left-0 top-1/2 -translate-y-1/2 h-4/5 w-[3px] rounded-r-full bg-primary" />
      )} */}
      
      <div className="flex flex-col relative z-10">
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1 min-w-0">
            {/* 卡片顶部区域 */}
            <div className="flex items-center gap-2 mb-1.5">
              {/* 状态指示灯 */}
              <div className={`w-2 h-2 rounded-full transition-all duration-500 shrink-0 ${
                isActive
                  ? 'bg-magic-wood animate-pulse'
                  : 'bg-magic-metal'
              }`} />
              
              {/* 标题 */}
              <h3 className={`text-sm font-semibold truncate transition-colors ${
                isActive ? 'text-fg' : 'text-fg-2 group-hover:text-fg-2'
              }`}>
                {topic.title}
              </h3>
            </div>
          </div>

          {/* hover 状态操作按钮 */}
          <div className={`flex gap-1 transition-all duration-200 ${
            isHovered ? "opacity-100 translate-x-0" : "opacity-0 translate-x-2 pointer-events-none"
          }`}>
            {/* 话题结算按钮 */}
            <button
              className="p-1.5 rounded-md hover:bg-ink/10 hover:text-fg text-fg-3 transition-colors"
              aria-label="Settle topic"
              onClick={(e) => { 
                e.stopPropagation(); 
                if (onSettle) onSettle(topic.id);
              }}
            >
              <BookmarkCheck className="w-3.5 h-3.5" />
            </button>

            {/* 话题删除按钮 */}
            <button
              className="p-1.5 rounded-md hover:bg-magic-fire/20 text-fg-3 hover:text-magic-fire transition-colors"
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
            isActive ? 'text-fg-2' : 'text-fg-3/80'
          }`}>
            {topic.summary}
          </p>
        )}
        
        {/* Meta 信息 */}
        <div className={`flex items-center gap-2 mt-2.5 text-xs font-medium transition-colors ${
          isActive ? 'text-fg-3' : 'text-fg-3'
        }`}>
          <span>{topic.messageCount || 0} msgs</span>
          <span>•</span>
          <span>{formatTimeAgo(topic.lastActive)}</span>
          <span>•</span>
          <span>{topic.model}</span>
        </div>
      </div>
    </div>
  );
}
