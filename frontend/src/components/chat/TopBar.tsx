import { MoreVertical } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useChatStore, useTopicStore } from '@/stores';
import { fetchFoldTokenThreshold } from '@/services/configApi';

interface TopBarProps {
  activeTopicTitle: string;
}

export default function TopBar({ activeTopicTitle }: TopBarProps) {
  const { currentTopicId } = useChatStore();
  const { topics } = useTopicStore();
  const [maxTokens, setMaxTokens] = useState(32768);

  useEffect(() => {
    fetchFoldTokenThreshold().then(setMaxTokens).catch(() => setMaxTokens(32768));
  }, []);

  const activeTopic = topics.find(t => t.id === currentTopicId);
  const currentTokens = activeTopic?.totalTokens || 0; 
  
  const percentage = Math.min(100, Math.round((currentTokens / maxTokens) * 100));
  
  // Format tokens for display (e.g. 1245 -> 1.2k)
  const formatTokens = (tokens: number) => {
    if (tokens < 1000) return tokens.toString();
    return (tokens / 1000).toFixed(1) + 'k';
  };

  return (
    <div className="h-14 flex items-center px-8 border-b border-white/5 justify-between bg-surface-dim shrink-0">
      {/* 活跃话题信息 */}
      <div className="flex items-center gap-3">
        <div className="w-2 h-2 rounded-full bg-primary animate-pulse shadow-[0_0_8px_#c59aff]"></div>
        <h2 className="font-manrope font-semibold text-lg tracking-tight">{activeTopicTitle}</h2>
      </div>
      
      {/* 上下文剩余信息 */}
      <div className="flex items-center gap-6">
        <div className="flex flex-col gap-1.5 w-48">
          <div className="flex items-center justify-between">
            <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">Context Window</span>
            <span className="text-[10px] text-slate-300 font-mono">{formatTokens(currentTokens)} / {formatTokens(maxTokens)}</span>
          </div>
          <div className="h-1 w-full bg-black/40 rounded-full overflow-hidden border border-white/5">
            <div
              className={`h-full transition-all duration-500 ease-out ${
                percentage > 90 ? 'bg-magic-fire shadow-[0_0_8px_hsla(340,80%,60%,0.5)]' :
                percentage > 75 ? 'bg-magic-metal shadow-[0_0_8px_hsla(52,82%,73%,0.5)]' :
                'bg-primary shadow-[0_0_8px_#c59aff]'
              }`}
              style={{ width: `${percentage}%` }}
            ></div>
          </div>
        </div>
        <button className="text-slate-400 hover:text-white transition-colors ml-2">
          <MoreVertical className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
