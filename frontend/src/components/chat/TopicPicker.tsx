import { useEffect, useState } from 'react';
import { Hash } from 'lucide-react';
import { fetchTopics } from '@/services/topicApi';
import type { Topic } from '@/types';

export interface DesignatedTopic {
  id: string;
  title: string;
}

interface TopicPickerProps {
  onPick: (topic: DesignatedTopic) => void;
  onClose: () => void;
}

/** 输入 # 或点击话题按钮时弹出：列出短期话题池中的话题，按标题筛选后选一个作为这一轮的话题。 */
export default function TopicPicker({ onPick, onClose }: TopicPickerProps) {
  const [topics, setTopics] = useState<Topic[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [query, setQuery] = useState('');

  useEffect(() => {
    let alive = true;
    fetchTopics()
      .then((list) => { if (alive) setTopics(list); })
      .catch(() => { if (alive) setFailed(true); });
    return () => { alive = false; };
  }, []);

  const shown = (topics ?? []).filter((t) => t.title.toLowerCase().includes(query.trim().toLowerCase()));
  const pick = (topic: Topic) => onPick({ id: topic.id, title: topic.title === topic.id ? '未命名话题' : topic.title });

  return (
    <div
      role="dialog"
      aria-label="引用话题"
      className="absolute bottom-full left-0 z-50 mb-2 flex w-[360px] flex-col gap-1 rounded-2xl border border-line-strong bg-surface-container p-2 shadow-pop"
      onKeyDown={(e) => {
        if (e.key === 'Escape') onClose();
      }}
    >
      <label className="block">
        <span className="sr-only">按标题筛选话题</span>
        <input
          autoFocus
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && shown.length > 0) {
              e.preventDefault();
              pick(shown[0]);
            }
          }}
          placeholder="短期话题池 · 按标题筛选"
          className="h-8 w-full rounded-lg border border-line bg-field px-2.5 text-[13px] text-fg placeholder:text-fg-3 outline-none focus:border-primary/50"
        />
      </label>
      <div role="listbox" aria-label="话题" className="scrollbar-thin flex max-h-64 flex-col gap-0.5 overflow-y-auto">
        {topics === null && !failed && <span className="px-2 py-3 text-[13px] text-fg-3">少女祈祷中...</span>}
        {failed && <span className="px-2 py-3 text-[13px] text-fg-3">话题暂不可用</span>}
        {topics !== null && shown.length === 0 && <span className="px-2 py-3 text-[13px] text-fg-3">没有匹配的话题</span>}
        {shown.map((topic) => (
          <button
            key={topic.id}
            role="option"
            aria-selected={false}
            onClick={() => pick(topic)}
            className="flex flex-col gap-0.5 rounded-xl px-2.5 py-2 text-left hover:bg-surface-container-high"
          >
            <span className="flex items-center gap-1.5 text-[13px] font-semibold text-fg">
              <Hash className="h-3 w-3 text-accent" aria-hidden="true" />
              {topic.title === topic.id ? '未命名话题' : topic.title}
            </span>
            {topic.summary && <span className="line-clamp-1 text-xs text-fg-3">{topic.summary}</span>}
          </button>
        ))}
      </div>
    </div>
  );
}
