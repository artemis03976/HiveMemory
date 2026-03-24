import { MessageSquare } from 'lucide-react';
import type { Topic } from '@/types';
import TopicCard from './TopicCard';
import { useTopicStore } from '@/stores/topicStore';

interface TopicTabProps {
  topics: Topic[];
  activeTopicId: string;
  onTopicSelect: (id: string) => void;
}

export default function TopicTab({
  topics,
  activeTopicId,
  onTopicSelect
}: TopicTabProps) {
  const { isLoading, archiveTopic, deleteTopic } = useTopicStore();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12 text-slate-500 text-sm">
        <span className="animate-pulse">加载话题中...</span>
      </div>
    );
  }

  if (topics.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 gap-3 text-slate-500">
        <MessageSquare className="w-8 h-8 opacity-30" />
        <p className="text-sm">暂无活跃话题</p>
        <p className="text-xs opacity-60">开始对话后，话题将自动出现</p>
      </div>
    );
  }

  return (
    <>
      {topics.map((topic) => (
        <TopicCard
          key={topic.id}
          topic={topic}
          isActive={topic.id === activeTopicId}
          onClick={() => onTopicSelect(topic.id)}
          onArchive={() => archiveTopic(topic.id)}
          onDelete={() => deleteTopic(topic.id)}
        />
      ))}
    </>
  );
}