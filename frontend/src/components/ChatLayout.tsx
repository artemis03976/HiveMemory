import { useState, useEffect } from 'react';
import type { MemoryAtom } from '@/types';
import ContextSidebar from './chat/ContextSidebar';
import ChatWorkspace from './chat/ChatWorkspace';
import KernelVision from './chat/KernelVision';
import { useTopicStore } from '@/stores/topicStore';
import { useChatStore } from '@/stores/chatStore';

interface ChatLayoutProps {
  memories: MemoryAtom[];
}

export default function ChatLayout({ memories }: ChatLayoutProps) {
  const { topics, fetchTopics } = useTopicStore();
  const { currentTopicId } = useChatStore();
  
  const [activeTopicId, setActiveTopicId] = useState('');
  const [isContextSidebarCollapsed, setIsContextSidebarCollapsed] = useState(false);
  const [isKernelVisionCollapsed, setIsKernelVisionCollapsed] = useState(false);

  useEffect(() => {
    fetchTopics();
  }, [fetchTopics]);

  useEffect(() => {
    if (currentTopicId) {
      setActiveTopicId(currentTopicId);
    } else if (topics.length > 0 && !activeTopicId) {
      setActiveTopicId(topics[0].id);
    }
  }, [currentTopicId, topics, activeTopicId]);

  const activeTopic = topics.find((t) => t.id === activeTopicId) || topics[0];

  return (
    <>
      <ContextSidebar
        topics={topics}
        activeTopicId={activeTopicId}
        onTopicSelect={setActiveTopicId}
        isCollapsed={isContextSidebarCollapsed}
        onToggleCollapse={() => setIsContextSidebarCollapsed(!isContextSidebarCollapsed)}
      />
      <ChatWorkspace
        activeTopicTitle={activeTopic?.title || ''}
      />
      <KernelVision
        memories={memories}
        isCollapsed={isKernelVisionCollapsed}
        onToggleCollapse={() => setIsKernelVisionCollapsed(!isKernelVisionCollapsed)}
      />
    </>
  );
}