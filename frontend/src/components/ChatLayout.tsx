import { useState, useEffect } from 'react';
import ContextSidebar from './chat/ContextSidebar';
import ChatWorkspace from './chat/ChatWorkspace';
import KernelVision from './chat/KernelVision';
import { useTopicStore } from '@/stores/topicStore';
import { useChatStore } from '@/stores/chatStore';

export default function ChatLayout() {
  const { topics, fetchTopics } = useTopicStore();
  const { currentTopicId, retrievedMemories } = useChatStore();
  
  const [activeTopicId, setActiveTopicId] = useState('');
  const [isContextSidebarCollapsed, setIsContextSidebarCollapsed] = useState(false);
  const [isKernelVisionCollapsed, setIsKernelVisionCollapsed] = useState(false);

  useEffect(() => {
    fetchTopics();
  }, [fetchTopics]);

  const resolvedActiveTopicId = currentTopicId || activeTopicId || topics[0]?.id || '';
  const activeTopic = topics.find((t) => t.id === resolvedActiveTopicId) || topics[0];

  return (
    <>
      <ContextSidebar
        topics={topics}
        activeTopicId={resolvedActiveTopicId}
        onTopicSelect={setActiveTopicId}
        isCollapsed={isContextSidebarCollapsed}
        onToggleCollapse={() => setIsContextSidebarCollapsed(!isContextSidebarCollapsed)}
      />
      <ChatWorkspace
        activeTopicTitle={activeTopic?.title || ''}
      />
      <KernelVision
        memories={retrievedMemories}
        isCollapsed={isKernelVisionCollapsed}
        onToggleCollapse={() => setIsKernelVisionCollapsed(!isKernelVisionCollapsed)}
      />
    </>
  );
}
