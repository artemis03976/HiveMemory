import { useState, useEffect } from 'react';
import ContextSidebar from './chat/ContextSidebar';
import ChatWorkspace from './chat/ChatWorkspace';
import KernelVision from './chat/KernelVision';
import { useTopicStore } from '@/stores/topicStore';
import { useChatStore } from '@/stores/chatStore';
import { useChatUiStore } from '@/stores/chatUiStore';

export default function ChatLayout() {
  const { topics, fetchTopics } = useTopicStore();
  const { currentTopicId, retrievedMemories } = useChatStore();
  const {
    isContextSidebarCollapsed,
    toggleContextSidebar,
    isKernelVisionCollapsed,
    toggleKernelVision,
  } = useChatUiStore();

  const [activeTopicId, setActiveTopicId] = useState('');

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
        onToggleCollapse={toggleContextSidebar}
      />
      <ChatWorkspace
        activeTopicTitle={activeTopic?.title || ''}
      />
      <KernelVision
        memories={retrievedMemories}
        isCollapsed={isKernelVisionCollapsed}
        onToggleCollapse={toggleKernelVision}
      />
    </>
  );
}
