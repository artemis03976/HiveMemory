import { useState, useEffect } from 'react';
import ContextSidebar from './chat/ContextSidebar';
import ChatPane from './chat/ChatPane';
import WorkspaceVision from './chat/WorkspaceVision';
import { useChatStore, useChatUiStore, useTopicStore } from '@/stores';

export default function ChatLayout() {
  const { topics, fetchTopics } = useTopicStore();
  const { currentTopicId, retrievedMemories } = useChatStore();
  const {
    isContextSidebarCollapsed,
    toggleContextSidebar,
    isWorkspaceVisionCollapsed,
    toggleWorkspaceVision,
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
      <ChatPane
        activeTopicTitle={activeTopic?.title || ''}
      />
      <WorkspaceVision
        memories={retrievedMemories}
        isCollapsed={isWorkspaceVisionCollapsed}
        onToggleCollapse={toggleWorkspaceVision}
      />
    </>
  );
}
