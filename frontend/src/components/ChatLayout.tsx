import { useEffect } from 'react';
import ContextSidebar from './chat/ContextSidebar';
import ChatPane from './chat/ChatPane';
import WorkspaceVision from './chat/WorkspaceVision';
import { useChatStore, useChatUiStore } from '@/stores';

export default function ChatLayout() {
  const { retrievedMemories, initializeSessions } = useChatStore();
  const { isContextSidebarCollapsed, toggleContextSidebar, isWorkspaceVisionCollapsed, toggleWorkspaceVision } = useChatUiStore();
  useEffect(() => { void initializeSessions(); }, [initializeSessions]);
  return <>
    <ContextSidebar isCollapsed={isContextSidebarCollapsed} onToggleCollapse={toggleContextSidebar} />
    <ChatPane />
    <WorkspaceVision memories={retrievedMemories} isCollapsed={isWorkspaceVisionCollapsed} onToggleCollapse={toggleWorkspaceVision} />
  </>;
}
