import { useEffect, useState } from 'react';
import ContextSidebar from './chat/ContextSidebar';
import ChatPane from './chat/ChatPane';
import WorkspaceVision from './chat/WorkspaceVision';
import { useChatStore, useChatUiStore } from '@/stores';
import { useMediaQuery } from '@/hooks/useMediaQuery';

/**
 * 三栏：窗口放得下时左右两栏都在行内；放不下右栏时它收成一条细栏，点开后作为抽屉覆盖在中间栏上；
 * 再窄时会话栏也这样处理。抽屉的开合只在本页有效，不改动持久化的折叠状态。
 */
export default function ChatLayout() {
  const { retrievedMemories, initializeSessions } = useChatStore();
  const { isContextSidebarCollapsed, toggleContextSidebar, isWorkspaceVisionCollapsed, toggleWorkspaceVision } = useChatUiStore();
  const visionInline = useMediaQuery('(min-width: 1280px)');
  const sidebarInline = useMediaQuery('(min-width: 1024px)');
  const [visionDrawerOpen, setVisionDrawerOpen] = useState(false);
  const [sidebarDrawerOpen, setSidebarDrawerOpen] = useState(false);
  useEffect(() => { void initializeSessions(); }, [initializeSessions]);

  const drawerOpen = (!visionInline && visionDrawerOpen) || (!sidebarInline && sidebarDrawerOpen);
  const closeDrawers = () => {
    setVisionDrawerOpen(false);
    setSidebarDrawerOpen(false);
  };

  return <>
    <ContextSidebar
      isCollapsed={sidebarInline ? isContextSidebarCollapsed : !sidebarDrawerOpen}
      onToggleCollapse={sidebarInline ? toggleContextSidebar : () => setSidebarDrawerOpen((open) => !open)}
      overlay={!sidebarInline}
    />
    <ChatPane />
    <WorkspaceVision
      memories={retrievedMemories}
      isCollapsed={visionInline ? isWorkspaceVisionCollapsed : !visionDrawerOpen}
      onToggleCollapse={visionInline ? toggleWorkspaceVision : () => setVisionDrawerOpen((open) => !open)}
      overlay={!visionInline}
    />
    {drawerOpen && <div aria-hidden="true" onClick={closeDrawers} className="absolute inset-0 z-30 bg-scrim" />}
  </>;
}
