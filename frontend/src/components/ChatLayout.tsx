import { useEffect, useState } from 'react';
import ContextSidebar from './chat/ContextSidebar';
import ChatPane from './chat/ChatPane';
import WorkspaceVision from './chat/WorkspaceVision';
import { useChatStore, useChatUiStore, useWorkspaceStore } from '@/stores';
import { useMediaQuery } from '@/hooks/useMediaQuery';

type Drawer = 'sessions' | 'vision';

/**
 * 三栏：窗口放得下时左右两栏都在行内，各自可以收起，收起后由中间栏头部的按钮展开；
 * 放不下右栏时，它改由同一个按钮打开，作为抽屉覆盖在中间栏上，再窄时会话栏也这样处理。
 * 点遮罩或按 Esc 关闭抽屉。收起状态只在行内生效，抽屉的开合不改动它。
 */
export default function ChatLayout() {
  const { retrievedMemories, initializeSessions } = useChatStore();
  const { isContextSidebarCollapsed, toggleContextSidebar, isWorkspaceVisionCollapsed, toggleWorkspaceVision } = useChatUiStore();
  const connectRuntimeEvents = useWorkspaceStore((state) => state.connectRuntimeEvents);
  const visionInline = useMediaQuery('(min-width: 1280px)');
  const sidebarInline = useMediaQuery('(min-width: 1024px)');
  const [drawer, setDrawer] = useState<Drawer | null>(null);
  useEffect(() => { void initializeSessions(); }, [initializeSessions]);
  // 右栏收为抽屉时也要接收记忆任务的事件，事件流不随右栏的显示与否开关
  useEffect(() => { connectRuntimeEvents(); }, [connectRuntimeEvents]);

  const sessionsDrawer = !sidebarInline && drawer === 'sessions';
  const visionDrawer = !visionInline && drawer === 'vision';
  const sessionsShown = sidebarInline ? !isContextSidebarCollapsed : sessionsDrawer;
  const visionShown = visionInline ? !isWorkspaceVisionCollapsed : visionDrawer;
  const closeDrawer = () => setDrawer(null);

  useEffect(() => {
    if (!sessionsDrawer && !visionDrawer) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setDrawer(null);
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [sessionsDrawer, visionDrawer]);

  return <>
    {sessionsShown && <ContextSidebar overlay={sessionsDrawer} onHide={sessionsDrawer ? closeDrawer : toggleContextSidebar} />}
    <ChatPane
      onOpenSessions={sessionsShown ? undefined : sidebarInline ? toggleContextSidebar : () => setDrawer('sessions')}
      onOpenVision={visionShown ? undefined : visionInline ? toggleWorkspaceVision : () => setDrawer('vision')}
    />
    {visionShown && (
      <WorkspaceVision memories={retrievedMemories} overlay={visionDrawer} onHide={visionDrawer ? closeDrawer : toggleWorkspaceVision} />
    )}
    {(sessionsDrawer || visionDrawer) && <div aria-hidden="true" onClick={closeDrawer} className="absolute inset-0 z-30 bg-scrim" />}
  </>;
}
