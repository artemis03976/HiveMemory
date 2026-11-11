import { useEffect, useState } from 'react';
import ContextSidebar from './chat/ContextSidebar';
import ChatPane from './chat/ChatPane';
import WorkspaceVision from './chat/WorkspaceVision';
import { useChatStore, useWorkspaceStore } from '@/stores';
import { useMediaQuery } from '@/hooks/useMediaQuery';

type Drawer = 'sessions' | 'vision';

/**
 * 三栏：窗口放得下时左右两栏都在行内；放不下右栏时，它改由中间栏头部的按钮打开，作为抽屉覆盖在中间栏上；
 * 再窄时会话栏也这样处理。点遮罩或按 Esc 关闭抽屉。
 */
export default function ChatLayout() {
  const { retrievedMemories, initializeSessions } = useChatStore();
  const connectRuntimeEvents = useWorkspaceStore((state) => state.connectRuntimeEvents);
  const visionInline = useMediaQuery('(min-width: 1280px)');
  const sidebarInline = useMediaQuery('(min-width: 1024px)');
  const [drawer, setDrawer] = useState<Drawer | null>(null);
  useEffect(() => { void initializeSessions(); }, [initializeSessions]);
  // 右栏收为抽屉时也要接收记忆任务的事件，事件流不随右栏的显示与否开关
  useEffect(() => { connectRuntimeEvents(); }, [connectRuntimeEvents]);

  const sessionsDrawer = !sidebarInline && drawer === 'sessions';
  const visionDrawer = !visionInline && drawer === 'vision';
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
    {(sidebarInline || sessionsDrawer) && <ContextSidebar onClose={sessionsDrawer ? closeDrawer : undefined} />}
    <ChatPane
      onOpenSessions={sidebarInline ? undefined : () => setDrawer('sessions')}
      onOpenVision={visionInline ? undefined : () => setDrawer('vision')}
    />
    {(visionInline || visionDrawer) && (
      <WorkspaceVision memories={retrievedMemories} onClose={visionDrawer ? closeDrawer : undefined} />
    )}
    {(sessionsDrawer || visionDrawer) && <div aria-hidden="true" onClick={closeDrawer} className="absolute inset-0 z-30 bg-scrim" />}
  </>;
}
