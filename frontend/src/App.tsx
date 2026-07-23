/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useLayoutEffect } from 'react';
import GlobalNavBar from './components/GlobalNavBar';
import ChatLayout from './components/ChatLayout';
import MemoryLibrary from './components/MemoryLibrary';
import SettingsPanel from './components/SettingsPanel';
import AgentManagement from './components/AgentManagement';
import DynamicToast from './components/common/DynamicToast';
import { useChatUiStore } from '@/stores';

export default function App() {
  const { activeNavTab, setActiveNavTab, theme } = useChatUiStore();

  // 将主题同步到 <html data-theme>，驱动 CSS 变量切换（useLayoutEffect 避免首屏闪烁）
  useLayoutEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background">
      <DynamicToast />
      <GlobalNavBar activeTab={activeNavTab} onTabChange={setActiveNavTab} />
      <div className="flex flex-1 ml-16 h-full overflow-hidden">
        {activeNavTab === 'chat' && (
          <ChatLayout />
        )}
        {activeNavTab === 'database' && (
          <MemoryLibrary />
        )}
        {activeNavTab === 'agents' && (
          <AgentManagement />
        )}
        {activeNavTab === 'settings' && (
          <SettingsPanel />
        )}
      </div>
    </div>
  );
}
