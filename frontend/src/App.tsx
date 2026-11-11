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
import RunView from './components/runs/RunView';
import DynamicToast from './components/common/DynamicToast';
import ConnectionBanner from './components/common/ConnectionBanner';
import { useChatUiStore } from '@/stores';
import { resolveTheme } from '@/stores/ui/theme';
import { useMediaQuery } from '@/hooks/useMediaQuery';
import { useUnsavedSettings } from '@/stores/ui/unsavedSettings';
import type { NavTab } from './types';

export default function App() {
  const { activeNavTab, setActiveNavTab, theme } = useChatUiStore();
  const systemPrefersDark = useMediaQuery('(prefers-color-scheme: dark)');
  const resolvedTheme = resolveTheme(theme, systemPrefersDark);

  // 设置页有未保存的更改时，离开前先确认，免得更改随页面卸载丢失
  const changeTab = (tab: NavTab) => {
    const leavingDirtySettings = activeNavTab === 'settings' && tab !== 'settings' && useUnsavedSettings.getState().dirty;
    if (leavingDirtySettings && !window.confirm('设置还有未保存的更改，离开后会丢失。确定离开吗？')) return;
    setActiveNavTab(tab);
  };

  // 将主题同步到 <html data-theme>，驱动 CSS 变量切换（useLayoutEffect 避免首屏闪烁）
  useLayoutEffect(() => {
    document.documentElement.dataset.theme = resolvedTheme;
  }, [resolvedTheme]);

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background">
      <DynamicToast />
      <GlobalNavBar activeTab={activeNavTab} onTabChange={changeTab} resolvedTheme={resolvedTheme} />
      <div className="flex flex-1 flex-col ml-16 h-full min-w-0 overflow-hidden">
        <ConnectionBanner />
        <div className="relative flex flex-1 min-h-0 min-w-0">
        {activeNavTab === 'chat' && (
          <ChatLayout />
        )}
        {activeNavTab === 'database' && (
          <MemoryLibrary />
        )}
        {activeNavTab === 'agents' && (
          <AgentManagement />
        )}
        {activeNavTab === 'terminal' && (
          <RunView />
        )}
        {activeNavTab === 'settings' && (
          <SettingsPanel />
        )}
        </div>
      </div>
    </div>
  );
}
