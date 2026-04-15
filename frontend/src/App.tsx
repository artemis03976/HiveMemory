/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';
import GlobalNavBar from './components/GlobalNavBar';
import ChatLayout from './components/ChatLayout';
import MemoryLibrary from './components/MemoryLibrary';
import SettingsPanel from './components/SettingsPanel';
import DynamicToast from './components/common/DynamicToast';
import type { NavTab } from './types';

export default function App() {
  const [activeNavTab, setActiveNavTab] = useState<NavTab>('chat');

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
        {activeNavTab === 'settings' && (
          <SettingsPanel />
        )}
      </div>
    </div>
  );
}