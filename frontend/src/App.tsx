/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';
import GlobalNavBar from './components/GlobalNavBar';
import ChatLayout from './components/ChatLayout';
import MemoryGarden from './components/MemoryGarden';
import SettingsPanel from './components/SettingsPanel';
import type { MemoryAtom, NavTab } from './types';

const MOCK_MEMORIES: MemoryAtom[] = [
  {
    id: 'mem1',
    alias: 'fact_project_env',
    summary: 'HiveMemory 项目使用 Python 3.12 和 FastAPI 框架',
    tags: ['python', 'config', 'environment'],
    content: 'Full memory content...',
    confidence_score: 0.95,
    title: '项目环境配置',
    memory_type: 'FACT',
    status: 'Active',
    access_count: 5,
    isPinned: true,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    sourceAgent: 'System',
    vitality_score: 0.9,
    user_id: 'default'
  },
  {
    id: 'mem2',
    alias: 'code_mtp_protocol',
    summary: 'MTP 协议的实现细节和使用方法',
    tags: ['mtp', 'protocol', 'code'],
    content: 'Full memory content...',
    confidence_score: 0.88,
    title: 'MTP协议解析',
    memory_type: 'CODE_SNIPPET',
    status: 'Active',
    access_count: 12,
    isPinned: false,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    sourceAgent: 'System',
    vitality_score: 0.8,
    user_id: 'default'
  },
];

export default function App() {
  const [activeNavTab, setActiveNavTab] = useState<NavTab>('chat');

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background">
      <GlobalNavBar activeTab={activeNavTab} onTabChange={setActiveNavTab} />
      <div className="flex flex-1 ml-16 h-full overflow-hidden">
        {activeNavTab === 'chat' && (
          <ChatLayout
            memories={MOCK_MEMORIES}
          />
        )}
        {activeNavTab === 'database' && (
          <MemoryGarden />
        )}
        {activeNavTab === 'settings' && (
          <SettingsPanel />
        )}
      </div>
    </div>
  );
}