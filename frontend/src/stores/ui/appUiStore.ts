import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { NavTab } from '@/types';

type ContextSidebarTab = 'topics' | 'config';
type KernelVisionTab = 'context' | 'memory-runtime' | 'terminal';
type ThemeMode = 'dark' | 'light';

interface ChatUiStore {
  // Global navigation
  activeNavTab: NavTab;
  setActiveNavTab: (tab: NavTab) => void;

  // Theme
  theme: ThemeMode;
  toggleTheme: () => void;

  // Chat layout panels
  isContextSidebarCollapsed: boolean;
  setContextSidebarCollapsed: (collapsed: boolean) => void;
  toggleContextSidebar: () => void;

  isKernelVisionCollapsed: boolean;
  setKernelVisionCollapsed: (collapsed: boolean) => void;
  toggleKernelVision: () => void;

  // Sidebar tabs
  contextSidebarTab: ContextSidebarTab;
  setContextSidebarTab: (tab: ContextSidebarTab) => void;

  kernelVisionTab: KernelVisionTab;
  setKernelVisionTab: (tab: KernelVisionTab) => void;

  // Settings page
  settingsActiveCategory: string;
  setSettingsActiveCategory: (category: string) => void;

  // OmniInput preferences
  enableMemory: boolean;
  setEnableMemory: (enabled: boolean) => void;
}

export const useChatUiStore = create<ChatUiStore>()(
  devtools(
    persist(
      (set) => ({
        activeNavTab: 'chat',
        setActiveNavTab: (tab) => set({ activeNavTab: tab }),

        theme: 'dark',
        toggleTheme: () => set((s) => ({ theme: s.theme === 'dark' ? 'light' : 'dark' })),

        isContextSidebarCollapsed: false,
        setContextSidebarCollapsed: (collapsed) => set({ isContextSidebarCollapsed: collapsed }),
        toggleContextSidebar: () => set((s) => ({ isContextSidebarCollapsed: !s.isContextSidebarCollapsed })),

        isKernelVisionCollapsed: false,
        setKernelVisionCollapsed: (collapsed) => set({ isKernelVisionCollapsed: collapsed }),
        toggleKernelVision: () => set((s) => ({ isKernelVisionCollapsed: !s.isKernelVisionCollapsed })),

        contextSidebarTab: 'topics',
        setContextSidebarTab: (tab) => set({ contextSidebarTab: tab }),

        kernelVisionTab: 'context',
        setKernelVisionTab: (tab) => set({ kernelVisionTab: tab }),

        settingsActiveCategory: 'general',
        setSettingsActiveCategory: (category) => set({ settingsActiveCategory: category }),

        enableMemory: true,
        setEnableMemory: (enabled) => set({ enableMemory: enabled }),
      }),
      {
        name: 'chat-ui-store',
        version: 2,
        migrate: (persisted, version) => {
          // v2: 'theme' 标签已改为主题切换开关，迁移历史状态中失效的标签值
          const state = persisted as { activeNavTab?: string };
          if (version < 2 && state.activeNavTab === 'theme') {
            state.activeNavTab = 'chat';
          }
          return persisted as ChatUiStore;
        },
      },
    ),
    { name: 'ChatUiStore' },
  ),
);
