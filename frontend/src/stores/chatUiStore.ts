import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { NavTab } from '@/types';

type ContextSidebarTab = 'topics' | 'config';
type KernelVisionTab = 'context' | 'terminal';

interface ChatUiStore {
  // Global navigation
  activeNavTab: NavTab;
  setActiveNavTab: (tab: NavTab) => void;

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
        version: 1,
      },
    ),
    { name: 'ChatUiStore' },
  ),
);
