import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { NavTab } from '@/types';

type ContextSidebarTab = 'sessions' | 'config';
type WorkspaceVisionTab = 'context' | 'memory-runtime' | 'terminal';
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

  isWorkspaceVisionCollapsed: boolean;
  setWorkspaceVisionCollapsed: (collapsed: boolean) => void;
  toggleWorkspaceVision: () => void;

  // Sidebar tabs
  contextSidebarTab: ContextSidebarTab;
  setContextSidebarTab: (tab: ContextSidebarTab) => void;

  workspaceVisionTab: WorkspaceVisionTab;
  setWorkspaceVisionTab: (tab: WorkspaceVisionTab) => void;

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

        isWorkspaceVisionCollapsed: false,
        setWorkspaceVisionCollapsed: (collapsed) => set({ isWorkspaceVisionCollapsed: collapsed }),
        toggleWorkspaceVision: () => set((s) => ({ isWorkspaceVisionCollapsed: !s.isWorkspaceVisionCollapsed })),

        contextSidebarTab: 'sessions',
        setContextSidebarTab: (tab) => set({ contextSidebarTab: tab }),

        workspaceVisionTab: 'context',
        setWorkspaceVisionTab: (tab) => set({ workspaceVisionTab: tab }),

        settingsActiveCategory: 'general',
        setSettingsActiveCategory: (category) => set({ settingsActiveCategory: category }),

        enableMemory: true,
        setEnableMemory: (enabled) => set({ enableMemory: enabled }),
      }),
      {
        name: 'chat-ui-store',
        version: 4,
        migrate: (persisted, version) => {
          const state = persisted as {
            contextSidebarTab?: string;
            activeNavTab?: string;
            isKernelVisionCollapsed?: boolean;
            kernelVisionTab?: WorkspaceVisionTab;
            isWorkspaceVisionCollapsed?: boolean;
            workspaceVisionTab?: WorkspaceVisionTab;
          };
          if (version < 4 && state.contextSidebarTab === 'topics') state.contextSidebarTab = 'sessions';
          // v2: 'theme' 标签已改为主题切换开关，迁移历史状态中失效的标签值
          if (version < 2 && state.activeNavTab === 'theme') {
            state.activeNavTab = 'chat';
          }
          // v3: 右侧观测面板由 Kernel Vision 改称 Workspace Vision，迁移折叠状态与选中的 tab
          if (version < 3) {
            if (state.isKernelVisionCollapsed !== undefined) {
              state.isWorkspaceVisionCollapsed = state.isKernelVisionCollapsed;
            }
            if (state.kernelVisionTab !== undefined) {
              state.workspaceVisionTab = state.kernelVisionTab;
            }
            delete state.isKernelVisionCollapsed;
            delete state.kernelVisionTab;
          }
          return persisted as ChatUiStore;
        },
      },
    ),
    { name: 'ChatUiStore' },
  ),
);
