import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { NavTab } from '@/types';
import type { RunFilters } from '@/stores/runs';
import { CHAT_UI_STORE_VERSION, migrateChatUiState } from '@/stores/ui/chatUiMigration';

type ContextSidebarTab = 'sessions' | 'config';
type WorkspaceVisionTab = 'context' | 'memory-runtime';
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

  // 运行页的列表筛选（视图偏好）
  runFilters: RunFilters;
  setRunFilters: (filters: Partial<RunFilters>) => void;
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

        runFilters: { kind: 'all', onlyActive: false, onlyIssues: false, query: '' },
        setRunFilters: (filters) => set((s) => ({ runFilters: { ...s.runFilters, ...filters } })),
      }),
      {
        name: 'chat-ui-store',
        version: CHAT_UI_STORE_VERSION,
        migrate: (persisted, version) => migrateChatUiState(persisted, version) as ChatUiStore,
      },
    ),
    { name: 'ChatUiStore' },
  ),
);
