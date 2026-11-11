import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { NavTab } from '@/types';
import type { RunFilters } from '@/stores/runs';
import { CHAT_UI_STORE_VERSION, migrateChatUiState } from '@/stores/ui/chatUiMigration';
import type { ThemeMode } from '@/stores/ui/theme';

export type WorkspaceVisionTab = 'context' | 'memory-runtime' | 'model';

interface ChatUiStore {
  // Global navigation
  activeNavTab: NavTab;
  setActiveNavTab: (tab: NavTab) => void;

  // 主题选项；已保存的深色或浅色保持不变，新安装默认跟随系统
  theme: ThemeMode;
  setTheme: (theme: ThemeMode) => void;

  // 聊天页两栏的收起状态：只在窗口放得下该栏时生效，放不下时两栏改为抽屉
  isContextSidebarCollapsed: boolean;
  toggleContextSidebar: () => void;
  isWorkspaceVisionCollapsed: boolean;
  toggleWorkspaceVision: () => void;

  // 聊天页右栏的标签
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

        theme: 'system',
        setTheme: (theme) => set({ theme }),

        isContextSidebarCollapsed: false,
        toggleContextSidebar: () => set((s) => ({ isContextSidebarCollapsed: !s.isContextSidebarCollapsed })),
        isWorkspaceVisionCollapsed: false,
        toggleWorkspaceVision: () => set((s) => ({ isWorkspaceVisionCollapsed: !s.isWorkspaceVisionCollapsed })),

        workspaceVisionTab: 'context',
        setWorkspaceVisionTab: (tab) => set({ workspaceVisionTab: tab }),

        settingsActiveCategory: 'common',
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
