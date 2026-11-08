/** chat-ui-store 持久化状态的版本迁移；不依赖 zustand，便于单独测试。 */
export const CHAT_UI_STORE_VERSION = 5;

interface PersistedChatUiState {
  contextSidebarTab?: string;
  activeNavTab?: string;
  isKernelVisionCollapsed?: boolean;
  kernelVisionTab?: string;
  isWorkspaceVisionCollapsed?: boolean;
  workspaceVisionTab?: string;
}

export function migrateChatUiState(persisted: unknown, version: number): unknown {
  const state = persisted as PersistedChatUiState;
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
  // v5: Workspace Vision 的终端标签移到全局运行页，选中它的历史状态回到引用记忆
  if (version < 5 && state.workspaceVisionTab === 'terminal') {
    state.workspaceVisionTab = 'context';
  }
  return persisted;
}
