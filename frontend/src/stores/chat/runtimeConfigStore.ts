import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

import type { ChatGenerationOptions } from '@/types/chat';

interface ChatRuntimeConfigStore {
  generationOptions: Required<ChatGenerationOptions>;
  updateGenerationOptions: (patch: Partial<ChatGenerationOptions>) => void;
}

// model 语义：注册表模型 ID 的会话级覆盖。空字符串 = 跟随 Agent Profile 默认。
// 具体下发时由 OmniInput 过滤空 model（后端 model 字段要求 min_length=1）。
const DEFAULT_GENERATION_OPTIONS: Required<ChatGenerationOptions> = {
  model: '',
  temperature: 0.7,
  top_p: 1,
  max_tokens: 4096,
};

export const useChatRuntimeConfigStore = create<ChatRuntimeConfigStore>()(
  devtools(
    persist(
      (set) => ({
        generationOptions: DEFAULT_GENERATION_OPTIONS,
        updateGenerationOptions: (patch) =>
          set((state) => ({
            generationOptions: {
              ...state.generationOptions,
              ...patch,
            },
          })),
      }),
      {
        name: 'chat-runtime-config-store',
        // v2: model 语义从 litellm 字符串改为注册表 ID。
        // 迁移旧持久化值——把遗留的 litellm 字符串（含 '/' 或旧硬编码值）清空，
        // 回落到"跟随 Agent 默认"，避免被当成不存在的注册表 ID 而报错。
        version: 2,
        migrate: (persisted: unknown, version: number) => {
          const state = persisted as { generationOptions?: Required<ChatGenerationOptions> } | undefined;
          if (state?.generationOptions && version < 2) {
            state.generationOptions.model = '';
          }
          return state as ChatRuntimeConfigStore;
        },
      },
    ),
    { name: 'ChatRuntimeConfigStore' },
  ),
);
