import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

import type { ChatGenerationOptions } from '@/types/chat';

interface ChatRuntimeConfigStore {
  generationOptions: Required<ChatGenerationOptions>;
  /** 是否覆盖生成参数（temperature/top_p/max_tokens）。
   *  false = 跟随 Agent Profile / 模型定义默认，不下发这些参数。 */
  overrideParams: boolean;
  updateGenerationOptions: (patch: Partial<ChatGenerationOptions>) => void;
  setOverrideParams: (value: boolean) => void;
}

// model 语义：注册表模型 ID 的会话级覆盖。空字符串 = 跟随 Agent Profile 默认。
// 具体下发时由 OmniInput 过滤空 model（后端 model 字段要求 min_length=1）。
// temperature/top_p/max_tokens 仅在 overrideParams=true 时下发。
const DEFAULT_GENERATION_OPTIONS: Required<ChatGenerationOptions> = {
  model: '',
  temperature: 1.0,
  top_p: 1,
  max_tokens: 32768,
};

export const useChatRuntimeConfigStore = create<ChatRuntimeConfigStore>()(
  devtools(
    persist(
      (set) => ({
        generationOptions: DEFAULT_GENERATION_OPTIONS,
        overrideParams: false,
        updateGenerationOptions: (patch) =>
          set((state) => ({
            generationOptions: {
              ...state.generationOptions,
              ...patch,
            },
          })),
        setOverrideParams: (value) => set({ overrideParams: value }),
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
