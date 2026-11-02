import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

import type { ChatModelSelection } from '@/types/chat';

interface ChatRuntimeConfigStore {
  modelSelection: Required<ChatModelSelection>;
  /** 是否覆盖生成参数（temperature/top_p/max_tokens）。
   *  false = 跟随 Agent Profile / 模型定义默认，不下发这些参数。 */
  overrideParams: boolean;
  updateModelSelection: (patch: Partial<ChatModelSelection>) => void;
  setOverrideParams: (value: boolean) => void;
}

// model_id 语义：注册表模型 ID 的会话级覆盖。空字符串 = 跟随 Agent Profile 默认。
// 具体下发时由 OmniInput 过滤空 model_id（后端 model_id 要求 min_length=1）。
// temperature/top_p/max_tokens 仅在 overrideParams=true 时下发。
const DEFAULT_MODEL_SELECTION: Required<ChatModelSelection> = {
  model_id: '',
  temperature: 1.0,
  top_p: 1,
  max_tokens: 32768,
};

/** 版本 2 及以前持久化的会话覆盖：字段名为 generationOptions，模型键为 model。 */
interface PersistedV2 {
  generationOptions?: { model: string; temperature: number; top_p: number; max_tokens: number };
}

export const useChatRuntimeConfigStore = create<ChatRuntimeConfigStore>()(
  devtools(
    persist(
      (set) => ({
        modelSelection: DEFAULT_MODEL_SELECTION,
        overrideParams: false,
        updateModelSelection: (patch) =>
          set((state) => ({
            modelSelection: {
              ...state.modelSelection,
              ...patch,
            },
          })),
        setOverrideParams: (value) => set({ overrideParams: value }),
      }),
      {
        name: 'chat-runtime-config-store',
        // v2: model 语义从 litellm 字符串改为注册表 ID，遗留的 litellm 字符串清空，
        // 回落到"跟随 Agent 默认"，避免被当成不存在的注册表 ID 而报错。
        // v3: 字段改为与后端 ModelSelection 一致的 modelSelection / model_id，原值保留。
        version: 3,
        migrate: (persisted: unknown, version: number) => {
          const state = persisted as (PersistedV2 & Partial<ChatRuntimeConfigStore>) | undefined;
          const legacy = state?.generationOptions;
          if (state && legacy) {
            const { model, ...params } = legacy;
            state.modelSelection = { ...params, model_id: version < 2 ? '' : model };
            delete state.generationOptions;
          }
          return state as ChatRuntimeConfigStore;
        },
      },
    ),
    { name: 'ChatRuntimeConfigStore' },
  ),
);
