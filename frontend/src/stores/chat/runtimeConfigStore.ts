import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

import type { ChatGenerationOptions } from '@/types/chat';

interface ChatRuntimeConfigStore {
  generationOptions: Required<ChatGenerationOptions>;
  updateGenerationOptions: (patch: Partial<ChatGenerationOptions>) => void;
}

const DEFAULT_GENERATION_OPTIONS: Required<ChatGenerationOptions> = {
  model: 'deepseek/deepseek-chat',
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
      },
    ),
    { name: 'ChatRuntimeConfigStore' },
  ),
);
