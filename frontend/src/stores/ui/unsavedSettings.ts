import { create } from 'zustand';

/** 设置页是否有未保存的更改：离开设置页前据此提醒，不持久化。 */
export const useUnsavedSettings = create<{ dirty: boolean; setDirty: (dirty: boolean) => void }>((set) => ({
  dirty: false,
  setDirty: (dirty) => set({ dirty }),
}));
