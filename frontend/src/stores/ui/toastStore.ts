import { create } from 'zustand';

export type ToastType = 'success' | 'error' | 'info' | 'warning';

export interface Toast {
  id: string;
  message: string;
  type: ToastType;
  duration?: number;
}

/** 未指定时长时按类型取值：成功一闪而过，错误停留更久并可以手动关闭。 */
const DEFAULT_DURATION: Record<ToastType, number> = {
  success: 2000,
  info: 3000,
  warning: 4000,
  error: 6000,
};

/** 同一条提示（文字与类型都相同）在显示期间只出现一次，返回 null 表示不再追加。 */
export function pushToast(toasts: Toast[], toast: Toast): Toast[] | null {
  const duplicate = toasts.some((t) => t.message === toast.message && t.type === toast.type);
  return duplicate ? null : [...toasts, toast];
}

interface ToastState {
  toasts: Toast[];
  addToast: (message: string, type?: ToastType, duration?: number) => void;
  removeToast: (id: string) => void;
}

export const useToastStore = create<ToastState>((set, get) => ({
  toasts: [],
  addToast: (message, type = 'info', duration = DEFAULT_DURATION[type]) => {
    const id = Math.random().toString(36).substring(2, 9);
    const next = pushToast(get().toasts, { id, message, type, duration });
    if (!next) return;
    set({ toasts: next });

    if (duration > 0) {
      setTimeout(() => {
        set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) }));
      }, duration);
    }
  },
  removeToast: (id) => {
    set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) }));
  },
}));
