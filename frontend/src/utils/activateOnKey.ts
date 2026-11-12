import type { KeyboardEvent } from 'react';

/**
 * 带 `role="button"` 的卡片与行也能用 Enter 与空格触发，与原生按钮一致。
 * 只响应元素自身获得焦点时的按键，卡片内部的按钮与输入框各自处理自己的按键。
 */
export function activateOnKey(action: () => void) {
  return (event: KeyboardEvent<HTMLElement>) => {
    if (event.target !== event.currentTarget) return;
    if (event.key !== 'Enter' && event.key !== ' ') return;
    event.preventDefault();
    action();
  };
}
