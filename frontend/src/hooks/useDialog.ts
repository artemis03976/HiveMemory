import { useEffect, useRef, type KeyboardEvent } from 'react';

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/**
 * 对话框的键盘行为：打开时焦点移到对话框面板上，Tab 只在对话框内循环，Esc 关闭，关闭后焦点回到打开前的位置。
 *
 * `panelRef` 与 `onKeyDown` 都放在面板上，面板带 `tabIndex={-1}`。叠在另一个对话框之上的对话框处理 Esc
 * 后停止冒泡，一次只关闭最上层。
 */
export function useDialog<T extends HTMLElement>(open: boolean, onClose: () => void) {
  const panelRef = useRef<T>(null);

  useEffect(() => {
    if (!open) return;
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    panelRef.current?.focus();
    return () => previous?.focus();
  }, [open]);

  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key === 'Escape') {
      event.stopPropagation();
      onClose();
      return;
    }
    const panel = panelRef.current;
    if (event.key !== 'Tab' || !panel) return;
    const items = [...panel.querySelectorAll<HTMLElement>(FOCUSABLE)];
    if (items.length === 0) {
      event.preventDefault();
      return;
    }
    const first = items[0];
    const last = items[items.length - 1];
    const active = document.activeElement;
    // 焦点在面板本身时，Tab 进入第一项、Shift+Tab 进入最后一项，不离开对话框
    if (event.shiftKey && (active === first || active === panel)) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && (active === last || active === panel)) {
      event.preventDefault();
      first.focus();
    }
  };

  return { panelRef, onKeyDown };
}
