import { useCallback, useSyncExternalStore } from 'react';

/** 订阅一条媒体查询，返回它当前是否成立；窗口尺寸或系统深浅色变化时重新渲染。 */
export function useMediaQuery(query: string): boolean {
  const subscribe = useCallback(
    (onChange: () => void) => {
      const list = window.matchMedia(query);
      list.addEventListener('change', onChange);
      return () => list.removeEventListener('change', onChange);
    },
    [query],
  );
  return useSyncExternalStore(subscribe, () => window.matchMedia(query).matches);
}
