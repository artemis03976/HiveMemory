import { useEffect, useState } from 'react';

/** 当前时间，按给定间隔刷新；随时间变化的显示（相对时间、耗时）以它为准，不在渲染中直接读时钟。 */
export function useNow(intervalMs: number): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(timer);
  }, [intervalMs]);
  return now;
}
