/** 主题选项：深色、浅色或跟随系统；不依赖 zustand，便于单独测试。 */
export type ThemeMode = 'dark' | 'light' | 'system';
export type ResolvedTheme = 'dark' | 'light';

export const THEME_OPTIONS: { mode: ThemeMode; label: string }[] = [
  { mode: 'dark', label: '深色' },
  { mode: 'light', label: '浅色' },
  { mode: 'system', label: '跟随系统' },
];

/** 跟随系统时按系统的深浅色取值，否则就是选定的主题。 */
export function resolveTheme(mode: ThemeMode, systemPrefersDark: boolean): ResolvedTheme {
  if (mode === 'system') return systemPrefersDark ? 'dark' : 'light';
  return mode;
}
