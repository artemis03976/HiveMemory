import { useEffect, useRef, useState } from 'react';
import { MessageSquare, Database, Bot, Moon, Sun, Terminal, Settings, Palette, Sparkle } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import type { NavTab } from '../types';
import { useChatUiStore } from '@/stores';
import { THEME_OPTIONS, type ResolvedTheme } from '@/stores/ui/theme';

interface GlobalNavBarProps {
  activeTab: NavTab;
  onTabChange: (tab: NavTab) => void;
  resolvedTheme: ResolvedTheme;
}

interface NavItem {
  id: NavTab;
  icon: LucideIcon;
  label: string;
}

const NAV_ITEMS: NavItem[] = [
  { id: 'chat', icon: MessageSquare, label: '对话' },
  { id: 'database', icon: Database, label: '记忆图书馆' },
  { id: 'agents', icon: Bot, label: 'Agent' },
  { id: 'terminal', icon: Terminal, label: '运行' },
];

const SETTINGS_ITEM: NavItem = { id: 'settings', icon: Settings, label: '系统设置' };

/** 悬停或键盘聚焦时出现在按钮右侧的提示。 */
function Tooltip({ text }: { text: string }) {
  return (
    <span className="pointer-events-none absolute left-14 top-1/2 z-50 -translate-x-1 -translate-y-1/2 whitespace-nowrap rounded-lg border border-line-strong bg-surface-container-high px-2.5 py-1 text-xs font-semibold text-fg opacity-0 shadow-pop transition-all group-hover:translate-x-0 group-hover:opacity-100 group-focus-visible:translate-x-0 group-focus-visible:opacity-100">
      {text}
    </span>
  );
}

function BrandMark() {
  return (
    <div aria-hidden="true" className="mb-3 flex h-10 w-10 items-center justify-center rounded-xl bg-brand shadow-glow">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ffffff" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 2.5l8.2 4.75v9.5L12 21.5l-8.2-4.75v-9.5z" />
        <path d="M12 7.5l1.3 3.2 3.2 1.3-3.2 1.3L12 16.5l-1.3-3.2L7.5 12l3.2-1.3z" />
      </svg>
    </div>
  );
}

/** 头像菜单：没有账户系统，只放主题选择与外观设置入口。 */
function AvatarMenu({ onOpenAppearance }: { onOpenAppearance: () => void }) {
  const theme = useChatUiStore((s) => s.theme);
  const setTheme = useChatUiStore((s) => s.setTheme);
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false);
    };
    document.addEventListener('pointerdown', onPointerDown);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('pointerdown', onPointerDown);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [open]);

  return (
    <div ref={rootRef} className="relative mt-1">
      <button
        onClick={() => setOpen((value) => !value)}
        aria-label="外观与主题"
        aria-haspopup="menu"
        aria-expanded={open}
        className="block h-[38px] w-[38px] rounded-full bg-brand p-[2px] transition-transform active:scale-95"
      >
        <span className="flex h-full w-full items-center justify-center rounded-full bg-surface-container-low text-accent">
          <Sparkle className="h-4 w-4" />
        </span>
      </button>
      {open && (
        <div role="menu" aria-label="外观" className="absolute bottom-0 left-14 z-50 flex w-60 flex-col gap-2.5 rounded-2xl border border-line-strong bg-surface-container p-3 shadow-pop">
          <span className="text-xs font-semibold text-fg-3">主题</span>
          <div role="group" aria-label="主题" className="flex gap-1 rounded-xl border border-line bg-field p-[3px]">
            {THEME_OPTIONS.map((option) => (
              <button
                key={option.mode}
                onClick={() => setTheme(option.mode)}
                aria-pressed={theme === option.mode}
                className={`h-[30px] flex-1 rounded-lg text-[13px] font-semibold transition-colors ${
                  theme === option.mode ? 'bg-primary/15 text-primary' : 'text-fg-2 hover:text-fg'
                }`}
              >
                {option.label}
              </button>
            ))}
          </div>
          <button
            role="menuitem"
            onClick={() => {
              setOpen(false);
              onOpenAppearance();
            }}
            className="flex h-[34px] items-center gap-2 rounded-lg px-2 text-left text-[13px] text-fg hover:bg-ink/5"
          >
            <Palette className="h-4 w-4 text-fg-3" />
            外观设置
          </button>
        </div>
      )}
    </div>
  );
}

export default function GlobalNavBar({ activeTab, onTabChange, resolvedTheme }: GlobalNavBarProps) {
  const setTheme = useChatUiStore((s) => s.setTheme);
  const setSettingsActiveCategory = useChatUiStore((s) => s.setSettingsActiveCategory);

  const renderButton = (item: NavItem) => {
    const isActive = activeTab === item.id;
    const Icon = item.icon;
    return (
      <button
        key={item.id}
        onClick={() => onTabChange(item.id)}
        aria-label={item.label}
        aria-current={isActive ? 'page' : undefined}
        className={`group relative flex h-11 w-11 items-center justify-center rounded-xl transition-colors active:scale-95 ${
          isActive ? 'bg-primary/15 text-primary' : 'text-fg-3 hover:bg-surface-container-high hover:text-fg'
        }`}
      >
        {isActive && <span aria-hidden="true" className="absolute -left-2.5 top-2.5 bottom-2.5 w-[3px] rounded-full bg-brand" />}
        <Icon className="h-5 w-5" />
        <Tooltip text={item.label} />
      </button>
    );
  };

  const themeLabel = resolvedTheme === 'dark' ? '切换到浅色' : '切换到深色';

  return (
    <aside className="fixed left-0 top-0 z-50 flex h-full w-16 flex-col items-center justify-between border-r border-line bg-surface-container-low py-3.5">
      <nav aria-label="主导航" className="flex w-full flex-col items-center gap-1.5">
        <BrandMark />
        {NAV_ITEMS.map(renderButton)}
      </nav>

      <div className="flex w-full flex-col items-center gap-2">
        {renderButton(SETTINGS_ITEM)}
        <button
          onClick={() => setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')}
          aria-label={themeLabel}
          className="group relative flex h-11 w-11 items-center justify-center rounded-xl text-fg-3 transition-colors hover:bg-surface-container-high hover:text-fg active:scale-95"
        >
          {resolvedTheme === 'dark' ? <Moon className="h-5 w-5" /> : <Sun className="h-5 w-5" />}
          <Tooltip text={themeLabel} />
        </button>
        <AvatarMenu
          onOpenAppearance={() => {
            setSettingsActiveCategory('general');
            onTabChange('settings');
          }}
        />
      </div>
    </aside>
  );
}
