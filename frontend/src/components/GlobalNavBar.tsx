import { Sparkles, MessageSquare, Database, Bot, Moon, Sun, Terminal, Settings } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import type { NavTab } from '../types';
import { useChatUiStore } from '@/stores';

interface GlobalNavBarProps {
  activeTab: NavTab;
  onTabChange: (tab: NavTab) => void;
}

export default function GlobalNavBar({ activeTab, onTabChange }: GlobalNavBarProps) {
  const theme = useChatUiStore((s) => s.theme);
  const toggleTheme = useChatUiStore((s) => s.toggleTheme);

  const navItems = [
    { id: 'chat' as const, icon: MessageSquare, fill: true },
    { id: 'database' as const, icon: Database },
    { id: 'agents' as const, icon: Bot },
  ];

  const footerItems = [
    { id: 'terminal' as const, icon: Terminal },
    { id: 'settings' as const, icon: Settings },
  ];

  const renderButton = (item: { id: NavTab; icon: LucideIcon; fill?: boolean }) => {
    const isActive = activeTab === item.id;
    const Icon = item.icon;

    return (
      <button
        key={item.id}
        onClick={() => onTabChange(item.id)}
        className={`relative flex items-center justify-center w-full py-2 transition-all active:scale-95 ${
          isActive ? 'text-primary' : 'text-slate-500 hover:text-slate-300 hover:bg-white/5'
        }`}
      >
        {isActive && (
          <div className="absolute left-0 h-8 w-0.5 bg-primary shadow-[0_0_15px_#c59aff]" />
        )}
        <Icon className={`w-5 h-5 ${item.fill && isActive ? 'fill-current' : ''}`} />
      </button>
    );
  };

  return (
    <aside className="fixed left-0 top-0 h-full z-50 w-16 flex flex-col items-center py-4 border-r border-white/5 bg-surface-container-low backdrop-blur-xl shadow-[4px_0_24px_rgba(0,0,0,0.5)] light:shadow-[4px_0_24px_rgba(0,0,0,0.08)] font-manrope">
      <div className="flex flex-col items-center gap-8 w-full">
        {/* 系统/软件图标*/}
        <div className="w-10 h-10 flex items-center justify-center rounded-xl bg-surface-container-highest ghost-border">
          <Sparkles className="w-5 h-5 text-primary" />
        </div>

        {/* 顶部导航栏项目 */}
        <nav className="flex flex-col gap-6 w-full items-center">
          {navItems.map(renderButton)}
        </nav>
      </div>

      {/* 底部导航栏项目 */}
      <div className="mt-auto flex flex-col gap-6 w-full items-center">
        {/* 主题切换开关 */}
        <button
          onClick={toggleTheme}
          title={theme === 'dark' ? '切换为浅色主题' : '切换为深色主题'}
          className="relative flex items-center justify-center w-full py-2 transition-all active:scale-95 text-slate-500 hover:text-slate-300 hover:bg-white/5"
        >
          {theme === 'dark' ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
        </button>
        {footerItems.map(renderButton)}
        <div className="mt-2">
          <img
            alt="User Profile"
            className="w-8 h-8 rounded-full border border-white/10"
            src="https://lh3.googleusercontent.com/aida-public/AB6AXuAOfRVlFqvlT7MmIXghjW-xDwJaWL9uuWzMrMysEmti0APnV7dLcEmevwpY3R0KZVETrJ6pGMHgwk7ME-OubXIS5o_9J6hCuSspa-aWaww14z7tzG9lbrj-EKDwiFxJSKq23QsfrOq0DD_UjE3FjgJJCWz03i3mBJ4fSHAVr1ZMADL-thVKUjXQVlMCIwBdyVL0BgrvSyIoJvml9ZhWaLmfej6ZkWNhiWYEn1BgmdXRc41Ii1OYacO38h_cjDZmQVogPlWhrLiQPleV"
            referrerPolicy="no-referrer"
          />
        </div>
      </div>
    </aside>
  );
}
