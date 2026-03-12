import { useState } from 'react';
import { MessageSquare, BookOpen, Bot, Moon, Terminal, Settings } from 'lucide-react';
import { cn } from '@/lib/utils';

// 每个导航项的独立色彩身份
type AccentColor = 'moon' | 'sun' | 'water' | 'wood' | 'fire' | 'earth' | 'metal' | 'neutral';

const accentStyles: Record<AccentColor, { text: string; shadow: string; indicator: string }> = {
  moon: {
    text: 'text-[hsl(260,80%,65%)]',
    shadow: 'drop-shadow-[0_0_8px_hsla(260,80%,65%,0.6)]',
    indicator: 'bg-[hsl(260,80%,65%)] shadow-[0_0_10px_hsla(260,80%,65%,0.8)]',
  },
  sun: {
    text: 'text-[hsl(45,100%,60%)]',
    shadow: 'drop-shadow-[0_0_8px_hsla(45,100%,60%,0.6)]',
    indicator: 'bg-[hsl(45,100%,60%)] shadow-[0_0_10px_hsla(45,100%,60%,0.8)]',
  },
  water: {
    text: 'text-[hsl(190,90%,50%)]',
    shadow: 'drop-shadow-[0_0_8px_hsla(190,90%,50%,0.6)]',
    indicator: 'bg-[hsl(190,90%,50%)] shadow-[0_0_10px_hsla(190,90%,50%,0.8)]',
  },
  wood: {
    text: 'text-[hsl(150,80%,45%)]',
    shadow: 'drop-shadow-[0_0_8px_hsla(150,80%,45%,0.6)]',
    indicator: 'bg-[hsl(150,80%,45%)] shadow-[0_0_10px_hsla(150,80%,45%,0.8)]',
  },
  fire: {
    text: 'text-[hsl(340,80%,60%)]',
    shadow: 'drop-shadow-[0_0_8px_hsla(340,80%,60%,0.6)]',
    indicator: 'bg-[hsl(340,80%,60%)] shadow-[0_0_10px_hsla(340,80%,60%,0.8)]',
  },
  earth: {
    text: 'text-[hsl(30,20%,50%)]',
    shadow: 'drop-shadow-[0_0_8px_hsla(30,20%,50%,0.6)]',
    indicator: 'bg-[hsl(30,20%,50%)] shadow-[0_0_10px_hsla(30,20%,50%,0.8)]',
  },
  metal: {
    text: 'text-[hsl(40,90%,55%)]',
    shadow: 'drop-shadow-[0_0_8px_hsla(40,90%,55%,0.6)]',
    indicator: 'bg-[hsl(40,90%,55%)] shadow-[0_0_10px_hsla(40,90%,55%,0.8)]',
  },
  neutral: {
    text: 'text-foreground/80',
    shadow: 'drop-shadow-[0_0_6px_rgba(255,255,255,0.15)]',
    indicator: 'bg-foreground/60 shadow-[0_0_8px_rgba(255,255,255,0.2)]',
  },
};

interface NavItemConfig {
  id: string;
  icon: React.ElementType;
  label: string;
  accent: AccentColor;
  disabled?: boolean;
  /** 底部按钮可以是 toggle 行为，不参与主导航切换 */
  toggle?: boolean;
}

const topNavItems: NavItemConfig[] = [
  { id: 'chat', icon: MessageSquare, label: 'Chat', accent: 'moon' },
  { id: 'memory', icon: BookOpen, label: 'Memory Garden', accent: 'sun' },
  { id: 'agents', icon: Bot, label: 'Agents', accent: 'water', disabled: true },
];

const bottomNavItems: NavItemConfig[] = [
  { id: 'theme', icon: Moon, label: 'Theme', accent: 'neutral', toggle: true },
  { id: 'kernel', icon: Terminal, label: 'Kernel Console', accent: 'water' },
  { id: 'settings', icon: Settings, label: 'Settings', accent: 'neutral' },
];

export function GlobalNavBar() {
  const [activeNav, setActiveNav] = useState('chat');
  const [activeToggles, setActiveToggles] = useState<Set<string>>(new Set());

  const handleClick = (item: NavItemConfig) => {
    if (item.disabled) return;
    if (item.toggle) {
      setActiveToggles(prev => {
        const next = new Set(prev);
        if (next.has(item.id)) next.delete(item.id);
        else next.add(item.id);
        return next;
      });
    } else {
      setActiveNav(item.id);
    }
  };

  const isActive = (item: NavItemConfig) =>
    item.toggle ? activeToggles.has(item.id) : activeNav === item.id;

  return (
    <nav className="glass-nav w-16 h-screen flex flex-col items-center py-6 relative z-50">
      {/* Top Section */}
      <div className="flex-1 flex flex-col gap-4">
        {topNavItems.map((item) => (
          <NavButton
            key={item.id}
            {...item}
            active={isActive(item)}
            onClick={() => handleClick(item)}
          />
        ))}
      </div>

      {/* Bottom Section */}
      <div className="flex flex-col gap-4">
        {bottomNavItems.map((item) => (
          <NavButton
            key={item.id}
            {...item}
            active={isActive(item)}
            onClick={() => handleClick(item)}
          />
        ))}
      </div>
    </nav>
  );
}

interface NavButtonProps extends NavItemConfig {
  active: boolean;
  onClick: () => void;
}

function NavButton({ icon: Icon, label, accent, active, disabled, onClick }: NavButtonProps) {
  const colors = accentStyles[accent];

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={cn(
        'group relative w-12 h-12 flex items-center justify-center rounded-xl',
        'transition-all duration-300 ease-out',
        !active && !disabled && 'text-muted-foreground hover:text-foreground hover:bg-white/5',
        disabled && 'text-muted-foreground/30 cursor-not-allowed',
        !disabled && 'cursor-pointer',
        active && colors.text,
        active && colors.shadow,
      )}
      aria-label={label}
    >
      {/* 左侧霓虹指示条 — 颜色跟随 accent */}
      {active && (
        <div className={cn(
          'absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-10 rounded-r-full',
          colors.indicator,
        )} />
      )}

      <Icon className={cn(
        "w-[22px] h-[22px] transition-transform duration-300",
        active ? "scale-110" : "group-hover:scale-110"
      )} />

      {/* Tooltip */}
      <span className={cn(
        'absolute left-full ml-3 px-3 py-1.5 rounded-md',
        'bg-background/80 backdrop-blur-md border border-white/10 shadow-xl',
        'text-xs font-medium text-foreground whitespace-nowrap',
        'opacity-0 -translate-x-2 group-hover:opacity-100 group-hover:translate-x-0',
        'transition-all duration-200 ease-out pointer-events-none z-50'
      )}>
        {label}
      </span>
    </button>
  );
}