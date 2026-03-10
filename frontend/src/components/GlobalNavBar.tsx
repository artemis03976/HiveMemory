import { MessageSquare, BookOpen, Bot, Moon, Terminal, Settings, User } from 'lucide-react';
import { cn } from '@/lib/utils';

interface NavItem {
  icon: React.ElementType;
  label: string;
  active?: boolean;
  onClick?: () => void;
  disabled?: boolean;
}

const topNavItems: NavItem[] = [
  { icon: MessageSquare, label: 'Chat', active: true },
  { icon: BookOpen, label: 'Memory Garden' },
  { icon: Bot, label: 'Agents', disabled: true },
];

const bottomNavItems: NavItem[] = [
  { icon: Moon, label: 'Theme' },
  { icon: Terminal, label: 'Kernel Console' },
  { icon: Settings, label: 'Settings' },
  // { icon: User, label: 'Profile' }, // 暂时注释，等账号系统开发后再启用
];

export function GlobalNavBar() {
  return (
    <nav className="glass-panel w-16 h-screen flex flex-col items-center py-4 border-r">
      {/* Top Section */}
      <div className="flex-1 flex flex-col gap-2">
        {topNavItems.map((item) => (
          <NavButton key={item.label} {...item} />
        ))}
      </div>

      {/* Bottom Section */}
      <div className="flex flex-col gap-2">
        {bottomNavItems.map((item) => (
          <NavButton key={item.label} {...item} />
        ))}
      </div>
    </nav>
  );
}

function NavButton({ icon: Icon, label, active, disabled, onClick }: NavItem) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={cn(
        'group relative w-12 h-12 rounded-lg flex items-center justify-center',
        'transition-all duration-200',
        // 选中状态：星云紫背景 + 边框 + 阴影
        active && 'bg-purple-600/20 text-purple-400 border border-purple-500/50 shadow-lg shadow-purple-500/20',
        // 未选中状态
        !active && !disabled && 'text-muted-foreground hover:text-foreground hover:bg-white/5 border border-transparent',
        // 禁用状态
        disabled && 'text-muted-foreground/30 cursor-not-allowed border border-transparent',
        !disabled && 'cursor-pointer'
      )}
      aria-label={label}
    >
      <Icon className="w-5 h-5" />

      {/* Tooltip */}
      <span className={cn(
        'absolute left-full ml-2 px-2 py-1 rounded-md',
        'bg-background/90 backdrop-blur-sm border border-white/10',
        'text-xs text-foreground whitespace-nowrap',
        'opacity-0 group-hover:opacity-100 transition-opacity duration-200',
        'pointer-events-none z-50'
      )}>
        {label}
      </span>
    </button>
  );
}
