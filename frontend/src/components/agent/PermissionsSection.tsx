import { Shield, Search, BookOpen, Play, Edit3, RefreshCw } from 'lucide-react';
import { Toggle } from '@/components/common/FormControls';
import type { MTPVerb, AgentProfileConfig } from '@/types';
import { ALL_MTP_VERBS } from '@/types';

interface PermissionsSectionProps {
  config: AgentProfileConfig;
  onChange: (config: AgentProfileConfig) => void;
}

const MTP_VERB_META: Record<MTPVerb, { label: string; description: string; icon: typeof Search }> = {
  SEARCH: { label: 'SEARCH', description: '模糊检索，返回记忆索引菜单', icon: Search },
  READ:   { label: 'READ',   description: '查阅记忆原子的完整 Payload', icon: BookOpen },
  RUN:    { label: 'RUN',    description: '调用内核工具或记忆中的代码', icon: Play },
  WRITE:  { label: 'WRITE',  description: '向帕秋莉发送高优先级保存信号', icon: Edit3 },
  UPDATE: { label: 'UPDATE', description: '请求更新已有记忆原子', icon: RefreshCw },
};

export function PermissionsSection({ config, onChange }: PermissionsSectionProps) {
  const isAllAllowed = config.allowed_mtp_verbs.length === 0;

  const isVerbAllowed = (verb: MTPVerb) =>
    isAllAllowed || config.allowed_mtp_verbs.includes(verb);

  const toggleVerb = (verb: MTPVerb, enabled: boolean) => {
    if (isAllAllowed) {
      // 从全部允许切换到白名单模式：移除该 verb
      onChange({ ...config, allowed_mtp_verbs: ALL_MTP_VERBS.filter(v => v !== verb) });
    } else if (enabled) {
      const next = [...config.allowed_mtp_verbs, verb];
      // 如果全部选中，回到空列表（全部允许）
      onChange({ ...config, allowed_mtp_verbs: next.length === ALL_MTP_VERBS.length ? [] : next });
    } else {
      onChange({ ...config, allowed_mtp_verbs: config.allowed_mtp_verbs.filter(v => v !== verb) });
    }
  };

  return (
    <section className="space-y-4">
      <div className="flex items-center justify-between">
        <label className="text-sm font-bold text-slate-200 flex items-center gap-2">
          <Shield className="w-4 h-4 text-primary" />
          MTP 指令权限
        </label>
        {isAllAllowed && (
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            全部允许
          </span>
        )}
      </div>
      <div className="grid grid-cols-2 gap-3">
        {ALL_MTP_VERBS.map(verb => {
          const meta = MTP_VERB_META[verb];
          const Icon = meta.icon;
          const allowed = isVerbAllowed(verb);
          return (
            <div
              key={verb}
              className={`border rounded-2xl p-4 flex items-center justify-between transition-colors ${
                allowed
                  ? 'bg-surface-container-high border-white/5 hover:bg-white/2'
                  : 'bg-black/20 border-white/5 opacity-60'
              }`}
            >
              <div className="flex items-center gap-3">
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${allowed ? 'bg-primary/15 text-primary' : 'bg-white/5 text-slate-500'}`}>
                  <Icon className="w-4 h-4" />
                </div>
                <div>
                  <div className="text-sm font-bold text-slate-200 font-mono">{meta.label}</div>
                  <div className="text-[11px] text-slate-500 mt-0.5">{meta.description}</div>
                </div>
              </div>
              <Toggle
                checked={allowed}
                onChange={c => toggleVerb(verb, c)}
              />
            </div>
          );
        })}
      </div>
    </section>
  );
}
