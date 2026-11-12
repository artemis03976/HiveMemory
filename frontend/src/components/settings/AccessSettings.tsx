import { useCallback, useEffect, useState } from 'react';
import { ArrowRight, KeyRound, Lock, Pencil, RefreshCw, Trash2 } from 'lucide-react';

import { Input, Toggle } from '../common/FormControls';
import { useChatUiStore, useToastStore } from '@/stores';
import { useAgents } from '@/hooks/useAgents';
import {
  deleteActorRecord,
  fetchActorRecords,
  fetchPrincipals,
  putActorRecord,
  updatePrincipal,
} from '@/services/accessApi';
import type { AccessPrincipal, ActorRecord, ExternalAdapter, WorkspaceOperationName } from '@/types/access';

const OPERATIONS: { value: WorkspaceOperationName; label: string }[] = [
  { value: 'memory.read', label: '读取' },
  { value: 'memory.search', label: '检索' },
  { value: 'memory.run', label: '运行' },
  { value: 'memory.write', label: '写入' },
  { value: 'memory.update', label: '更新' },
  { value: 'agent.call', label: '调用 Agent' },
];

const ADAPTERS: { value: ExternalAdapter; label: string }[] = [
  { value: 'mcp', label: 'MCP 记忆工具' },
  { value: 'import', label: '对话导入' },
];

const errorText = (err: unknown) => (err instanceof Error ? err.message : '未知错误');

/** 逗号或空白分隔的 Agent 列表；为空表示不限定。 */
function parseAgents(text: string): string[] | null {
  const agents = text.split(/[\s,，]+/).map((a) => a.trim()).filter(Boolean);
  return agents.length > 0 ? agents : null;
}

function toggleIn<T>(list: T[], value: T, on: boolean): T[] {
  return on ? [...list.filter((v) => v !== value), value] : list.filter((v) => v !== value);
}

const chipCls = 'text-xs font-bold px-1.5 py-0.5 rounded tracking-wide';
const buttonCls =
  'px-4 py-1.5 text-sm font-bold bg-primary/20 hover:bg-primary/30 text-primary rounded-lg border border-primary/30 transition-all disabled:opacity-50';
const iconButtonCls =
  'p-2 text-fg-3 hover:text-primary hover:bg-ink/5 rounded-lg transition-all disabled:opacity-30 disabled:cursor-not-allowed';

interface PrincipalDraft {
  principal_id: string;
  adapters: ExternalAdapter[];
  agents: string;
}

/** 修改一个外部来源：接入方式与可声明的 Agent。一个 harness 可以以多个 Agent 运行。 */
function PrincipalEditor({
  draft,
  saving,
  onChange,
  onSubmit,
  onCancel,
}: {
  draft: PrincipalDraft;
  saving: boolean;
  onChange: (patch: Partial<PrincipalDraft>) => void;
  onSubmit: () => void;
  onCancel: () => void;
}) {
  const rowCls = 'flex items-center justify-between gap-4 py-2';
  const labelCls = 'text-sm text-fg-2 shrink-0 w-32';
  return (
    <div className="bg-field border border-primary/20 rounded-2xl p-5 space-y-1 border border-line">
      <div className="text-sm font-bold text-primary mb-3">修改 {draft.principal_id}</div>
      <div className={rowCls}>
        <span className={labelCls}>接入方式</span>
        <div className="flex gap-4">
          {ADAPTERS.map((a) => (
            <label key={a.value} className="flex items-center gap-2 text-sm text-fg-2">
              <input
                type="checkbox"
                checked={draft.adapters.includes(a.value)}
                onChange={(e) => onChange({ adapters: toggleIn(draft.adapters, a.value, e.target.checked) })}
                className="accent-primary"
              />
              {a.label}
            </label>
          ))}
        </div>
      </div>
      <div className={rowCls}>
        <span className={labelCls}>可声明的 Agent</span>
        <Input
          value={draft.agents}
          onChange={(v: string) => onChange({ agents: v })}
          placeholder="例如 claude_code, reviewer；留空表示不限定"
          className="w-64"
        />
      </div>
      <div className="flex justify-end gap-2 pt-3">
        <button onClick={onCancel} disabled={saving} className="px-4 py-1.5 text-sm text-fg-3 hover:text-fg transition-colors disabled:opacity-50">
          取消
        </button>
        <button onClick={onSubmit} disabled={saving} className={buttonCls}>
          {saving ? '保存中...' : '保存'}
        </button>
      </div>
    </div>
  );
}

function CredentialSummary({ principal }: { principal: AccessPrincipal }) {
  if (principal.credential_source === 'env') {
    return (
      <span className="flex items-center gap-1 text-magic-metal">
        <Lock className="w-3 h-3" />
        环境变量 <span className="font-mono">{principal.credential_env}</span>
      </span>
    );
  }
  if (principal.credential_source === 'generated') {
    return <span className="font-mono">生成凭据 {principal.credential_masked}</span>;
  }
  return <span className="text-magic-fire">没有可用凭据，认证会被拒绝</span>;
}

function PrincipalRow({
  principal,
  busy,
  onToggle,
  onEdit,
}: {
  principal: AccessPrincipal;
  busy: boolean;
  onToggle: (enabled: boolean) => void;
  onEdit: () => void;
}) {
  const external = principal.editable;
  return (
    <div className="flex items-center justify-between gap-4 p-4 border-b border-line last:border-0 hover:bg-ink/2 transition-colors">
      <div className="flex items-start gap-3 min-w-0">
        <KeyRound className="w-4 h-4 mt-0.5 text-fg-3 shrink-0" />
        <div className="min-w-0 space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium text-fg font-mono">{principal.principal_id}</span>
            {!external && <span className={`${chipCls} text-fg-3 bg-ink/5`}>进程内 · 只读</span>}
            {!principal.enabled && <span className={`${chipCls} text-magic-fire bg-magic-fire/10`}>已禁用</span>}
          </div>
          <div className="text-xs text-fg-3">
            {principal.adapters.join(' · ')}
            {external && (
              <>
                {' · '}
                {principal.allowed_agent_ids
                  ? `可声明 ${principal.allowed_agent_ids.join('、')}`
                  : '可声明任意 Agent'}
              </>
            )}
          </div>
          {external && (
            <div className="text-xs text-fg-3">
              <CredentialSummary principal={principal} />
            </div>
          )}
        </div>
      </div>
      {external && (
        <div className="flex items-center gap-1 shrink-0">
          <Toggle checked={principal.enabled} onChange={onToggle} disabled={busy} label="允许接入" />
          <button onClick={onEdit} disabled={busy} className={iconButtonCls} title="修改接入方式与可声明的 Agent" aria-label="修改">
            <Pencil className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
}

/** 一条访问记录的编辑行：准入开关与六项操作，每次改动立即保存。 */
function ActorRecordRow({
  record,
  busy,
  onSave,
  onDelete,
}: {
  record: ActorRecord;
  busy: boolean;
  onSave: (enabled: boolean, operations: WorkspaceOperationName[]) => void;
  onDelete: () => void;
}) {
  const userLevel = record.agent_id === null;
  return (
    <div className="flex items-center justify-between gap-4 p-4 border-b border-line last:border-0 hover:bg-ink/2 transition-colors">
      <div className="min-w-0 w-48 shrink-0">
        <div className="text-sm font-medium text-fg font-mono truncate">
          {userLevel ? '其余所有 Agent' : record.agent_id}
        </div>
        <div className="text-xs text-fg-3">{userLevel ? '用户级记录' : '精确记录，优先于用户级记录'}</div>
      </div>
      <div className="flex flex-1 flex-wrap items-center gap-3">
        {OPERATIONS.map((op) => (
          <label key={op.value} className={`flex items-center gap-1.5 text-xs ${record.enabled ? 'text-fg-2' : 'text-fg-3'}`}>
            <input
              type="checkbox"
              checked={record.allowed_operations.includes(op.value)}
              disabled={busy || !record.enabled}
              onChange={(e) => onSave(true, toggleIn(record.allowed_operations, op.value, e.target.checked))}
              className="accent-primary"
            />
            {op.label}
          </label>
        ))}
      </div>
      <div className="flex items-center gap-1 shrink-0">
        <span title={record.enabled ? '允许进入；关闭后清空操作' : '不允许进入'}>
          <Toggle
            checked={record.enabled}
            disabled={busy}
            label="允许进入"
            // 禁用的记录不能带白名单：关闭准入时一并清空操作。
            onChange={(enabled) => onSave(enabled, enabled ? record.allowed_operations : [])}
          />
        </span>
        <button onClick={onDelete} disabled={busy} className={`${iconButtonCls} hover:text-magic-fire`} title="删除记录" aria-label="删除记录">
          <Trash2 className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}

/**
 * 接入与权限：接入登记与各 Agent 的操作权限，修改立即生效并写回登记文件。
 *
 * 外部 harness 的来源在插件页安装插件时登记、卸载时删除，凭据也由插件页交给 harness；
 * 这里只能启停来源、调整它能声明哪些 Agent。
 */
export function AccessSettings() {
  const { addToast } = useToastStore();
  const setSettingsActiveCategory = useChatUiStore((s) => s.setSettingsActiveCategory);
  const { rawAgents } = useAgents();
  const [principals, setPrincipals] = useState<AccessPrincipal[]>([]);
  const [records, setRecords] = useState<ActorRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [draft, setDraft] = useState<PrincipalDraft | null>(null);
  const [editing, setEditing] = useState<AccessPrincipal | null>(null);
  const [newAgent, setNewAgent] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [loadedPrincipals, loadedRecords] = await Promise.all([fetchPrincipals(), fetchActorRecords()]);
      setPrincipals(loadedPrincipals);
      setRecords(loadedRecords);
    } catch (err) {
      addToast('读取接入与权限失败: ' + errorText(err), 'error');
    } finally {
      setLoading(false);
    }
  }, [addToast]);

  useEffect(() => { void load(); }, [load]);

  /** 执行一次修改并重新读取；失败时提示原因，页面保持原样。 */
  const run = async (action: () => Promise<void>, success: string) => {
    setBusy(true);
    try {
      await action();
      addToast(success, 'success', 1800);
      await load();
    } catch (err) {
      addToast(errorText(err), 'error');
    } finally {
      setBusy(false);
    }
  };

  const closeEditor = () => {
    setDraft(null);
    setEditing(null);
  };

  const submitPrincipal = async () => {
    if (!draft || !editing) return;
    if (draft.adapters.length === 0) {
      addToast('至少选择一种接入方式', 'error');
      return;
    }
    await run(async () => {
      await updatePrincipal(editing.principal_id, {
        adapters: draft.adapters,
        allowed_agent_ids: parseAgents(draft.agents),
        enabled: editing.enabled,
      });
      closeEditor();
    }, '接入来源已修改');
  };

  const addRecord = async () => {
    const agentId = newAgent.trim();
    if (!agentId) return;
    await run(async () => {
      await putActorRecord(agentId, { enabled: true, allowed_operations: ['memory.read', 'memory.search'] });
      setNewAgent('');
    }, 'Agent 记录已添加');
  };

  const removeRecord = async (record: ActorRecord) => {
    const message = record.agent_id === null
      ? '删除用户级记录后，没有单独记录的 Agent（包括对话中的 Agent）都不能再进入 Workspace。确定删除？'
      : `删除 ${record.agent_id} 的记录后，它改按用户级记录授权。确定删除？`;
    if (!window.confirm(message)) return;
    await run(() => deleteActorRecord(record.agent_id), '访问记录已删除');
  };

  const profileAgents = rawAgents
    .map((a) => a.alias)
    .filter((alias) => !records.some((r) => r.agent_id === alias));

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12 text-primary">
        <RefreshCw className="w-6 h-6 animate-spin" />
      </div>
    );
  }

  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
      <section className="mb-8">
        <div className="flex items-start justify-between gap-4 mb-4">
          <div>
            <h2 className="text-lg font-bold text-fg">外部接入</h2>
            <p className="mt-1 text-[13px] text-fg-3">
              外部 harness 的接入在插件页安装时登记、卸载时删除。这里可以停用，或调整它能以哪些 Agent 运行：一个 harness 可以声明多个 Agent。
            </p>
          </div>
          <button
            onClick={() => setSettingsActiveCategory('plugins')}
            className="flex shrink-0 items-center gap-1.5 px-3 py-1.5 bg-primary/15 hover:bg-primary/25 text-primary rounded-xl border border-primary/30 transition-all text-sm font-semibold"
          >
            前往插件页
            <ArrowRight className="w-3.5 h-3.5" aria-hidden="true" />
          </button>
        </div>
        {draft && (
          <div className="mb-4">
            <PrincipalEditor
              draft={draft}
              saving={busy}
              onChange={(patch) => setDraft((d) => (d ? { ...d, ...patch } : d))}
              onSubmit={() => void submitPrincipal()}
              onCancel={closeEditor}
            />
          </div>
        )}
        <div className="bg-surface-container-low border border-line rounded-2xl overflow-hidden border border-line">
          {principals.map((p) => (
            <PrincipalRow
              key={p.principal_id}
              principal={p}
              busy={busy}
              onToggle={(enabled) => void run(
                () => updatePrincipal(p.principal_id, {
                  adapters: p.adapters as ExternalAdapter[],
                  allowed_agent_ids: p.allowed_agent_ids,
                  enabled,
                }).then(() => undefined),
                enabled ? '已启用' : '已禁用，它之后的请求会被拒绝',
              )}
              onEdit={() => {
                setEditing(p);
                setDraft({
                  principal_id: p.principal_id,
                  adapters: p.adapters as ExternalAdapter[],
                  agents: (p.allowed_agent_ids ?? []).join(', '),
                });
              }}
            />
          ))}
        </div>
        <p className="text-xs text-fg-3 mt-3 leading-relaxed">
          外部 harness 以登记的来源标识与凭据接入，凭据由插件页安装时生成并交给 harness；更早手工配置的来源也可以继续把凭据写在 .env 中（环境变量优先）。
          进程内来源关系到本页面、对话与导入本身，只能修改登记文件后重启。修改立即生效并写回登记文件。
        </p>
      </section>

      <section className="mb-8">
        <h2 className="text-lg font-bold text-fg">Agent 操作权限</h2>
        <p className="mt-1 mb-4 text-[13px] text-fg-3">
          权限跟随 Agent，不跟随 harness：安装插件时只为还没有记录的缺省 Agent 补一条，卸载插件不删除它。
        </p>
        <div className="bg-surface-container-low border border-line rounded-2xl overflow-hidden border border-line">
          {records.map((r) => (
            <ActorRecordRow
              key={r.agent_id ?? '__user__'}
              record={r}
              busy={busy}
              onSave={(enabled, operations) => void run(
                () => putActorRecord(r.agent_id, { enabled, allowed_operations: operations }).then(() => undefined),
                '访问记录已保存',
              )}
              onDelete={() => void removeRecord(r)}
            />
          ))}
          <div className="flex items-center gap-3 p-4">
            <Input
              value={newAgent}
              onChange={(v: string) => setNewAgent(v)}
              placeholder="Agent 标识，例如 coder_doll"
              className="w-64"
            />
            {profileAgents.length > 0 && (
              <select
                value=""
                onChange={(e) => setNewAgent(e.target.value)}
                className="bg-field border border-line rounded-lg px-3 py-1.5 text-sm text-fg"
              >
                <option value="">从已有 Agent 中选择</option>
                {profileAgents.map((alias) => <option key={alias} value={alias}>{alias}</option>)}
              </select>
            )}
            <button onClick={() => void addRecord()} disabled={busy || !newAgent.trim()} className={buttonCls}>
              添加 Agent 记录
            </button>
          </div>
        </div>
        <p className="text-xs text-fg-3 mt-3 leading-relaxed">
          每次检索、读取、写入等操作都按这里的权限即时判定。进行中的对话里，收回的操作仍会出现在本轮的工具说明中，
          调用时被拒绝；新授予的操作从下一轮开始出现。
        </p>
      </section>
    </div>
  );
}
