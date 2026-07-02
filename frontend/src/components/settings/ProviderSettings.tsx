import { useEffect, useState, useCallback } from 'react';
import { Plus, Pencil, Trash2, RefreshCw, Lock, KeyRound } from 'lucide-react';

import { Input } from '../common/FormControls';
import { useToastStore } from '@/stores';
import {
  fetchProviders,
  upsertProvider,
  deleteProvider,
} from '@/services/providerRegistryApi';
import type { RegisteredProvider, ProviderUpsertPayload } from '@/types/provider';

/** 编辑器草稿：新增与编辑共用 */
interface ProviderDraft {
  name: string;
  api_key: string;
  api_base: string;
}

const EMPTY_DRAFT: ProviderDraft = { name: '', api_key: '', api_base: '' };

/** 提供商凭证编辑器 */
function ProviderEditor({
  draft,
  isNew,
  saving,
  onChange,
  onSubmit,
  onCancel,
}: {
  draft: ProviderDraft;
  isNew: boolean;
  saving: boolean;
  onChange: (patch: Partial<ProviderDraft>) => void;
  onSubmit: () => void;
  onCancel: () => void;
}) {
  const rowCls = 'flex items-center justify-between gap-4 py-2';
  const labelCls = 'text-sm text-slate-300 shrink-0 w-28';

  return (
    <div className="bg-black/20 border border-primary/20 rounded-2xl p-5 space-y-1 ghost-border">
      <div className="text-sm font-bold text-primary mb-3">
        {isNew ? '添加提供商' : `编辑提供商: ${draft.name}`}
      </div>

      <div className={rowCls}>
        <span className={labelCls}>提供商名称</span>
        <Input
          value={draft.name}
          onChange={(v: string) => onChange({ name: v })}
          disabled={!isNew}
          placeholder="deepseek / openai / anthropic"
          className="w-64"
        />
      </div>

      <div className={rowCls}>
        <span className={labelCls}>API Key</span>
        <Input
          value={draft.api_key}
          onChange={(v: string) => onChange({ api_key: v })}
          placeholder={isNew ? '必填' : '留空保持不变'}
          className="w-64"
        />
      </div>

      <div className={rowCls}>
        <span className={labelCls}>API Base</span>
        <Input
          value={draft.api_base}
          onChange={(v: string) => onChange({ api_base: v })}
          placeholder="可选，留空使用提供商默认地址"
          className="w-64"
        />
      </div>

      <div className="flex justify-end gap-2 pt-3">
        <button
          onClick={onCancel}
          disabled={saving}
          className="px-4 py-1.5 text-sm text-slate-400 hover:text-white transition-colors disabled:opacity-50"
        >
          取消
        </button>
        <button
          onClick={onSubmit}
          disabled={saving}
          className="px-4 py-1.5 text-sm font-bold bg-primary/20 hover:bg-primary/30 text-primary rounded-lg border border-primary/30 transition-all disabled:opacity-50"
        >
          {saving ? '保存中...' : '保存'}
        </button>
      </div>
    </div>
  );
}

/** 单条提供商展示行 */
function ProviderRow({
  provider,
  onEdit,
  onDelete,
}: {
  provider: RegisteredProvider;
  onEdit: () => void;
  onDelete: () => void;
}) {
  return (
    <div className="flex items-center justify-between p-4 border-b border-white/5 last:border-0 hover:bg-white/2 transition-colors">
      <div className="flex items-center gap-3 min-w-0">
        <KeyRound className="w-4 h-4 text-slate-500 shrink-0" />
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-slate-200 font-mono">{provider.name}</span>
            {provider.is_from_env && (
              <span className="flex items-center gap-1 text-[10px] font-bold text-amber-400 bg-amber-400/10 px-1.5 py-0.5 rounded uppercase tracking-wide">
                <Lock className="w-2.5 h-2.5" />
                环境变量
              </span>
            )}
          </div>
          <div className="text-xs text-slate-500 mt-0.5 font-mono">
            {provider.api_key_masked ? (
              <span>{provider.api_key_masked}</span>
            ) : (
              <span className="text-slate-600 italic">未配置 API Key</span>
            )}
            {provider.api_base && (
              <span className="ml-2 text-slate-600">· {provider.api_base}</span>
            )}
          </div>
        </div>
      </div>

      <div className="flex items-center gap-1 shrink-0">
        <button
          onClick={onEdit}
          disabled={provider.is_from_env}
          className="p-2 text-slate-400 hover:text-primary hover:bg-white/5 rounded-lg transition-all disabled:opacity-30 disabled:cursor-not-allowed"
          title={provider.is_from_env ? '来自环境变量，请直接修改 .env 文件' : '编辑'}
        >
          <Pencil className="w-4 h-4" />
        </button>
        <button
          onClick={onDelete}
          disabled={provider.is_from_env}
          className="p-2 text-slate-400 hover:text-red-400 hover:bg-white/5 rounded-lg transition-all disabled:opacity-30 disabled:cursor-not-allowed"
          title={provider.is_from_env ? '来自环境变量，无法通过 UI 删除' : '删除'}
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}

/** 提供商凭证管理面板 */
export function ProviderSettings() {
  const { addToast } = useToastStore();
  const [providers, setProviders] = useState<RegisteredProvider[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [draft, setDraft] = useState<ProviderDraft | null>(null);
  const [editingName, setEditingName] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setProviders(await fetchProviders());
    } catch (err) {
      addToast('加载提供商列表失败: ' + (err instanceof Error ? err.message : '未知错误'), 'error');
    } finally {
      setLoading(false);
    }
  }, [addToast]);

  useEffect(() => { load(); }, [load]);

  const startCreate = () => {
    setEditingName(null);
    setDraft({ ...EMPTY_DRAFT });
  };

  const startEdit = (p: RegisteredProvider) => {
    setEditingName(p.name);
    setDraft({ name: p.name, api_key: '', api_base: p.api_base ?? '' });
  };

  const cancel = () => {
    setDraft(null);
    setEditingName(null);
  };

  const submit = async () => {
    if (!draft) return;
    if (!draft.name.trim()) {
      addToast('提供商名称为必填', 'error');
      return;
    }
    if (editingName === null && !draft.api_key.trim()) {
      addToast('新建提供商时 API Key 为必填', 'error');
      return;
    }
    setSaving(true);
    try {
      const payload: ProviderUpsertPayload = {
        api_key: draft.api_key.trim() || null,
        api_base: draft.api_base.trim() || null,
      };
      await upsertProvider(draft.name.trim().toLowerCase(), payload);
      addToast(editingName === null ? '提供商已添加' : '提供商已更新', 'success');
      cancel();
      await load();
    } catch (err) {
      addToast('保存失败: ' + (err instanceof Error ? err.message : '未知错误'), 'error');
    } finally {
      setSaving(false);
    }
  };

  const remove = async (p: RegisteredProvider) => {
    if (!window.confirm(`确定删除提供商 "${p.name}" 的凭证？此操作不可撤销。`)) return;
    try {
      await deleteProvider(p.name);
      addToast('提供商已删除', 'success');
      await load();
    } catch (err) {
      addToast('删除失败: ' + (err instanceof Error ? err.message : '未知错误'), 'error');
    }
  };

  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
      <section className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">提供商凭证</h2>
          <button
            onClick={startCreate}
            disabled={draft !== null}
            className="flex items-center gap-2 px-4 py-2 bg-primary/20 hover:bg-primary/30 text-primary rounded-xl border border-primary/30 transition-all text-sm font-bold disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <Plus className="w-4 h-4" />
            添加提供商
          </button>
        </div>

        {draft && (
          <div className="mb-4">
            <ProviderEditor
              draft={draft}
              isNew={editingName === null}
              saving={saving}
              onChange={(patch) => setDraft((d) => (d ? { ...d, ...patch } : d))}
              onSubmit={submit}
              onCancel={cancel}
            />
          </div>
        )}

        <div className="bg-surface-container-low border border-white/5 rounded-2xl overflow-hidden ghost-border">
          {loading ? (
            <div className="flex items-center justify-center py-12 text-primary">
              <RefreshCw className="w-6 h-6 animate-spin" />
            </div>
          ) : providers.length === 0 ? (
            <div className="py-12 text-center text-sm text-slate-500">
              暂无提供商凭证。点击"添加提供商"配置 API Key。
            </div>
          ) : (
            providers.map((p) => (
              <ProviderRow
                key={p.name}
                provider={p}
                onEdit={() => startEdit(p)}
                onDelete={() => remove(p)}
              />
            ))
          )}
        </div>

        <div className="text-xs text-slate-500 mt-3 leading-relaxed space-y-1">
          <p>
            凭证按 Provider 名称（如 <span className="font-mono">deepseek</span>、<span className="font-mono">openai</span>）
            存储，与模型注册表中的 <span className="font-mono">provider</span> 字段对应。
          </p>
          <p>
            标记为 <span className="text-amber-400">环境变量</span> 的提供商来自
            <span className="font-mono"> HIVEMEMORY__PROVIDERS__&lt;NAME&gt;__API_KEY</span>，
            优先级高于此处配置，只读（请直接修改 <span className="font-mono">.env</span> 文件）。
          </p>
          <p>
            参考 <span className="font-mono">configs/providers.secrets.example.yaml</span> 了解手动配置格式。
          </p>
        </div>
      </section>
    </div>
  );
}
