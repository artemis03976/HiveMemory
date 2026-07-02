import { useEffect, useState, useCallback } from 'react';
import { Plus, Pencil, Trash2, Star, RefreshCw, ChevronDown, ChevronRight } from 'lucide-react';

import { Input } from '../common/FormControls';
import { useToastStore } from '@/stores';
import {
  fetchModels,
  createModel,
  updateModel,
  deleteModel,
} from '@/services/modelRegistryApi';
import { fetchProviders } from '@/services/providerRegistryApi';
import type {
  RegisteredModel,
  ModelCreatePayload,
  ModelUpdatePayload,
} from '@/types/model';
import type { RegisteredProvider } from '@/types/provider';

/** 编辑器草稿：新增与编辑共用的表单状态 */
interface ModelDraft {
  id: string;
  display_name: string;
  litellm_model: string;
  provider: string;
  // api_key / api_base 仅在"高级覆盖"展开时使用
  api_key: string;
  api_base: string;
  temperature: number;
  max_tokens: number;
  top_p: number;
  is_default: boolean;
}

const EMPTY_DRAFT: ModelDraft = {
  id: '',
  display_name: '',
  litellm_model: '',
  provider: '',
  api_key: '',
  api_base: '',
  temperature: 1.0,
  max_tokens: 32768,
  top_p: 1.0,
  is_default: false,
};

/** 模型编辑器 — 新增/编辑共用的内联表单卡片 */
function ModelEditor({
  draft,
  isNew,
  saving,
  providers,
  onChange,
  onSubmit,
  onCancel,
}: {
  draft: ModelDraft;
  isNew: boolean;
  saving: boolean;
  providers: RegisteredProvider[];
  onChange: (patch: Partial<ModelDraft>) => void;
  onSubmit: () => void;
  onCancel: () => void;
}) {
  const [showOverride, setShowOverride] = useState(false);
  const rowCls = 'flex items-center justify-between gap-4 py-2';
  const labelCls = 'text-sm text-slate-300 shrink-0 w-28';

  return (
    <div className="bg-black/20 border border-primary/20 rounded-2xl p-5 space-y-1 ghost-border">
      <div className="text-sm font-bold text-primary mb-3">
        {isNew ? '添加模型' : `编辑模型: ${draft.id}`}
      </div>

      <div className={rowCls}>
        <span className={labelCls}>模型 ID</span>
        <Input
          value={draft.id}
          onChange={(v: string) => onChange({ id: v })}
          disabled={!isNew}
          placeholder="deepseek-chat"
          className="w-64"
        />
      </div>

      <div className={rowCls}>
        <span className={labelCls}>显示名称</span>
        <Input
          value={draft.display_name}
          onChange={(v: string) => onChange({ display_name: v })}
          placeholder="DeepSeek Chat"
          className="w-64"
        />
      </div>

      <div className={rowCls}>
        <span className={labelCls}>模型标识</span>
        <Input
          value={draft.litellm_model}
          onChange={(v: string) => {
            const patch: Partial<ModelDraft> = { litellm_model: v };
            // 自动推导 provider（litellm_model 前缀，如 deepseek/deepseek-chat → deepseek）
            if (v.includes('/') && !draft.provider) {
              patch.provider = v.split('/')[0].toLowerCase();
            }
            onChange(patch);
          }}
          placeholder="deepseek/deepseek-chat"
          className="w-64"
        />
      </div>

      {/* Provider 选择器 */}
      <div className={rowCls}>
        <span className={labelCls}>Provider</span>
        {providers.length > 0 ? (
          <select
            value={draft.provider}
            onChange={(e) => onChange({ provider: e.target.value })}
            className="w-64 h-9 px-3 bg-white/5 border border-white/10 rounded-lg text-sm text-slate-200 focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/30"
          >
            <option value="">-- 选择提供商（或输入自定义）--</option>
            {providers.map((p) => (
              <option key={p.name} value={p.name}>
                {p.name}
                {p.api_key_masked ? ` · ${p.api_key_masked}` : ' · 未配置 Key'}
              </option>
            ))}
          </select>
        ) : (
          <div className="w-64 flex items-center gap-2">
            <Input
              value={draft.provider}
              onChange={(v: string) => onChange({ provider: v })}
              placeholder="如 deepseek（留空自动推导）"
              className="flex-1"
            />
            <span className="text-xs text-amber-400 shrink-0">暂无提供商</span>
          </div>
        )}
      </div>

      {/* 参数调整 */}
      <div className={rowCls}>
        <span className={labelCls}>默认温度</span>
        <Input type="number" value={draft.temperature} onChange={(v: number) => onChange({ temperature: v })} step="0.1" className="w-24" />
      </div>
      <div className={rowCls}>
        <span className={labelCls}>max_tokens</span>
        <Input type="number" value={draft.max_tokens} onChange={(v: number) => onChange({ max_tokens: v })} className="w-32" />
      </div>
      <div className={rowCls}>
        <span className={labelCls}>top_p</span>
        <Input type="number" value={draft.top_p} onChange={(v: number) => onChange({ top_p: v })} step="0.05" className="w-24" />
      </div>
      <div className={rowCls}>
        <span className={labelCls}>设为默认</span>
        <button
          onClick={() => onChange({ is_default: !draft.is_default })}
          className={`relative w-11 h-6 rounded-full transition-colors ${draft.is_default ? 'bg-primary' : 'bg-white/10'}`}
        >
          <div className={`absolute top-1 left-1 w-4 h-4 rounded-full bg-white transition-transform ${draft.is_default ? 'translate-x-5' : 'translate-x-0'}`} />
        </button>
      </div>

      {/* 高级：覆盖 API 凭证（折叠，仅需特殊处理时展开） */}
      <div className="pt-2">
        <button
          type="button"
          onClick={() => setShowOverride((v) => !v)}
          className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-300 transition-colors"
        >
          {showOverride ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
          高级：为本模型单独覆盖 API 凭证
        </button>
        {showOverride && (
          <div className="mt-2 pl-4 border-l border-white/10 space-y-1">
            <p className="text-xs text-slate-500 mb-2">
              通常无需填写，凭证由上方 Provider 提供。仅在此模型需要不同密钥时使用（如同 Provider 下的不同账单主体）。
            </p>
            <div className={rowCls}>
              <span className={labelCls}>API Key</span>
              <Input
                value={draft.api_key}
                onChange={(v: string) => onChange({ api_key: v })}
                placeholder={isNew ? '留空则用 Provider 凭证' : '留空保持不变'}
                className="w-64"
              />
            </div>
            <div className={rowCls}>
              <span className={labelCls}>API Base</span>
              <Input
                value={draft.api_base}
                onChange={(v: string) => onChange({ api_base: v })}
                placeholder="可选，留空用 Provider 默认地址"
                className="w-64"
              />
            </div>
          </div>
        )}
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

/** 单条模型的展示行 */
function ModelRow({
  model,
  onEdit,
  onDelete,
}: {
  model: RegisteredModel;
  onEdit: () => void;
  onDelete: () => void;
}) {
  return (
    <div className="flex items-center justify-between p-4 border-b border-white/5 last:border-0 hover:bg-white/2 transition-colors">
      <div className="flex items-center gap-3 min-w-0">
        {model.is_default ? (
          <Star className="w-4 h-4 text-primary fill-primary shrink-0" />
        ) : (
          <Star className="w-4 h-4 text-slate-600 shrink-0" />
        )}
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-slate-200 truncate">{model.display_name}</span>
            {model.is_default && (
              <span className="text-[10px] font-bold text-primary bg-primary/10 px-1.5 py-0.5 rounded uppercase tracking-wide">默认</span>
            )}
            {model.provider && (
              <span className="text-[10px] font-medium text-slate-500 bg-white/5 px-1.5 py-0.5 rounded font-mono">{model.provider}</span>
            )}
          </div>
          <div className="text-xs text-slate-500 mt-0.5 font-mono truncate">
            {model.litellm_model}
            {model.api_key_masked && (
              <span className="ml-2 text-slate-600">· {model.api_key_masked} <span className="text-amber-500/70">(覆盖)</span></span>
            )}
          </div>
        </div>
      </div>
      <div className="flex items-center gap-1 shrink-0">
        <button onClick={onEdit} className="p-2 text-slate-400 hover:text-primary hover:bg-white/5 rounded-lg transition-all" title="编辑">
          <Pencil className="w-4 h-4" />
        </button>
        <button onClick={onDelete} className="p-2 text-slate-400 hover:text-red-400 hover:bg-white/5 rounded-lg transition-all" title="删除">
          <Trash2 className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}

/** 模型注册表管理面板 — 独立于设置草稿机制，所有变更即时生效 */
export function ModelRegistrySettings() {
  const { addToast } = useToastStore();
  const [models, setModels] = useState<RegisteredModel[]>([]);
  const [providers, setProviders] = useState<RegisteredProvider[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  // draft 非空时展示编辑器；editingId 为 null 表示新增
  const [draft, setDraft] = useState<ModelDraft | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      // 并行加载模型列表和提供商列表
      const [modelList, providerList] = await Promise.all([fetchModels(), fetchProviders()]);
      setModels(modelList);
      setProviders(providerList);
    } catch (err) {
      addToast('加载数据失败: ' + (err instanceof Error ? err.message : '未知错误'), 'error');
    } finally {
      setLoading(false);
    }
  }, [addToast]);

  useEffect(() => { load(); }, [load]);

  const startCreate = () => {
    setEditingId(null);
    setDraft({ ...EMPTY_DRAFT });
  };

  const startEdit = (m: RegisteredModel) => {
    setEditingId(m.id);
    // api_key 脱敏值不回填（留空表示保持不变）
    setDraft({
      id: m.id,
      display_name: m.display_name,
      litellm_model: m.litellm_model,
      provider: m.provider,
      api_key: '',
      api_base: m.api_base ?? '',
      temperature: m.temperature,
      max_tokens: m.max_tokens,
      top_p: m.top_p,
      is_default: m.is_default,
    });
  };

  const cancel = () => {
    setDraft(null);
    setEditingId(null);
  };

  const submit = async () => {
    if (!draft) return;
    if (!draft.id.trim() || !draft.display_name.trim() || !draft.litellm_model.trim()) {
      addToast('模型 ID、显示名称、模型标识均为必填', 'error');
      return;
    }
    setSaving(true);
    try {
      if (editingId === null) {
        const payload: ModelCreatePayload = {
          id: draft.id.trim(),
          display_name: draft.display_name.trim(),
          litellm_model: draft.litellm_model.trim(),
          provider: draft.provider.trim() || undefined,
          api_key: draft.api_key.trim() || null,
          api_base: draft.api_base.trim() || null,
          temperature: draft.temperature,
          max_tokens: draft.max_tokens,
          top_p: draft.top_p,
          is_default: draft.is_default,
        };
        await createModel(payload);
        addToast('模型已添加', 'success');
      } else {
        // 更新：api_key 留空表示保持不变
        const payload: ModelUpdatePayload = {
          display_name: draft.display_name.trim(),
          litellm_model: draft.litellm_model.trim(),
          provider: draft.provider.trim() || undefined,
          api_base: draft.api_base.trim() || null,
          temperature: draft.temperature,
          max_tokens: draft.max_tokens,
          top_p: draft.top_p,
          is_default: draft.is_default,
        };
        if (draft.api_key.trim()) payload.api_key = draft.api_key.trim();
        await updateModel(editingId, payload);
        addToast('模型已更新', 'success');
      }
      cancel();
      await load();
    } catch (err) {
      addToast('保存失败: ' + (err instanceof Error ? err.message : '未知错误'), 'error');
    } finally {
      setSaving(false);
    }
  };

  const remove = async (m: RegisteredModel) => {
    if (!window.confirm(`确定删除模型 "${m.display_name}"？此操作不可撤销。`)) return;
    try {
      await deleteModel(m.id);
      addToast('模型已删除', 'success');
      await load();
    } catch (err) {
      addToast('删除失败: ' + (err instanceof Error ? err.message : '未知错误'), 'error');
    }
  };

  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
      <section className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">模型注册表</h2>
          <button
            onClick={startCreate}
            disabled={draft !== null}
            className="flex items-center gap-2 px-4 py-2 bg-primary/20 hover:bg-primary/30 text-primary rounded-xl border border-primary/30 transition-all text-sm font-bold disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <Plus className="w-4 h-4" />
            添加模型
          </button>
        </div>

        {draft && (
          <div className="mb-4">
            <ModelEditor
              draft={draft}
              isNew={editingId === null}
              saving={saving}
              providers={providers}
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
          ) : models.length === 0 ? (
            <div className="py-12 text-center text-sm text-slate-500">
              注册表为空。请先在"提供商凭证"中配置 API Key，再添加模型。
            </div>
          ) : (
            models.map((m) => (
              <ModelRow key={m.id} model={m} onEdit={() => startEdit(m)} onDelete={() => remove(m)} />
            ))
          )}
        </div>

        <p className="text-xs text-slate-500 mt-3 leading-relaxed">
          模型注册表是系统可用 LLM 的单一数据源。添加模型前请先在"
          <span className="text-slate-400">提供商凭证</span>"中配置对应 Provider 的 API Key，
          然后在此处选择 Provider 挂载即可，无需重复填写密钥。
        </p>
      </section>
    </div>
  );
}
