import { useEffect, useState, type ReactNode } from 'react';
import { Bot, ChevronDown, PowerOff, ShieldCheck, Trash2 } from 'lucide-react';
import { useAgentManagement } from '@/hooks/useAgentManagement';
import { useDraft } from '@/hooks/useDraft';
import { useChatUiStore } from '@/stores';
import { AgentSidebar } from './agent/AgentSidebar';
import { TagsSection } from './agent/TagsSection';
import { AgentIcon } from './agent/AgentIcon';
import { Toggle } from './common/FormControls';
import { ConfirmDialog } from './common/ConfirmDialog';
import type { AgentData } from '@/types/agent';
import { fetchModels } from '@/services/modelRegistryApi';
import type { RegisteredModel } from '@/types/model';
import { formatDateTime, formatFullDateTime } from '@/utils/dateTime';

const EMPTY_AGENT: AgentData = {
  id: '',
  alias: '',
  name: '',
  summary: '',
  tags: [],
  systemPrompt: '',
  model: 'default',
  status: 'Inactive',
  config: { model_name: 'default', temperature: 0.7, top_p: null, language: 'zh' },
};

const fieldCls = 'rounded-[9px] border border-line-strong bg-field text-fg focus:border-primary/50 focus:outline-none';

function Card({ title, aside, children }: { title: string; aside?: ReactNode; children: ReactNode }) {
  return (
    <section className="flex flex-col gap-4 rounded-[18px] border border-line bg-surface-container p-5">
      <div className="flex items-baseline gap-2.5">
        <h2 className="flex-1 font-manrope text-base font-bold text-fg">{title}</h2>
        {aside}
      </div>
      {children}
    </section>
  );
}

/** Agent 页：左栏列表，右侧编辑选中 Agent 的名称、模型、系统指令与标签；改动保存后生效。 */
export default function AgentManagement() {
  const {
    filteredAgents,
    selectedAgent,
    selectedId,
    setSelectedId,
    searchQuery,
    setSearchQuery,
    loadError,
    createAgent,
    saveAgent,
    deleteAgent,
  } = useAgentManagement();
  const setActiveNavTab = useChatUiStore((s) => s.setActiveNavTab);
  const setSettingsActiveCategory = useChatUiStore((s) => s.setSettingsActiveCategory);

  const { draft, isDirty, isSaving, updateDraft, save, reset } = useDraft({
    initialData: selectedAgent || EMPTY_AGENT,
    onSave: async (draftData) => {
      await saveAgent(draftData);
    },
  });
  const [models, setModels] = useState<RegisteredModel[]>([]);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  useEffect(() => {
    fetchModels().then(setModels).catch(() => setModels([]));
  }, []);

  // 读不到注册表时不判断“不在注册表中”，免得把可用的模型误标为无效
  const modelUnregistered =
    models.length > 0 && draft.model !== 'default' && !models.some((m) => m.id === draft.model);
  const enabled = draft.status === 'Active';
  const isDraft = draft.id.startsWith('pending_');
  const updateConfig = (patch: Partial<AgentData['config']>) => updateDraft({ config: { ...draft.config, ...patch } });

  const openAccessSettings = () => {
    setSettingsActiveCategory('access');
    setActiveNavTab('settings');
  };

  return (
    <div className="flex h-full flex-1 overflow-hidden bg-background">
      <AgentSidebar
        agents={filteredAgents}
        selectedId={selectedId}
        searchQuery={searchQuery}
        onSelect={setSelectedId}
        onSearch={setSearchQuery}
        onCreate={createAgent}
      />

      {selectedAgent ? (
        <main className="scrollbar-thin min-w-0 flex-1 overflow-y-auto bg-atmo">
          <div className="mx-auto flex max-w-[860px] flex-col gap-4.5 px-9 pb-10 pt-7">
            <section className="flex flex-wrap items-start gap-4.5 rounded-[18px] border border-line bg-surface-container p-5">
              <span aria-hidden="true" className="h-16 w-16 shrink-0 rounded-[20px] bg-brand p-[2px]">
                <span className="flex h-full w-full items-center justify-center rounded-[18px] bg-surface-container text-primary">
                  <AgentIcon alias={draft.alias} className="h-[30px] w-[30px]" strokeWidth={1.8} />
                </span>
              </span>
              <div className="flex min-w-[260px] flex-1 flex-col gap-2">
                <label className="block">
                  <span className="sr-only">名称</span>
                  <input
                    type="text"
                    value={draft.name}
                    onChange={(e) => updateDraft({ name: e.target.value })}
                    placeholder="Agent 名称"
                    className="-ml-1.5 w-full rounded-lg border border-transparent bg-transparent px-1.5 py-0.5 font-manrope text-2xl font-extrabold text-fg placeholder:text-fg-3 hover:border-line focus:border-primary/50 focus:outline-none"
                  />
                </label>
                <span className="flex flex-wrap items-center gap-2 text-xs text-fg-3">
                  别名
                  {isDraft ? (
                    <>
                      <input
                        type="text"
                        aria-label="别名"
                        value={draft.alias}
                        onChange={(e) => updateDraft({ alias: e.target.value })}
                        placeholder="例如 reviewer_doll"
                        className={`h-6 w-56 px-2 font-mono text-xs ${fieldCls}`}
                      />
                      <span>保存后不能再改</span>
                    </>
                  ) : (
                    <span title="对话与访问登记以别名指代这个 Agent，保存后不能再改" className="rounded-md bg-surface-container-high px-1.5 py-px font-mono text-fg-2">
                      {draft.alias}
                    </span>
                  )}
                </span>
                <label className="block">
                  <span className="sr-only">描述</span>
                  <input
                    type="text"
                    value={draft.summary}
                    onChange={(e) => updateDraft({ summary: e.target.value })}
                    placeholder="一句话描述这个 Agent"
                    className={`h-[34px] w-full px-2.5 text-sm text-fg-2 ${fieldCls}`}
                  />
                </label>
              </div>
              <div className="flex shrink-0 flex-col items-end gap-3">
                <label
                  title="停用后，对话中的 Agent 选择与 @ 提及不再列出它，也不能用它发起对话"
                  className={`flex h-8 cursor-pointer items-center gap-2 rounded-full border border-line-strong bg-surface-container-low pl-1.5 pr-3 text-[13px] font-semibold ${
                    enabled ? 'text-magic-wood' : 'text-fg-3'
                  }`}
                >
                  <span className="flex scale-75 items-center">
                    <Toggle checked={enabled} onChange={(on) => updateDraft({ status: on ? 'Active' : 'Inactive' })} label="启用这个 Agent" />
                  </span>
                  {enabled ? '启用' : '已停用'}
                </label>
                <div className="flex gap-2">
                  <button
                    onClick={reset}
                    disabled={!isDirty}
                    className="h-9 rounded-[10px] border border-line-strong bg-surface-container-low px-3.5 text-sm font-semibold text-fg transition-colors hover:bg-surface-container-high disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    撤销更改
                  </button>
                  <button
                    onClick={save}
                    disabled={!isDirty || isSaving}
                    className="h-9 rounded-[10px] bg-brand px-4 text-sm font-semibold text-white shadow-glow transition-transform active:scale-95 disabled:cursor-not-allowed disabled:opacity-50 disabled:shadow-none"
                  >
                    {isSaving ? '保存中…' : '保存'}
                  </button>
                </div>
              </div>
            </section>

            {!enabled && (
              <div role="status" className="flex items-center gap-2.5 rounded-xl border border-line bg-surface-container-low px-3.5 py-2.5 text-[13px] text-fg-2">
                <PowerOff className="h-4 w-4 shrink-0 text-fg-3" aria-hidden="true" />
                这个 Agent 已停用：对话中的 Agent 选择与 @ 提及不再列出它，也不能用它发起新的对话。
              </div>
            )}

            <Card title="模型">
              <div className="flex flex-col gap-1.5">
                <label htmlFor="agent-model" className="text-[13px] font-semibold text-fg-2">使用的模型</label>
                <span className="relative block max-w-[420px]">
                  <select
                    id="agent-model"
                    value={draft.model}
                    onChange={(e) => {
                      const model = e.target.value;
                      updateDraft({ model, config: { ...draft.config, model_name: model } });
                    }}
                    className={`h-[38px] w-full cursor-pointer appearance-none rounded-[10px] border bg-field pl-3 pr-9 text-sm text-fg focus:border-primary/50 focus:outline-none ${
                      modelUnregistered ? 'border-magic-metal' : 'border-line-strong'
                    }`}
                  >
                    <option value="default">模型注册表的默认模型</option>
                    {draft.model !== 'default' && !models.some((m) => m.id === draft.model) && (
                      <option value={draft.model}>{modelUnregistered ? `${draft.model}（不在注册表中）` : draft.model}</option>
                    )}
                    {models.map((m) => (
                      <option key={m.id} value={m.id}>{m.display_name ? `${m.id} · ${m.display_name}` : m.id}</option>
                    ))}
                  </select>
                  <ChevronDown className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-fg-3" aria-hidden="true" />
                </span>
                <span className={`text-xs ${modelUnregistered ? 'text-magic-metal' : 'text-fg-3'}`}>
                  {modelUnregistered
                    ? `当前值 ${draft.model} 不在模型注册表中，对话时会解析失败；请选择一个已登记的模型。`
                    : '从模型注册表中选择；要添加模型，到“系统设置 · 模型注册表”。'}
                </span>
              </div>
              <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
                <label className="flex flex-col gap-2">
                  <span className="flex items-baseline text-[13px] font-semibold text-fg-2">
                    温度<span className="flex-1" /><span className="font-mono text-fg">{draft.config.temperature.toFixed(1)}</span>
                  </span>
                  <input
                    type="range"
                    min="0"
                    max="2"
                    step="0.1"
                    value={draft.config.temperature}
                    onChange={(e) => updateConfig({ temperature: parseFloat(e.target.value) })}
                    className="w-full accent-primary"
                  />
                  <span className="text-xs text-fg-3">越高回复越发散，0 表示尽量确定</span>
                </label>
                <div className="flex flex-col gap-2">
                  <label htmlFor="agent-top-p" className="flex items-baseline text-[13px] font-semibold text-fg-2">
                    Top P<span className="flex-1" />
                    {draft.config.top_p === null
                      ? <span className="text-fg-3">模型默认</span>
                      : <span className="font-mono text-fg">{draft.config.top_p.toFixed(2)}</span>}
                  </label>
                  {/* 留空时滑块停在 1 并变淡；拖动即为这个 Agent 单独设置 */}
                  <input
                    id="agent-top-p"
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={draft.config.top_p ?? 1}
                    onChange={(e) => updateConfig({ top_p: parseFloat(e.target.value) })}
                    className={`w-full accent-primary ${draft.config.top_p === null ? 'opacity-50' : ''}`}
                  />
                  <span className="text-xs text-fg-3">
                    一般不用改，留空时沿用模型定义
                    {draft.config.top_p !== null && (
                      <>
                        {' · '}
                        <button onClick={() => updateConfig({ top_p: null })} className="font-semibold text-primary hover:underline">
                          改回沿用模型定义
                        </button>
                      </>
                    )}
                  </span>
                </div>
              </div>
            </Card>

            <Card title="系统指令" aside={<span className="text-xs text-fg-3">{draft.systemPrompt.length} 字</span>}>
              <label className="block">
                <span className="sr-only">系统指令</span>
                <textarea
                  value={draft.systemPrompt}
                  onChange={(e) => updateDraft({ systemPrompt: e.target.value })}
                  rows={7}
                  placeholder="这个 Agent 的人设与行为准则"
                  className="w-full resize-y rounded-xl border border-line-strong bg-field px-3.5 py-3 text-sm leading-7 text-fg placeholder:text-fg-3 focus:border-primary/50 focus:outline-none"
                />
              </label>
            </Card>

            <Card title="标签">
              <TagsSection tags={draft.tags} onChange={(tags) => updateDraft({ tags })} />
            </Card>

            <section className="flex flex-wrap items-center gap-3.5 rounded-[18px] border border-line bg-surface-container-low px-5 py-4">
              <ShieldCheck className="h-5 w-5 shrink-0 text-fg-3" aria-hidden="true" />
              <div className="flex min-w-[240px] flex-1 flex-col gap-1">
                <span className="font-semibold text-fg">操作权限不在这里设置</span>
                <span className="text-[13px] text-fg-2">它能检索、读取、写入、更新、运行或调用哪些记忆操作，由 Workspace 的访问登记决定。</span>
              </div>
              <button
                onClick={openAccessSettings}
                className="h-[34px] whitespace-nowrap rounded-[10px] border border-line-strong bg-surface-container px-3 text-[13px] font-semibold text-fg transition-colors hover:bg-surface-container-high"
              >
                去“接入与权限” →
              </button>
            </section>

            <div className="flex items-center gap-3 px-1 pt-1">
              <span className="flex-1 text-xs text-fg-3">
                {draft.createdAt && draft.updatedAt ? (
                  <>
                    <span title={formatFullDateTime(Date.parse(draft.createdAt))}>创建于 {formatDateTime(Date.parse(draft.createdAt))}</span>
                    {' · '}
                    <span title={formatFullDateTime(Date.parse(draft.updatedAt))}>最近保存 {formatDateTime(Date.parse(draft.updatedAt))}</span>
                  </>
                ) : (
                  isDraft && '尚未保存'
                )}
              </span>
              <button
                onClick={() => setConfirmingDelete(true)}
                className="inline-flex h-[34px] items-center gap-1.5 rounded-[10px] px-3 text-[13px] font-semibold text-magic-fire transition-colors hover:bg-magic-fire/12"
              >
                <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
                删除 Agent
              </button>
            </div>
          </div>
        </main>
      ) : (
        <div className="flex flex-1 flex-col items-center justify-center gap-3 bg-atmo text-fg-3">
          <Bot className="h-14 w-14 opacity-25" aria-hidden="true" />
          <p className="text-[13px]">{loadError ?? '从左侧选择一个 Agent 查看与编辑'}</p>
        </div>
      )}

      <ConfirmDialog
        isOpen={confirmingDelete}
        title={`删除“${draft.name || draft.alias}”？`}
        message="删除后无法恢复。对话中已经使用过它的回复不受影响。"
        confirmText="删除"
        onConfirm={() => void deleteAgent(draft.id)}
        onCancel={() => setConfirmingDelete(false)}
      />
    </div>
  );
}
