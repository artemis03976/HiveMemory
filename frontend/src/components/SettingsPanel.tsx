import { Settings as SettingsIcon, Cpu, Database, Network, Activity, Save, RefreshCw, Undo2, TerminalSquare, Search, Eye, Sparkles, Boxes, KeyRound, ShieldCheck } from 'lucide-react';
import { useSettings } from '../hooks/useSettings';
import { useChatUiStore, useToastStore } from '@/stores';
import { GeneralSettings } from './settings/GeneralSettings';
import { ModelRegistrySettings } from './settings/ModelRegistrySettings';
import { ProviderSettings } from './settings/ProviderSettings';
import { AccessSettings } from './settings/AccessSettings';
import { EngineSettings } from './settings/EngineSettings';

const CATEGORIES = [
  { id: 'general', label: '通用设置', icon: SettingsIcon },
  { id: 'providers', label: '提供商凭证', icon: KeyRound },
  { id: 'models', label: '模型注册表', icon: Boxes },
  { id: 'access', label: '接入与权限', icon: ShieldCheck },
  { id: 'engines', label: '内部引擎', icon: Cpu },
  { id: 'infrastructure', label: '基础设施', icon: Database },
  { id: 'gateway', label: '全局网关', icon: Network },
  { id: 'perception', label: '记忆感知', icon: Eye },
  { id: 'generation', label: '记忆生成', icon: Sparkles },
  { id: 'retrieval', label: '记忆检索', icon: Search },
  { id: 'lifecycle', label: '生命周期', icon: Activity },
  { id: 'koakuma', label: 'MTP引擎 (Koakuma)', icon: TerminalSquare },
];

// 这些分类的表单仍按旧的前端配置结构读写，后端配置中没有对应的段，渲染真实配置会报错。
const UNALIGNED_CATEGORIES = new Set([
  'infrastructure',
  'gateway',
  'perception',
  'generation',
  'retrieval',
  'lifecycle',
  'koakuma',
]);

function UnalignedCategoryNotice() {
  return (
    <div className="rounded-xl border border-magic-metal/20 bg-magic-metal/5 px-5 py-4 text-sm text-fg-2 leading-relaxed">
      该分类尚未与后端配置结构对齐，暂不提供表单。需要修改时，请在
      <code className="mx-1 font-mono text-magic-metal/80">configs/config.local.yaml</code>
      中写入要改的字段并重启服务。
    </div>
  );
}

export default function Settings() {
  const { settingsActiveCategory: activeCategory, setSettingsActiveCategory: setActiveCategory } = useChatUiStore();
  const {
    config,
    loading,
    validationErrors,
    isDirty,
    isSaving,
    updateConfig,
    saveConfig,
    resetConfig,
    resetToDefaults,
  } = useSettings();
  const { addToast } = useToastStore();

  const getFieldError = (fieldPath: string) => {
    return validationErrors.find(e => e.field === fieldPath)?.message;
  };

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center h-full bg-background">
        <div className="animate-spin text-primary">
          <RefreshCw className="w-8 h-8" />
        </div>
      </div>
    );
  }

  if (!config) {
    return (
      <div className="flex-1 flex items-center justify-center h-full bg-background text-fg-3">
        配置加载失败，请稍后重试。
      </div>
    );
  }

  const hasErrors = validationErrors.some(e => e.severity === 'error');

  const handleSave = async () => {
    try {
      await saveConfig();
      addToast('设置保存成功', 'success');
    } catch (err) {
      addToast('保存设置失败: ' + (err instanceof Error ? err.message : '未知错误'), 'error');
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-background overflow-hidden">
      {/* Header */}
      <header className="flex items-center justify-between px-8 py-6 border-b border-line bg-surface/30 backdrop-blur-md z-10 shrink-0">
        <div>
          <h1 className="text-2xl font-black tracking-tighter text-primary flex items-center gap-3">
            <SettingsIcon className="w-6 h-6" />
            系统设置
          </h1>
          <p className="text-sm text-fg-3 mt-1 font-medium">配置 HiveMemory 核心参数</p>
        </div>
        <div className="flex items-center gap-3">
          <button 
            onClick={resetToDefaults}
            className="flex items-center gap-2 px-4 py-2 bg-ink/5 hover:bg-ink/10 text-fg-2 rounded-xl border border-line transition-all"
          >
            <RefreshCw className="w-4 h-4" />
            <span className="text-sm font-bold tracking-wide">恢复默认</span>
          </button>
          {isDirty && (
            <button 
              onClick={resetConfig}
              className="flex items-center gap-2 px-4 py-2 bg-ink/5 hover:bg-ink/10 text-fg-2 rounded-xl border border-line transition-all"
            >
              <Undo2 className="w-4 h-4" />
              <span className="text-sm font-bold tracking-wide">放弃更改</span>
            </button>
          )}
          <button 
            onClick={handleSave}
            disabled={!isDirty || hasErrors || isSaving}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl border transition-all ${
              !isDirty || hasErrors || isSaving
                ? 'bg-primary/5 text-primary/50 border-primary/10 cursor-not-allowed'
                : 'bg-primary/20 hover:bg-primary/30 text-primary border-primary/30'
            }`}
          >
            {isSaving ? (
              <span className="w-4 h-4 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
            ) : (
              <Save className="w-4 h-4" />
            )}
            <span className="text-sm font-bold tracking-wide">{isSaving ? '保存中...' : '保存更改'}</span>
          </button>
        </div>
      </header>

      {/* 侧边栏 */}
      <div className="flex flex-1 overflow-hidden">
        {/* Settings Sidebar */}
        <div className="w-64 border-r border-line bg-surface-container-lowest/50 p-4 flex flex-col gap-2 overflow-y-auto">
          {CATEGORIES.map(cat => {
            const Icon = cat.icon;
            const isActive = activeCategory === cat.id;
            return (
              <button
                key={cat.id}
                onClick={() => setActiveCategory(cat.id)}
                className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all text-sm font-medium ${
                  isActive 
                    ? 'bg-primary/10 text-primary border border-primary/20'
                    : 'text-fg-3 hover:bg-ink/5 hover:text-fg border border-transparent'
                }`}
              >
                <Icon className="w-4 h-4" />
                {cat.label}
              </button>
            );
          })}
        </div>

        {/* Settings Content */}
        <div className="flex-1 overflow-y-auto p-8 scrollbar-hide">
          <div className="max-w-4xl mx-auto pb-12">
            {activeCategory === 'general' && <GeneralSettings config={config} updateConfig={updateConfig} getFieldError={getFieldError} />}
            {activeCategory === 'providers' && <ProviderSettings />}
            {activeCategory === 'models' && <ModelRegistrySettings />}
            {activeCategory === 'access' && <AccessSettings />}
            {activeCategory === 'engines' && <EngineSettings config={config} updateConfig={updateConfig} getFieldError={getFieldError} />}
            {UNALIGNED_CATEGORIES.has(activeCategory) && <UnalignedCategoryNotice />}
          </div>
        </div>
      </div>
    </div>
  );
}
