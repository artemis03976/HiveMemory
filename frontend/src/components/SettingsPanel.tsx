import { Settings as SettingsIcon, Cpu, Database, Network, Activity, Save, RefreshCw, Undo2, TerminalSquare, Search, Eye, Sparkles } from 'lucide-react';
import { useSettings } from '../hooks/useSettings';
import { useToastStore } from '@/stores/toastStore';
import { useChatUiStore } from '@/stores/chatUiStore';
import { GeneralSettings } from './settings/GeneralSettings';
import { ModelSettings } from './settings/ModelSettings';
import { InfrastructureSettings } from './settings/InfrastructureSettings';
import { GatewaySettings } from './settings/GatewaySettings';
import { PerceptionSettings } from './settings/PerceptionSettings';
import { GenerationSettings } from './settings/GenerationSettings';
import { RetrievalSettings } from './settings/RetrievalSettings';
import { LifecycleSettings } from './settings/LifecycleSettings';
import { KoakumaSettings } from './settings/KoakumaSettings';

const CATEGORIES = [
  { id: 'general', label: '通用设置', icon: SettingsIcon },
  { id: 'models', label: '模型配置', icon: Cpu },
  { id: 'infrastructure', label: '基础设施', icon: Database },
  { id: 'gateway', label: '全局网关', icon: Network },
  { id: 'perception', label: '记忆感知', icon: Eye },
  { id: 'generation', label: '记忆生成', icon: Sparkles },
  { id: 'retrieval', label: '记忆检索', icon: Search },
  { id: 'lifecycle', label: '生命周期', icon: Activity },
  { id: 'koakuma', label: 'MTP引擎 (Koakuma)', icon: TerminalSquare },
];

export default function Settings() {
  const { settingsActiveCategory: activeCategory, setSettingsActiveCategory: setActiveCategory } = useChatUiStore();
  const {
    config,
    loading,
    validationErrors,
    isDirty,
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
      <div className="flex-1 flex items-center justify-center h-full bg-background text-slate-400">
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
      <header className="flex items-center justify-between px-8 py-6 border-b border-white/5 bg-surface/30 backdrop-blur-md z-10 shrink-0">
        <div>
          <h1 className="text-2xl font-black tracking-tighter text-primary drop-shadow-[0_0_12px_rgba(197,154,255,0.3)] flex items-center gap-3">
            <SettingsIcon className="w-6 h-6" />
            系统设置
          </h1>
          <p className="text-sm text-slate-400 mt-1 font-medium">配置 HiveMemory 核心参数</p>
        </div>
        <div className="flex items-center gap-3">
          <button 
            onClick={resetToDefaults}
            className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 text-white/70 rounded-xl border border-white/10 transition-all"
          >
            <RefreshCw className="w-4 h-4" />
            <span className="text-sm font-bold tracking-wide">恢复默认</span>
          </button>
          {isDirty && (
            <button 
              onClick={resetConfig}
              className="flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 text-white/70 rounded-xl border border-white/10 transition-all"
            >
              <Undo2 className="w-4 h-4" />
              <span className="text-sm font-bold tracking-wide">放弃更改</span>
            </button>
          )}
          <button 
            onClick={handleSave}
            disabled={!isDirty || hasErrors}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl border transition-all ${
              !isDirty || hasErrors
                ? 'bg-primary/5 text-primary/50 border-primary/10 cursor-not-allowed'
                : 'bg-primary/20 hover:bg-primary/30 text-primary border-primary/30 shadow-[0_0_15px_rgba(197,154,255,0.2)] hover:shadow-[0_0_25px_rgba(197,154,255,0.4)]'
            }`}
          >
            <Save className="w-4 h-4" />
            <span className="text-sm font-bold tracking-wide">保存更改</span>
          </button>
        </div>
      </header>

      {/* 侧边栏 */}
      <div className="flex flex-1 overflow-hidden">
        {/* Settings Sidebar */}
        <div className="w-64 border-r border-white/5 bg-surface-container-lowest/50 p-4 flex flex-col gap-2 overflow-y-auto">
          {CATEGORIES.map(cat => {
            const Icon = cat.icon;
            const isActive = activeCategory === cat.id;
            return (
              <button
                key={cat.id}
                onClick={() => setActiveCategory(cat.id)}
                className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all text-sm font-medium ${
                  isActive 
                    ? 'bg-primary/10 text-primary border border-primary/20 shadow-[inset_0_0_20px_rgba(197,154,255,0.05)]' 
                    : 'text-slate-400 hover:bg-white/5 hover:text-slate-200 border border-transparent'
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
            {activeCategory === 'models' && <ModelSettings config={config} updateConfig={updateConfig} getFieldError={getFieldError} />}
            {activeCategory === 'infrastructure' && <InfrastructureSettings config={config} updateConfig={updateConfig} getFieldError={getFieldError} />}
            {activeCategory === 'gateway' && <GatewaySettings config={config} updateConfig={updateConfig} />}
            {activeCategory === 'perception' && <PerceptionSettings config={config} updateConfig={updateConfig} getFieldError={getFieldError} />}
            {activeCategory === 'generation' && <GenerationSettings config={config} updateConfig={updateConfig} getFieldError={getFieldError} />}
            {activeCategory === 'retrieval' && <RetrievalSettings config={config} updateConfig={updateConfig} getFieldError={getFieldError} />}
            {activeCategory === 'lifecycle' && <LifecycleSettings config={config} updateConfig={updateConfig} getFieldError={getFieldError} />}
            {activeCategory === 'koakuma' && <KoakumaSettings config={config} updateConfig={updateConfig} />}
          </div>
        </div>
      </div>
    </div>
  );
}
