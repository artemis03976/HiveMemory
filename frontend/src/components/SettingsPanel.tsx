import { useEffect, useState } from 'react';
import { SlidersHorizontal, Wrench, RefreshCw, RotateCcw, Boxes, KeyRound, Plug, ShieldCheck } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { useSettings } from '../hooks/useSettings';
import PageHeader from './common/PageHeader';
import { useChatUiStore, useToastStore } from '@/stores';
import { useUnsavedSettings } from '@/stores/ui/unsavedSettings';
import { THEME_OPTIONS } from '@/stores/ui/theme';
import { ModelRegistrySettings } from './settings/ModelRegistrySettings';
import { ProviderSettings } from './settings/ProviderSettings';
import { AccessSettings } from './settings/AccessSettings';
import { PluginSettings } from './settings/PluginSettings';
import { TieredSettings } from './settings/TieredSettings';
import { ADVANCED_SECTIONS, COMMON_SECTIONS, changedSettingPaths, settingLabel } from '@/constants/settingsSchema';
import { fetchModels } from '@/services/modelRegistryApi';
import { fetchPlugins } from '@/services/pluginApi';
import type { RegisteredModel } from '@/types/model';

interface Category {
  id: string;
  label: string;
  icon: LucideIcon;
}

const TIERS: Category[] = [
  { id: 'common', label: '常用', icon: SlidersHorizontal },
  { id: 'advanced', label: '高级', icon: Wrench },
];
const MANAGEMENT: Category[] = [
  { id: 'plugins', label: '插件', icon: Plug },
  { id: 'providers', label: '提供商凭证', icon: KeyRound },
  { id: 'models', label: '模型注册表', icon: Boxes },
  { id: 'access', label: '接入与权限', icon: ShieldCheck },
];

/** 外观只保存在这台电脑上，选择即生效，不进入主配置的保存。 */
function AppearanceSection() {
  const theme = useChatUiStore((s) => s.theme);
  const setTheme = useChatUiStore((s) => s.setTheme);
  return (
    <section className="rounded-2xl border border-line bg-surface-container">
      <div className="border-b border-line px-5 pb-3 pt-4">
        <h3 className="font-manrope text-base font-bold text-fg">外观</h3>
        <p className="mt-1 text-[13px] text-fg-3">只保存在这台电脑上，选择即生效</p>
      </div>
      <div className="flex items-center gap-5 px-5 py-3.5">
        <div className="flex flex-1 flex-col gap-1">
          <span className="text-sm font-semibold text-fg">主题</span>
          <span className="text-[13px] text-fg-3">跟随系统时随 Windows 的深浅色设置切换</span>
        </div>
        <div role="group" aria-label="主题" className="flex gap-0.5 rounded-xl border border-line bg-field p-[3px]">
          {THEME_OPTIONS.map((option) => (
            <button
              key={option.mode}
              onClick={() => setTheme(option.mode)}
              aria-pressed={theme === option.mode}
              className={`h-7 rounded-lg px-3 text-[13px] font-semibold transition-colors ${
                theme === option.mode ? 'bg-primary/15 text-primary' : 'text-fg-2 hover:text-fg'
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Settings() {
  const { settingsActiveCategory, setSettingsActiveCategory } = useChatUiStore();
  const activeCategory = [...TIERS, ...MANAGEMENT].some((c) => c.id === settingsActiveCategory) ? settingsActiveCategory : 'common';
  const {
    config, savedConfig, immediateFields, pendingRestart, loading, validationErrors,
    isDirty, isSaving, updateConfig, saveConfig, resetConfig, restoreDefaults,
  } = useSettings();
  const { addToast } = useToastStore();
  const setUnsaved = useUnsavedSettings((s) => s.setDirty);
  const [models, setModels] = useState<RegisteredModel[]>([]);
  // 导航中“插件”旁的数字：已安装插件的 harness 数；插件页的安装与卸载随时更新它。
  const [installedPlugins, setInstalledPlugins] = useState<number | null>(null);

  useEffect(() => {
    fetchModels().then(setModels).catch(() => setModels([]));
    fetchPlugins()
      .then((list) => setInstalledPlugins(list.harnesses.filter((h) => h.installed).length))
      .catch(() => setInstalledPlugins(null));
  }, []);

  // 有未保存的更改时，离开设置页与关闭窗口前都先提醒
  useEffect(() => {
    setUnsaved(isDirty);
    if (!isDirty) return;
    const onBeforeUnload = (event: BeforeUnloadEvent) => event.preventDefault();
    window.addEventListener('beforeunload', onBeforeUnload);
    return () => window.removeEventListener('beforeunload', onBeforeUnload);
  }, [isDirty, setUnsaved]);
  useEffect(() => () => setUnsaved(false), [setUnsaved]);

  const onTierPage = activeCategory === 'common' || activeCategory === 'advanced';
  const changed = config && savedConfig ? changedSettingPaths(savedConfig, config) : [];
  const hasErrors = validationErrors.some((e) => e.severity === 'error');

  const handleSave = async () => {
    try {
      await saveConfig();
      addToast('设置已保存', 'success');
    } catch (err) {
      addToast('保存设置失败：' + (err instanceof Error ? err.message : '未知错误'), 'error');
    }
  };

  const renderCategory = (cat: Category) => {
    const Icon = cat.icon;
    const isActive = activeCategory === cat.id;
    return (
      <button
        key={cat.id}
        onClick={() => setSettingsActiveCategory(cat.id)}
        aria-current={isActive ? 'page' : undefined}
        className={`relative flex h-10 items-center gap-2.5 rounded-xl px-3 text-sm font-semibold transition-colors ${
          isActive ? 'bg-primary/15 text-fg' : 'text-fg-2 hover:bg-surface-container-high hover:text-fg'
        }`}
      >
        {isActive && <span aria-hidden="true" className="absolute left-0 top-2.5 bottom-2.5 w-[3px] rounded-full bg-brand" />}
        <Icon className="h-4 w-4" />
        <span className="flex-1 text-left">{cat.label}</span>
        {cat.id === 'plugins' && installedPlugins !== null && (
          <span className="text-xs font-medium text-fg-3">{installedPlugins}</span>
        )}
      </button>
    );
  };

  return (
    <div className="relative flex h-full flex-1 flex-col overflow-hidden bg-background">
      <PageHeader title="系统设置" description="配置 HiveMemory 的核心参数" />

      <div className="flex flex-1 overflow-hidden">
        <nav aria-label="设置分类" className="flex w-60 shrink-0 flex-col gap-1 overflow-y-auto border-r border-line bg-surface-container-low p-3">
          {TIERS.map(renderCategory)}
          <span className="px-3 pb-1.5 pt-4 text-xs font-semibold text-fg-3">管理</span>
          {MANAGEMENT.map(renderCategory)}
        </nav>

        <div className="scrollbar-thin flex-1 overflow-y-auto bg-atmo">
          <div className="mx-auto max-w-4xl px-8 pb-32 pt-7">
            {onTierPage && loading && (
              <div className="flex justify-center py-20 text-primary"><RefreshCw className="h-7 w-7 animate-spin" /></div>
            )}
            {onTierPage && !loading && !config && (
              <p className="py-20 text-center text-fg-3">设置暂不可用</p>
            )}
            {onTierPage && config && (
              <div className="flex flex-col gap-5">
                <div>
                  <h2 className="font-manrope text-[22px] font-extrabold text-fg">{activeCategory === 'common' ? '常用' : '高级'}</h2>
                  <p className="mt-1 text-sm text-fg-2">
                    {activeCategory === 'common'
                      ? '日常会调整的设置；更多参数在“高级”中。每一项都写明保存后何时生效。'
                      : '一般不需要改动。说明下方是配置文件中的字段名，便于对照 config.local.yaml。'}
                  </p>
                </div>
                {pendingRestart.length > 0 && (
                  <div role="status" className="flex items-center gap-2.5 rounded-xl border border-line bg-magic-metal/10 px-3.5 py-2.5 text-[13px] text-fg">
                    <RotateCcw className="h-4 w-4 shrink-0 text-magic-metal" aria-hidden="true" />
                    <span>
                      <b className="font-semibold">{pendingRestart.length} 项设置等待重启：</b>
                      {pendingRestart.map(settingLabel).join('、')}。重启 HiveMemory 后生效，在此之前仍按原值运行。
                    </span>
                  </div>
                )}
                {activeCategory === 'common' && <AppearanceSection />}
                <TieredSettings
                  sections={activeCategory === 'common' ? COMMON_SECTIONS : ADVANCED_SECTIONS}
                  config={config}
                  savedConfig={savedConfig}
                  immediateFields={immediateFields}
                  pendingRestart={pendingRestart}
                  models={models}
                  showPaths={activeCategory === 'advanced'}
                  updateConfig={updateConfig}
                  restoreDefaults={restoreDefaults}
                />
              </div>
            )}
            {activeCategory === 'plugins' && <PluginSettings onInstalledCount={setInstalledPlugins} />}
            {activeCategory === 'providers' && <ProviderSettings />}
            {activeCategory === 'models' && <ModelRegistrySettings />}
            {activeCategory === 'access' && <AccessSettings />}
          </div>
        </div>
      </div>

      {onTierPage && isDirty && (
        <div
          role="region"
          aria-label="未保存的更改"
          className="absolute bottom-6 left-1/2 z-20 flex w-[min(788px,calc(100%-72px))] -translate-x-1/2 items-center gap-3 rounded-2xl border border-primary/40 bg-surface-container py-2.5 pl-4.5 pr-2.5 shadow-pop"
        >
          <span aria-hidden="true" className="h-2 w-2 shrink-0 rounded-full bg-primary" />
          <span className="min-w-0 flex-1 truncate text-sm text-fg">
            <b className="font-semibold">{changed.length > 0 ? `${changed.length} 项未保存的更改` : '有未保存的更改'}</b>
            {changed.length > 0 && <span className="text-fg-3"> · {changed.map(settingLabel).join('、')}</span>}
          </span>
          <button
            onClick={resetConfig}
            className="h-9 shrink-0 rounded-xl border border-line-strong bg-surface-container-low px-3.5 text-sm font-semibold text-fg hover:bg-surface-container-high"
          >
            放弃
          </button>
          <button
            onClick={handleSave}
            disabled={hasErrors || isSaving}
            className="h-9 shrink-0 rounded-xl bg-brand px-4 text-sm font-semibold text-white shadow-glow disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isSaving ? '保存中…' : '保存'}
          </button>
        </div>
      )}
    </div>
  );
}
