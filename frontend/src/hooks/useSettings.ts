import { useState, useEffect } from 'react';
import type { HiveMemoryConfig, ValidationError } from '../types/config';
import { validateConfig } from '../utils/configValidation';
import { fetchConfig, updateConfig as updateConfigApi, fetchDefaultConfig } from '../services/configApi';
import { useDraft } from './useDraft';
import { readPath } from '@/constants/settingsSchema';

// debug 模式下后端不可用时使用的 mock 配置
import { MOCK_CONFIG } from '@/constants/settings';
import { isDebugMode } from '@/utils/debugMode';

interface UseSettingsReturn {
  config: HiveMemoryConfig | null;
  /** 最近一次读取或保存成功的配置，保存栏据此找出未保存的更改。 */
  savedConfig: HiveMemoryConfig | null;
  /** 保存后立即生效的字段路径；不在其中的字段重启后生效。 */
  immediateFields: string[];
  /** 已保存、但要重启才按新值运行的字段路径。 */
  pendingRestart: string[];
  loading: boolean;
  error: string | null;
  validationErrors: ValidationError[];
  isDirty: boolean;
  isSaving: boolean;
  updateConfig: (path: string, value: unknown) => void;
  saveConfig: () => Promise<void>;
  resetConfig: () => void;
  /** 把给出的字段恢复为默认值（不含个人覆盖）；只改草稿，仍需保存。 */
  restoreDefaults: (paths: string[]) => Promise<void>;
}

function setPath(config: HiveMemoryConfig, path: string, value: unknown): HiveMemoryConfig {
  const next = JSON.parse(JSON.stringify(config)) as HiveMemoryConfig;
  const keys = path.split('.');
  let current = next as unknown as Record<string, unknown>;
  for (let i = 0; i < keys.length - 1; i++) {
    const node = current[keys[i]];
    if (typeof node !== 'object' || node === null) current[keys[i]] = {};
    current = current[keys[i]] as Record<string, unknown>;
  }
  current[keys[keys.length - 1]] = value;
  return next;
}

export const useSettings = (): UseSettingsReturn => {
  const [originalConfig, setOriginalConfig] = useState<HiveMemoryConfig | null>(null);
  const [immediateFields, setImmediateFields] = useState<string[]>([]);
  const [pendingRestart, setPendingRestart] = useState<string[]>([]);
  const [defaults, setDefaults] = useState<HiveMemoryConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [validationErrors, setValidationErrors] = useState<ValidationError[]>([]);

  const { draft: config, isDirty, isSaving, updateDraft, save: saveConfigDraft, reset } = useDraft<HiveMemoryConfig | null>({
    initialData: originalConfig,
    onSave: async (draftData) => {
      if (!draftData) return;
      const errors = validateConfig(draftData);
      if (errors.some((e) => e.severity === 'error')) {
        throw new Error('Cannot save configuration with validation errors');
      }
      const saved = await updateConfigApi(draftData);
      setOriginalConfig(JSON.parse(JSON.stringify(saved.config)));
      setImmediateFields(saved.immediate_fields);
      setPendingRestart(saved.pending_restart);
    },
  });

  useEffect(() => {
    const loadConfig = async () => {
      try {
        setLoading(true);
        const { config: data, immediate_fields, pending_restart } = await fetchConfig();
        setOriginalConfig(JSON.parse(JSON.stringify(data)));
        setImmediateFields(immediate_fields);
        setPendingRestart(pending_restart);
        setValidationErrors(validateConfig(data));
      } catch (err) {
        // 非 debug 模式不换用 mock：配置保持为空，页面显示加载失败。
        if (isDebugMode()) {
          console.warn('Failed to load config from backend, using mock data:', err);
          setOriginalConfig(JSON.parse(JSON.stringify(MOCK_CONFIG)));
          setValidationErrors(validateConfig(MOCK_CONFIG));
        }
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    loadConfig();
  }, []);

  useEffect(() => {
    if (config) {
      setValidationErrors(validateConfig(config));
    }
  }, [config]);

  const updateConfig = (path: string, value: unknown) => {
    if (!config) return;
    updateDraft(setPath(config, path, value));
  };

  const restoreDefaults = async (paths: string[]) => {
    if (!config) return;
    const source = defaults ?? (isDebugMode() ? await fetchDefaultConfig().catch(() => MOCK_CONFIG) : await fetchDefaultConfig());
    if (!defaults) setDefaults(source);
    updateDraft(paths.reduce((next, path) => setPath(next, path, readPath(source, path)), config));
  };

  return {
    config,
    savedConfig: originalConfig,
    immediateFields,
    pendingRestart,
    loading,
    error,
    validationErrors,
    isDirty,
    isSaving,
    updateConfig,
    saveConfig: saveConfigDraft,
    resetConfig: reset,
    restoreDefaults,
  };
};
