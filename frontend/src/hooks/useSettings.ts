import { useState, useEffect } from 'react';
import type { HiveMemoryConfig, ValidationError } from '../types/config';
import { validateConfig } from '../utils/configValidation';
import { fetchConfig, updateConfig as updateConfigApi, fetchDefaultConfig } from '../services/configApi';
import { useDraft } from './useDraft';

// Mock data for development when backend is offline
import { MOCK_CONFIG } from '@/constants/settings';

interface UseSettingsReturn {
  config: HiveMemoryConfig | null;
  loading: boolean;
  error: string | null;
  validationErrors: ValidationError[];
  isDirty: boolean;
  isSaving: boolean;
  updateConfig: (path: string, value: unknown) => void;
  saveConfig: () => Promise<void>;
  resetConfig: () => void;
  resetToDefaults: () => Promise<void>;
}

export const useSettings = (): UseSettingsReturn => {
  const [originalConfig, setOriginalConfig] = useState<HiveMemoryConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [validationErrors, setValidationErrors] = useState<ValidationError[]>([]);

  const { draft: config, isDirty, isSaving, updateDraft, save: saveConfigDraft, reset } = useDraft<HiveMemoryConfig | null>({
    initialData: originalConfig,
    onSave: async (draftData) => {
      if (!draftData) return;
      const errors = validateConfig(draftData);
      const criticalErrors = errors.filter((e) => e.severity === 'error');
      
      if (criticalErrors.length > 0) {
        throw new Error('Cannot save configuration with validation errors');
      }

      await updateConfigApi(draftData);
      setOriginalConfig(JSON.parse(JSON.stringify(draftData)));
    },
  });

  // Load configuration from backend
  useEffect(() => {
    const loadConfig = async () => {
      try {
        setLoading(true);
        const data = await fetchConfig();
        setOriginalConfig(JSON.parse(JSON.stringify(data)));
        setValidationErrors(validateConfig(data));
      } catch (err) {
        console.warn('Failed to load config from backend, using mock data:', err);
        setOriginalConfig(JSON.parse(JSON.stringify(MOCK_CONFIG)));
        setValidationErrors(validateConfig(MOCK_CONFIG));
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    loadConfig();
  }, []);

  // 监听 config 变化更新校验错误
  useEffect(() => {
    if (config) {
      setValidationErrors(validateConfig(config));
    }
  }, [config]);

  // Update a specific configuration value
  const updateConfig = (path: string, value: unknown) => {
    if (!config) return;

    const newConfig = JSON.parse(JSON.stringify(config)) as HiveMemoryConfig;
    const keys = path.split('.');
    let current: Record<string, unknown> = newConfig as unknown as Record<string, unknown>;

    for (let i = 0; i < keys.length - 1; i++) {
      const key = keys[i];
      const next = current[key];
      if (typeof next !== 'object' || next === null) {
        current[key] = {};
      }
      current = current[key] as Record<string, unknown>;
    }

    current[keys[keys.length - 1]] = value;

    updateDraft(newConfig);
  };

  // Reset to default configuration
  const resetToDefaults = async () => {
    try {
      setLoading(true);
      const data = await fetchDefaultConfig();
      updateDraft(data);
    } catch (err) {
      console.warn('Failed to load defaults from backend, using mock data:', err);
      updateDraft(MOCK_CONFIG);
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  return {
    config,
    loading,
    error,
    validationErrors,
    isDirty,
    isSaving,
    updateConfig,
    saveConfig: saveConfigDraft,
    resetConfig: reset,
    resetToDefaults,
  };
};
