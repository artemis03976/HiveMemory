import { useState, useEffect } from 'react';
import type { HiveMemoryConfig, ValidationError } from '../types/config';
import { validateConfig } from '../utils/configValidation';
import { fetchConfig, updateConfig as updateConfigApi, fetchDefaultConfig } from '../services/configApi';

// Mock data for development when backend is offline
import { MOCK_CONFIG } from '@/constants/settings';

interface UseSettingsReturn {
  config: HiveMemoryConfig | null;
  loading: boolean;
  error: string | null;
  validationErrors: ValidationError[];
  isDirty: boolean;
  updateConfig: (path: string, value: unknown) => void;
  saveConfig: () => Promise<void>;
  resetConfig: () => void;
  resetToDefaults: () => Promise<void>;
}

export const useSettings = (): UseSettingsReturn => {
  const [config, setConfig] = useState<HiveMemoryConfig | null>(null);
  const [originalConfig, setOriginalConfig] = useState<HiveMemoryConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [validationErrors, setValidationErrors] = useState<ValidationError[]>([]);
  const [isDirty, setIsDirty] = useState(false);

  // Load configuration from backend
  useEffect(() => {
    const loadConfig = async () => {
      try {
        setLoading(true);
        const data = await fetchConfig();
        setConfig(data);
        setOriginalConfig(JSON.parse(JSON.stringify(data)));
        setValidationErrors(validateConfig(data));
      } catch (err) {
        console.warn('Failed to load config from backend, using mock data:', err);
        setConfig(MOCK_CONFIG);
        setOriginalConfig(JSON.parse(JSON.stringify(MOCK_CONFIG)));
        setValidationErrors(validateConfig(MOCK_CONFIG));
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    loadConfig();
  }, []);

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

    setConfig(newConfig);
    setValidationErrors(validateConfig(newConfig));
    setIsDirty(JSON.stringify(newConfig) !== JSON.stringify(originalConfig));
  };

  // Save configuration to backend
  const saveConfig = async () => {
    if (!config) return;

    const errors = validateConfig(config);
    const criticalErrors = errors.filter((e) => e.severity === 'error');

    if (criticalErrors.length > 0) {
      throw new Error('Cannot save configuration with validation errors');
    }

    try {
      setLoading(true);
      await updateConfigApi(config);
      setOriginalConfig(JSON.parse(JSON.stringify(config)));
      setIsDirty(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  // Reset to original configuration
  const resetConfig = () => {
    if (originalConfig) {
      setConfig(JSON.parse(JSON.stringify(originalConfig)));
      setValidationErrors(validateConfig(originalConfig));
      setIsDirty(false);
    }
  };

  // Reset to default configuration
  const resetToDefaults = async () => {
    try {
      setLoading(true);
      const data = await fetchDefaultConfig();
      setConfig(data);
      setValidationErrors(validateConfig(data));
      setIsDirty(JSON.stringify(data) !== JSON.stringify(originalConfig));
    } catch (err) {
      console.warn('Failed to load defaults from backend, using mock data:', err);
      setConfig(MOCK_CONFIG);
      setValidationErrors(validateConfig(MOCK_CONFIG));
      setIsDirty(JSON.stringify(MOCK_CONFIG) !== JSON.stringify(originalConfig));
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
    updateConfig,
    saveConfig,
    resetConfig,
    resetToDefaults,
  };
};
