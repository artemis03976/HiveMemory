import { useState, useEffect } from 'react';
import type { HiveMemoryConfig, ValidationError } from '../types/config';
import { validateConfig } from '../utils/configValidation';

interface UseSettingsReturn {
  config: HiveMemoryConfig | null;
  loading: boolean;
  error: string | null;
  validationErrors: ValidationError[];
  isDirty: boolean;
  updateConfig: (path: string, value: any) => void;
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
        const response = await fetch('/api/config');
        if (!response.ok) {
          throw new Error('Failed to load configuration');
        }
        const data = await response.json();
        setConfig(data);
        setOriginalConfig(JSON.parse(JSON.stringify(data)));
        setValidationErrors(validateConfig(data));
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    loadConfig();
  }, []);

  // Update a specific configuration value
  const updateConfig = (path: string, value: any) => {
    if (!config) return;

    const newConfig = JSON.parse(JSON.stringify(config));
    const keys = path.split('.');
    let current: any = newConfig;

    for (let i = 0; i < keys.length - 1; i++) {
      current = current[keys[i]];
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
      const response = await fetch('/api/config', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });

      if (!response.ok) {
        throw new Error('Failed to save configuration');
      }

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
      const response = await fetch('/api/config/defaults');
      if (!response.ok) {
        throw new Error('Failed to load default configuration');
      }
      const data = await response.json();
      setConfig(data);
      setValidationErrors(validateConfig(data));
      setIsDirty(JSON.stringify(data) !== JSON.stringify(originalConfig));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      throw err;
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
