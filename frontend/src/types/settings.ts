import type { HiveMemoryConfig } from './config';

export interface SettingsBaseProps {
  config: HiveMemoryConfig;
  updateConfig: (path: string, value: unknown) => void;
}

export interface SettingsWithValidationProps extends SettingsBaseProps {
  getFieldError: (fieldPath: string) => string | undefined;
}
