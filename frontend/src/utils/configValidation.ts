import type { HiveMemoryConfig, ValidationError } from '../types/config';
import { readPath } from '@/constants/settingsSchema';

/** 保存前的本地校验，按字段路径读取；其余取值由后端在保存时校验。 */
export const validateConfig = (config: HiveMemoryConfig): ValidationError[] => {
  const errors: ValidationError[] = [];

  // Validate logging level
  const validLogLevels = ['DEBUG', 'INFO', 'WARNING', 'ERROR'];
  if (!validLogLevels.includes(readPath(config, 'logging.level') as string)) {
    errors.push({
      field: 'logging.level',
      message: `Log level must be one of: ${validLogLevels.join(', ')}`,
      severity: 'error',
    });
  }

  // Validate LLM temperatures (0-2); null 表示沿用模型定义的默认值
  const validateTemperature = (field: string) => {
    const temp = readPath(config, field);
    if (typeof temp === 'number' && (temp < 0 || temp > 2)) {
      errors.push({
        field,
        message: 'Temperature must be between 0 and 2',
        severity: 'error',
      });
    }
  };

  validateTemperature('gateway.llm.temperature');
  validateTemperature('patchouli.llm.temperature');

  return errors;
};

export const validateField = (
  field: string,
  _value: unknown,
  config: HiveMemoryConfig
): ValidationError | null => {
  const errors = validateConfig(config);
  return errors.find((e) => e.field === field) || null;
};
