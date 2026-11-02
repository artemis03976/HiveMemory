import type { HiveMemoryConfig, ValidationError } from '../types/config';

/**
 * 只校验与后端配置结构一致的字段：读取后端配置中不存在的段会抛出，
 * 让设置页退回 mock 数据。
 */
export const validateConfig = (config: HiveMemoryConfig): ValidationError[] => {
  const errors: ValidationError[] = [];

  // Validate logging level
  const validLogLevels = ['DEBUG', 'INFO', 'WARNING', 'ERROR'];
  if (!validLogLevels.includes(config.logging.level)) {
    errors.push({
      field: 'logging.level',
      message: `Log level must be one of: ${validLogLevels.join(', ')}`,
      severity: 'error',
    });
  }

  // Validate LLM temperatures (0-2)
  const validateTemperature = (temp: number, field: string) => {
    if (temp < 0 || temp > 2) {
      errors.push({
        field,
        message: 'Temperature must be between 0 and 2',
        severity: 'error',
      });
    }
  };

  validateTemperature(config.gateway.llm.temperature, 'gateway.llm.temperature');
  validateTemperature(config.patchouli.llm.temperature, 'patchouli.llm.temperature');

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
