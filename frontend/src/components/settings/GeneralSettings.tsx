import { SettingSection, SettingRow, Input, Toggle, Select } from '../common/FormControls';
import type { SettingsWithValidationProps } from '@/types/settings';

export function GeneralSettings({ config, updateConfig, getFieldError }: SettingsWithValidationProps) {
  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
      <SettingSection title="系统">
        <SettingRow label="系统名称" description="HiveMemory 实例的名称。">
          <Input value={config.system.name} onChange={(v: string) => updateConfig('system.name', v)} disabled className="w-48 opacity-50 cursor-not-allowed" />
        </SettingRow>
        <SettingRow label="版本" description="当前系统版本。">
          <Input value={config.system.version} disabled className="w-48 opacity-50 cursor-not-allowed" />
        </SettingRow>
        <SettingRow label="调试模式" description="启用详细日志和调试功能。">
          <Toggle checked={config.system.debug} onChange={(v: boolean) => updateConfig('system.debug', v)} />
        </SettingRow>
      </SettingSection>

      <SettingSection title="日志">
        <SettingRow label="日志级别" description="最低日志输出级别。">
          <Select 
            value={config.logging.level}
            onChange={(v: string) => updateConfig('logging.level', v)}
            error={getFieldError('logging.level')}
            options={[
              {label: 'DEBUG', value: 'DEBUG'},
              {label: 'INFO', value: 'INFO'},
              {label: 'WARNING', value: 'WARNING'},
              {label: 'ERROR', value: 'ERROR'}
            ]} 
          />
        </SettingRow>
        <SettingRow label="控制台输出" description="将日志打印到标准输出。">
          <Toggle checked={config.logging.console_output} onChange={(v: boolean) => updateConfig('logging.console_output', v)} />
        </SettingRow>
        <SettingRow label="日志文件路径" description="保存日志文件的路径。">
          <Input value={config.logging.file_path || ''} onChange={(v: string) => updateConfig('logging.file_path', v || null)} placeholder="可选" className="w-64" />
        </SettingRow>
      </SettingSection>
    </div>
  );
}
