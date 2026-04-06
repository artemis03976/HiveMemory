import { SettingSection, SettingRow, Input, Toggle, Select } from '../common/FormControls';
import type { SettingsWithValidationProps } from '@/types/settings';

export function KoakumaSettings({ config, updateConfig }: SettingsWithValidationProps) {
  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
      <SettingSection title="Koakuma 运行时">
        <SettingRow label="启用 Koakuma" description="全局启用或禁用 Koakuma。">
          <Toggle checked={config.koakuma.enabled} onChange={(v: boolean) => updateConfig('koakuma.enabled', v)} />
        </SettingRow>
        <SettingRow label="执行超时 (秒)" description="允许单步执行的最大时间。">
          <Input type="number" value={config.koakuma.execution_timeout_seconds} onChange={(v: number) => updateConfig('koakuma.execution_timeout_seconds', v)} className="w-32" />
        </SettingRow>
        <SettingRow label="最大递归深度" description="连续调用 MTP 工具的最大次数。">
          <Input type="number" value={config.koakuma.max_recursion_depth} onChange={(v: number) => updateConfig('koakuma.max_recursion_depth', v)} className="w-24" />
        </SettingRow>
      </SettingSection>

      <SettingSection title="MTP 提示词">
        <SettingRow label="启用系统提示词" description="将 MTP 系统指令注入到对话上下文中。">
          <Toggle checked={config.koakuma.mtp_prompt.enabled} onChange={(v: boolean) => updateConfig('koakuma.mtp_prompt.enabled', v)} />
        </SettingRow>
        <SettingRow label="语言" description="系统指令的语言。">
          <Select 
            value={config.koakuma.mtp_prompt.language}
            onChange={(v: string) => updateConfig('koakuma.mtp_prompt.language', v)}
            options={[
              {label: 'English', value: 'en'},
              {label: 'Chinese (简体中文)', value: 'zh'}
            ]} 
          />
        </SettingRow>
        <SettingRow label="角色" description="预定义的系统角色。">
          <Select 
            value={config.koakuma.mtp_prompt.role}
            onChange={(v: string) => updateConfig('koakuma.mtp_prompt.role', v)}
            options={[
              {label: 'Coder (程序员)', value: 'coder'},
              {label: 'Chat (聊天伴侣)', value: 'chat'},
              {label: 'Default (默认)', value: 'default'}
            ]} 
          />
        </SettingRow>
        <SettingRow label="包含示例" description="在提示词中包含工具调用示例。">
          <Toggle checked={config.koakuma.mtp_prompt.include_demo} onChange={(v: boolean) => updateConfig('koakuma.mtp_prompt.include_demo', v)} />
        </SettingRow>
      </SettingSection>
    </div>
  );
}
