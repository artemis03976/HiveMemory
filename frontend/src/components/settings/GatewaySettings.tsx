import { SettingSection, SettingRow, Toggle } from '../common/FormControls';
import type { SettingsBaseProps } from '@/types/settings';

export function GatewaySettings({ config, updateConfig }: SettingsBaseProps) {
  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
      <SettingSection title="网关拦截器">
        <SettingRow label="启用拦截器" description="全局启用或禁用消息拦截功能。">
          <Toggle checked={config.gateway.interceptor.enabled} onChange={(v: boolean) => updateConfig('gateway.interceptor.enabled', v)} />
        </SettingRow>
        <SettingRow label="拦截系统指令" description="检测并拦截系统级别的管理指令。">
          <Toggle checked={config.gateway.interceptor.enable_system} onChange={(v: boolean) => updateConfig('gateway.interceptor.enable_system', v)} />
        </SettingRow>
        <SettingRow label="拦截闲聊" description="检测闲聊内容并跳过记忆管线以节省资源。">
          <Toggle checked={config.gateway.interceptor.enable_chat} onChange={(v: boolean) => updateConfig('gateway.interceptor.enable_chat', v)} />
        </SettingRow>
      </SettingSection>
    </div>
  );
}
