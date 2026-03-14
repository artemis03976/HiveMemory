import React from 'react';
import { CategorySection } from './CategorySection';
import {
  NumberInput,
  ToggleSwitch,
  SelectDropdown,
  FilePathInput,
} from './FormControls';
import type { HiveMemoryConfig } from '../../types/config';

interface KoakumaCategoriesProps {
  config: HiveMemoryConfig;
  updateConfig: (path: string, value: any) => void;
  getFieldError: (field: string) => string | undefined;
}

export const KoakumaCategories: React.FC<KoakumaCategoriesProps> = ({
  config,
  updateConfig,
  getFieldError,
}) => {
  return (
    <CategorySection
      title="高级设置 (Koakuma MTP 运行时)"
      paramCount={15}
      accentColor="hsl(var(--foreground) / 0.6)"
    >
      <div className="space-y-4">
        <h4 className="text-sm font-semibold text-foreground/90">运行时配置</h4>
        <ToggleSwitch
          label="启用"
          value={config.koakuma.enabled}
          onChange={(v) => updateConfig('koakuma.enabled', v)}
        />
        <NumberInput
          label="执行超时 (秒)"
          value={config.koakuma.execution_timeout_seconds}
          onChange={(v) => updateConfig('koakuma.execution_timeout_seconds', v)}
          min={1}
        />
        <NumberInput
          label="最大递归深度"
          value={config.koakuma.max_recursion_depth}
          onChange={(v) => updateConfig('koakuma.max_recursion_depth', v)}
          min={1}
        />
        <NumberInput
          label="工具缓存大小"
          value={config.koakuma.tool_cache_size}
          onChange={(v) => updateConfig('koakuma.tool_cache_size', v)}
          min={1}
        />
        <NumberInput
          label="Python REPL 超时 (秒)"
          value={config.koakuma.python_repl_timeout_seconds}
          onChange={(v) => updateConfig('koakuma.python_repl_timeout_seconds', v)}
          min={1}
        />
        <FilePathInput
          label="工作区路径"
          value={config.koakuma.workspace_path}
          onChange={(v) => updateConfig('koakuma.workspace_path', v)}
        />
        <NumberInput
          label="文件读取最大字节数"
          value={config.koakuma.file_read_max_bytes}
          onChange={(v) => updateConfig('koakuma.file_read_max_bytes', v)}
          min={1}
        />
        <NumberInput
          label="文件写入最大字节数"
          value={config.koakuma.file_write_max_bytes}
          onChange={(v) => updateConfig('koakuma.file_write_max_bytes', v)}
          min={1}
        />
        <NumberInput
          label="网络搜索超时 (秒)"
          value={config.koakuma.web_search_timeout_seconds}
          onChange={(v) => updateConfig('koakuma.web_search_timeout_seconds', v)}
          min={1}
        />

        <h4 className="text-sm font-semibold text-foreground/90 mt-6">MTP 提示词配置</h4>
        <ToggleSwitch
          label="启用"
          value={config.koakuma.mtp_prompt.enabled}
          onChange={(v) => updateConfig('koakuma.mtp_prompt.enabled', v)}
        />
        <SelectDropdown
          label="语言"
          value={config.koakuma.mtp_prompt.language}
          onChange={(v) => updateConfig('koakuma.mtp_prompt.language', v)}
          options={[
            { value: 'zh', label: '中文' },
            { value: 'en', label: '英文' },
          ]}
          error={getFieldError('koakuma.mtp_prompt.language')}
        />
        <SelectDropdown
          label="角色"
          value={config.koakuma.mtp_prompt.role}
          onChange={(v) => updateConfig('koakuma.mtp_prompt.role', v)}
          options={[
            { value: 'coder', label: '程序员' },
            { value: 'chat', label: '聊天' },
            { value: 'default', label: '默认' },
          ]}
          error={getFieldError('koakuma.mtp_prompt.role')}
        />
        <ToggleSwitch
          label="包含演示"
          value={config.koakuma.mtp_prompt.include_demo}
          onChange={(v) => updateConfig('koakuma.mtp_prompt.include_demo', v)}
        />
        <ToggleSwitch
          label="包含错误处理"
          value={config.koakuma.mtp_prompt.include_error_handling}
          onChange={(v) => updateConfig('koakuma.mtp_prompt.include_error_handling', v)}
        />
        <ToggleSwitch
          label="包含内核工具"
          value={config.koakuma.mtp_prompt.include_kernel_tools}
          onChange={(v) => updateConfig('koakuma.mtp_prompt.include_kernel_tools', v)}
        />
      </div>
    </CategorySection>
  );
};
