/** 支持的外部 harness，与后端插件服务的 harness 表一致 */
export type HarnessId = 'claude-code' | 'codex' | 'pi' | 'dsh';

/** 一个 harness 的插件状态；“已安装”以 HiveMemory 一侧的接入登记为准 */
export interface PluginStatus {
  harness: HarnessId;
  name: string;
  principal_id: string;
  /** 没有设置 HIVEMEMORY_AGENT_ID 时插件声明的 Agent */
  default_agent_id: string;
  /** copy：harness 复制插件，升级后要重新安装；link：链接插件目录、原地加载 */
  install_mode: 'copy' | 'link';
  /** 能否在这里代为安装：插件目录存在，并且运行在 Windows 上 */
  available: boolean;
  /** 找到的 harness 命令；null 表示没有找到 */
  command: string | null;
  installed: boolean;
  needs_update: boolean;
  /** 来源可声明的 Agent；未安装或不限定时为 null */
  allowed_agent_ids: string[] | null;
  credential_env: string;
  manual_commands: string[];
}

export interface PluginList {
  integrations_dir: string;
  version: string;
  harnesses: PluginStatus[];
}

export type PluginFailureReason =
  | 'unavailable'
  | 'harness_missing'
  | 'settings_unreadable'
  | 'command_failed'
  | 'command_timeout'
  | 'unexpected_output';

export interface PluginFailure {
  reason: PluginFailureReason;
  command: string | null;
  output_tail: string[];
}

/** 安装或卸载之后的状态；failure 不为 null 时说明哪一步没有完成 */
export interface PluginActionResult {
  status: PluginStatus;
  failure: PluginFailure | null;
}
