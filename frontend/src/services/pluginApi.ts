/**
 * 外部 harness 插件的管理接口：状态、安装与卸载。
 *
 * 插件命令失败不是 HTTP 错误：响应照常返回，failure 说明哪一步没有完成。安装要为
 * 插件的缺省 Agent 补访问记录，经 root API，所以携带身份请求头。
 */
import { accessError } from '@/services/accessApi';
import { identityHeaders } from '@/services/identity';
import { apiUrl } from '@/transports/backendUrl';
import { backendFetch } from '@/transports/connectionStatus';
import type { HarnessId, PluginActionResult, PluginList } from '@/types/plugin';

const BASE = '/api/v1/plugins';

async function json<T>(response: Response, fallback: string): Promise<T> {
  if (!response.ok) throw await accessError(response, fallback);
  return response.json();
}

export async function fetchPlugins(): Promise<PluginList> {
  return json(await backendFetch(apiUrl(BASE)), '读取插件状态失败');
}

/** runPluginCommands 为 false：用户已经按命令手动装好插件，只登记接入、交付凭据。 */
export async function installPlugin(
  harness: HarnessId,
  runPluginCommands = true,
): Promise<PluginActionResult> {
  return json(
    await backendFetch(apiUrl(`${BASE}/${harness}/install`), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...identityHeaders() },
      body: JSON.stringify({ run_plugin_commands: runPluginCommands }),
    }),
    '安装插件失败',
  );
}

export async function uninstallPlugin(harness: HarnessId): Promise<PluginActionResult> {
  return json(
    await backendFetch(apiUrl(`${BASE}/${harness}/uninstall`), { method: 'POST' }),
    '卸载插件失败',
  );
}
