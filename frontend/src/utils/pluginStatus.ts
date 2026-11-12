import type { PluginFailure, PluginStatus } from '@/types/plugin';

/** 插件卡片的状态：由后端给出的状态推出，页面按它决定显示什么、能做什么。 */
export type PluginCardState = 'installed' | 'needs_update' | 'not_installed' | 'missing' | 'unavailable';

export function pluginCardState(status: PluginStatus): PluginCardState {
  if (status.installed) return status.needs_update ? 'needs_update' : 'installed';
  if (!status.available) return 'unavailable';
  return status.command === null ? 'missing' : 'not_installed';
}

/**
 * 没有完成的那一步，换成用户能读懂、能处理的说明。
 *
 * 安装失败时什么都没有登记；卸载时接入总会删除，没完成的只是 harness 一侧的插件。
 */
export function pluginFailureMessage(
  failure: PluginFailure,
  status: PluginStatus,
  action: 'installed' | 'removed',
): string {
  const dshHint = status.harness === 'dsh' ? 'dsh 桌面端正在运行时不能改动插件，先从托盘退出 dsh。' : '';
  if (action === 'removed') {
    if (failure.reason === 'harness_missing') {
      return `没有找到 ${status.name}，插件可能还留在里面；它已经不能再访问记忆。`;
    }
    return `${status.name} 的卸载命令没有完成，插件可能还留在里面；它已经不能再访问记忆。${dshHint}`;
  }
  switch (failure.reason) {
    case 'unavailable':
      return '这里不能代为安装：只有 Windows 上的安装版代为安装插件，开发方式、Docker 或其他系统上运行的 HiveMemory 不代为安装。';
    case 'harness_missing':
      return `没有找到 ${status.name}。装好它之后点“重新检查”，或者按下面的命令手动安装。`;
    case 'command_timeout':
      return '插件命令超过时限没有完成。可以重试，或者按下面的命令手动安装。';
    case 'unexpected_output':
      return `${status.name} 的命令输出不是预期的格式，可能是它的版本变了。可以按下面的命令手动安装。`;
    case 'command_failed':
      return `${status.name} 的插件命令执行失败。${dshHint}可以重试，或者按下面的命令手动安装。`;
  }
}
