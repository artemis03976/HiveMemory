import { useSyncExternalStore } from 'react';
import { AlertTriangle, WifiOff } from 'lucide-react';
import { getConnectionState, subscribeConnection } from '@/transports/connectionStatus';

/** 主区域顶部的连接状态条：连不上后端优先于凭据失效，问题消失后状态条随之消失。 */
export default function ConnectionBanner() {
  const { unreachable, credentialInvalid } = useSyncExternalStore(subscribeConnection, getConnectionState);

  if (unreachable) {
    return (
      <div role="status" className="flex shrink-0 items-center gap-2.5 border-b border-line bg-magic-fire/10 px-6 py-2.5 text-[13px] text-fg">
        <WifiOff className="h-4 w-4 shrink-0 text-magic-fire" aria-hidden="true" />
        <span>
          <b className="font-semibold">连不上 HiveMemory 后端。</b>
          请确认后端正在运行；恢复连接后，这条提示会自动消失。
        </span>
      </div>
    );
  }

  if (credentialInvalid) {
    return (
      <div role="status" className="flex shrink-0 items-center gap-2.5 border-b border-line bg-magic-metal/10 px-6 py-2.5 text-[13px] text-fg">
        <AlertTriangle className="h-4 w-4 shrink-0 text-magic-metal" aria-hidden="true" />
        <span className="flex-1">
          <b className="font-semibold">管理凭据已失效。</b>
          后端换了凭据，刷新页面即可取得新的凭据。
        </span>
        <button
          onClick={() => window.location.reload()}
          className="h-7 shrink-0 rounded-lg border border-line-strong bg-surface-container px-2.5 text-xs font-semibold text-fg hover:bg-surface-container-high"
        >
          刷新页面
        </button>
      </div>
    );
  }

  return null;
}
