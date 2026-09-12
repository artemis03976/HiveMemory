import { CheckCircle2, Circle, FileText, Loader2, RotateCcw, X } from 'lucide-react';
import { useAttachmentStore } from '@/stores/attachment';

/** 格式化文件大小展示（B/KB/MB） */
function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function StatusBadge({
  status,
  assetState,
}: {
  status: string;
  assetState: string | null;
}) {
  if (status === 'queued') {
    return <span className="text-[11px] text-slate-500">等待上传</span>;
  }
  if (status === 'uploading') {
    return (
      <span className="flex items-center gap-1 text-[11px] text-slate-400">
        <Loader2 className="w-3 h-3 animate-spin" />
        上传中...
      </span>
    );
  }
  if (status === 'failed') {
    return <span className="text-[11px] text-red-400">上传失败</span>;
  }
  // uploaded：按服务端资产终态展示"可用/解析失败/解析中"
  if (assetState === 'ready') {
    return (
      <span className="flex items-center gap-1 text-[11px] text-emerald-400">
        <CheckCircle2 className="w-3 h-3" />
        可用
      </span>
    );
  }
  if (assetState === 'failed') {
    return <span className="text-[11px] text-red-400">解析失败</span>;
  }
  return <span className="text-[11px] text-slate-500">已上传，解析中...</span>;
}

/**
 * 附件队列条：展示逐文件上传状态、提供重试/移除入口与 Chat 回合选择。
 *
 * 单项失败只影响该项；队列项与 bound ref 只存在于当前运行时内存。
 * 选择集合在发送时以快照冻结进 Chat 请求（计划 9.1 节）。
 */
export default function AttachmentQueue() {
  const items = useAttachmentStore((state) => state.items);
  const selectedIds = useAttachmentStore((state) => state.selectedIds);
  const toggleSelect = useAttachmentStore((state) => state.toggleSelect);
  const retry = useAttachmentStore((state) => state.retry);
  const dismiss = useAttachmentStore((state) => state.dismiss);

  if (items.length === 0) return null;

  return (
    <div className="mb-2 flex flex-col gap-1.5" role="list" aria-label="附件上传队列">
      {items.map((item) => {
        // 只有服务端 required representation READY 的项可进入选择集合
        const selectable = item.status === 'uploaded' && item.assetState === 'ready';
        const selected = selectedIds.includes(item.id);
        return (
          <div
            key={item.id}
            role="listitem"
            className={`flex items-center gap-2.5 px-3 py-2 rounded-xl border text-xs transition-all ${
              selected
                ? 'bg-primary/15 border-primary/50'
                : 'bg-surface-container-high border-white/10'
            }`}
          >
            <FileText className="w-4 h-4 shrink-0 text-slate-400" />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="font-medium text-slate-200 truncate">{item.displayName}</span>
                <span className="text-slate-500 shrink-0">{formatSize(item.file.size)}</span>
              </div>
              <div className="mt-0.5 flex items-center gap-2">
                <StatusBadge status={item.status} assetState={item.assetState} />
                {item.errorMessage && (
                  <span className="text-[11px] text-red-400/80 truncate">{item.errorMessage}</span>
                )}
              </div>
            </div>
            {selectable && (
              <button
                onClick={() => toggleSelect(item.id)}
                className={`p-1.5 rounded-lg transition-all ${
                  selected
                    ? 'text-primary hover:bg-primary/10'
                    : 'text-slate-400 hover:text-white hover:bg-white/5'
                }`}
                title={selected ? '取消选择（不发送该附件）' : '选择该附件随下一条消息发送'}
                aria-label={selected ? `取消选择 ${item.displayName}` : `选择 ${item.displayName}`}
                aria-pressed={selected}
              >
                {selected ? <CheckCircle2 className="w-4 h-4" /> : <Circle className="w-4 h-4" />}
              </button>
            )}
            {item.status === 'failed' && (
              <button
                onClick={() => retry(item.id)}
                className="p-1.5 text-slate-400 hover:text-white hover:bg-white/5 rounded-lg transition-all"
                title="重新上传（沿用同一上传标识）"
                aria-label={`重新上传 ${item.displayName}`}
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            )}
            {item.status !== 'uploading' && (
              <button
                onClick={() => dismiss(item.id)}
                className="p-1.5 text-slate-500 hover:text-white hover:bg-white/5 rounded-lg transition-all"
                title="从队列中移除"
                aria-label={`移除 ${item.displayName}`}
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}
