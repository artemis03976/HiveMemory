import type { SessionCompactionCheckpoint } from '@/services/sessionApi';

/** 标记模型视图的覆盖范围；原始对话保持可见，摘要和引用按纯文本展示。 */
export default function SessionCompactionBoundary({ interactionId, checkpoint }: {
  interactionId?: string;
  checkpoint: SessionCompactionCheckpoint | null;
}) {
  if (!interactionId || !checkpoint) return null;
  const summarized = checkpoint.summarized_through === interactionId;
  const reduced = checkpoint.reduced_through === interactionId;
  if (!summarized && !reduced) return null;
  return <div className="border-t border-primary/20 pt-3 space-y-2 text-xs text-slate-400">
    {summarized && <details>
      <summary className="cursor-pointer text-primary/80">以上对话已纳入摘要 · 查看摘要（原文保留）</summary>
      <p className="mt-3 whitespace-pre-wrap break-words">{checkpoint.summary}</p>
      {checkpoint.references.length > 0 && <div className="mt-3">
        <p>引用线索（不代表写入已完成）</p>
        <ul className="list-disc pl-5 break-words">
          {checkpoint.references.map(ref => <li key={ref}>{ref}</li>)}
        </ul>
      </div>}
    </details>}
    {reduced && <p>以上轮次的工具结果已精简 · 原文保留</p>}
  </div>;
}
