/**
 * 附件上传队列的共享入口
 *
 * ChatWorkspace 工作区拖拽、composer Paperclip 与附件选择对话框都必须
 * 经过本 hook 的 `enqueueFiles` 进入队列，保证三个入口使用同一份入队
 * 逻辑，避免同一文件被重复入队（计划 A5 节）。
 *
 * 只接收真实文件列表：目录拖拽不会出现在 `DataTransfer.files` 中，
 * 无文件的 drag payload 在 `extractFiles` 中被直接忽略。
 */

import { useCallback } from 'react';
import { useAttachmentStore } from '@/stores/attachment';

/** 从 DataTransfer / FileList / 文件数组中提取真实文件列表 */
export function extractFiles(source: DataTransfer | FileList | File[] | null): File[] {
  if (!source) return [];
  const list = source instanceof DataTransfer ? source.files : source;
  return Array.from(list ?? []);
}

export function useAttachmentUpload() {
  const items = useAttachmentStore((state) => state.items);
  const enqueue = useAttachmentStore((state) => state.enqueue);
  const retry = useAttachmentStore((state) => state.retry);
  const dismiss = useAttachmentStore((state) => state.dismiss);

  const enqueueFiles = useCallback(
    (source: DataTransfer | FileList | File[] | null) => {
      enqueue(extractFiles(source));
    },
    [enqueue],
  );

  return { items, enqueueFiles, retry, dismiss };
}
