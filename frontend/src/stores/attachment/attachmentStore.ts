/**
 * Attachment Store - 附件上传队列的运行态状态
 *
 * Features:
 * - 逐文件上传队列：每项拥有独立的 queued/uploading/uploaded/failed 状态
 * - 队列项首次入队时生成稳定 operation ID（Idempotency-Key），重试沿用
 * - 上传是独立于 Chat SSE run 的资产命令，可在生成期间继续接收文件
 *
 * 明确不持久化：opaque ref 与 raw bytes 只保存在当前运行时内存中，
 * 不写入 localStorage；页面刷新后 ref 随服务进程内 Store 语义一并失效，
 * 用户需要重新上传。
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { AttachmentApiError, uploadAttachment } from '@/services/attachmentApi';
import type { AttachmentQueueItem } from '@/types/attachment';

interface AttachmentStore {
  items: AttachmentQueueItem[];

  /** 将拖入/选择的文件按顺序加入队列并立即逐项提交上传 */
  enqueue: (files: File[]) => void;
  /** 重试上传失败项（沿用原 operation ID）；仅对本地上传失败项有效 */
  retry: (id: string) => void;
  /** 移除单个队列项；单项失败只清理该项，不影响其他文件 */
  dismiss: (id: string) => void;

  _upload: (id: string) => Promise<void>;
  _patchItem: (id: string, patch: Partial<AttachmentQueueItem>) => void;
}

function newOperationId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `op-${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`;
}

export const useAttachmentStore = create<AttachmentStore>()(
  devtools(
    (set, get) => ({
      items: [],

      enqueue: (files) => {
        if (files.length === 0) return;
        const newItems: AttachmentQueueItem[] = files.map((file) => ({
          id: newOperationId(),
          file,
          status: 'queued',
          assetRef: null,
          assetId: null,
          assetState: null,
          errorMessage: null,
        }));
        set((state) => ({ items: [...state.items, ...newItems] }));
        // 逐项提交；单项失败不影响同一队列中的其他文件
        for (const item of newItems) {
          void get()._upload(item.id);
        }
      },

      retry: (id) => {
        const item = get().items.find((i) => i.id === id);
        // uploading 期间不重复发送；解析 FAILED 属于服务端终态，重试只会重放同一结果
        if (!item || item.status !== 'failed') return;
        void get()._upload(id);
      },

      dismiss: (id) => {
        const item = get().items.find((i) => i.id === id);
        if (!item || item.status === 'uploading') return;
        set((state) => ({ items: state.items.filter((i) => i.id !== id) }));
      },

      _patchItem: (id, patch) => {
        set((state) => ({
          items: state.items.map((item) => (item.id === id ? { ...item, ...patch } : item)),
        }));
      },

      _upload: async (id) => {
        const item = get().items.find((i) => i.id === id);
        if (!item) return;
        get()._patchItem(id, { status: 'uploading', errorMessage: null });
        try {
          const res = await uploadAttachment(item.file, id);
          const parseFailed = res.state === 'failed';
          get()._patchItem(id, {
            status: 'uploaded',
            assetRef: res.asset_ref,
            assetId: res.asset_id,
            assetState: res.state,
            // 解析 FAILED 仍属于 uploaded 项的服务端状态，展示安全摘要
            errorMessage: parseFailed
              ? (res.safe_error?.message ?? '附件解析失败')
              : null,
          });
        } catch (e) {
          const message =
            e instanceof AttachmentApiError ? e.message : '上传失败，请稍后重试';
          get()._patchItem(id, { status: 'failed', errorMessage: message });
        }
      },
    }),
  ),
);
