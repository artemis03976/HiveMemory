/**
 * Chat 附件上传相关的共享类型
 *
 * 与后端 `server/models/workspace_asset.py` 的上传响应 DTO 对应；
 * opaque ref 只保存在当前运行时的附件状态中，不写入 localStorage，
 * 也不作为本地路径使用。
 */

/** 后端上传响应中的 representation 安全摘要（对应 server/models DTO） */
export interface ApiAttachmentRepresentationSummary {
  representation_id: string;
  kind: string;
  revision: number;
  state: string;
  content_hash: string | null;
  producer: string;
  producer_version: string;
}

/** 前端上传队列项的本地上传状态 */
export type AttachmentUploadStatus = 'queued' | 'uploading' | 'uploaded' | 'failed';

/** 服务端 WorkspaceAsset 聚合状态（HTTP 响应中的 state 字段） */
export type AttachmentAssetState = 'processing' | 'ready' | 'failed';

/** 上传接口的响应摘要 */
export interface AttachmentUploadResult {
  assetRef: string;
  assetId: string;
  displayName: string;
  mediaType: string;
  sizeBytes: number;
  assetState: AttachmentAssetState;
  safeErrorMessage: string | null;
  rawRepresentation: ApiAttachmentRepresentationSummary | null;
}

/** 附件上传队列中的一个文件项（仅存在于当前运行时内存） */
export interface AttachmentQueueItem {
  /** 稳定 operation identity（Idempotency-Key），重试沿用同一取值 */
  id: string;
  /** 原始文件对象，只保存在内存中，不持久化 raw bytes */
  file: File;
  status: AttachmentUploadStatus;
  /** 上传成功后的 opaque ref（服务进程存活期内有效） */
  assetRef: string | null;
  assetId: string | null;
  /** 服务端资产状态；uploaded 项的解析 FAILED 仍属于 uploaded 的服务端状态 */
  assetState: AttachmentAssetState | null;
  /** 上传失败或解析失败时可见的安全文案 */
  errorMessage: string | null;
}
