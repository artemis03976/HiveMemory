/**
 * Attachment API Client
 *
 * Chat 附件上传的 HTTP 客户端：以 `multipart/form-data` 调用
 * `POST /api/v1/workspace/assets`，每个请求只携带一个文件，并附带
 * 统一身份请求头（x-user-id / x-workspace-id）与 `Idempotency-Key`。
 *
 * 注意：不手工设置 multipart 的 Content-Type，由浏览器生成 boundary。
 */

import { identityHeaders } from '@/services/identity';
import type { ApiAttachmentRepresentationSummary } from '@/types/attachment';

interface ApiAttachmentUploadResponse {
  asset_ref: string;
  asset_id: string;
  kind: string;
  display_name: string;
  media_type: string;
  size_bytes: number;
  state: 'processing' | 'ready' | 'failed';
  safe_error: { code: string; message: string } | null;
  required_representation: ApiAttachmentRepresentationSummary | null;
  raw_representation: ApiAttachmentRepresentationSummary | null;
}

/** 上传失败的受控错误：message 为后端返回的安全文案或网络层提示 */
export class AttachmentApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'AttachmentApiError';
    this.status = status;
  }
}

function friendlyStatusMessage(status: number): string {
  switch (status) {
    case 400:
      return '上传请求无效';
    case 409:
      return '同一上传标识携带了不一致的文件';
    case 410:
      return '原上传对应的附件已被移除，请重新上传';
    case 413:
      return '文件超过大小上限';
    case 415:
      return '不支持的附件格式';
    case 503:
      return '附件存储当前不可用，请稍后重试';
    default:
      return `上传失败 (${status})`;
  }
}

/**
 * 上传单个附件文件。
 *
 * @param file 待上传的文件对象（一个请求只发送一个文件）
 * @param operationId 稳定 operation identity，重试必须沿用同一取值
 */
export async function uploadAttachment(
  file: File,
  operationId: string,
): Promise<ApiAttachmentUploadResponse> {
  const form = new FormData();
  form.append('file', file);

  let res: Response;
  try {
    res = await fetch('/api/v1/workspace/assets', {
      method: 'POST',
      headers: {
        ...identityHeaders(),
        'Idempotency-Key': operationId,
      },
      body: form,
    });
  } catch {
    throw new AttachmentApiError('网络异常，上传未完成', 0);
  }

  if (!res.ok) {
    let detail = '';
    try {
      const body = (await res.json()) as { detail?: string };
      detail = typeof body.detail === 'string' ? body.detail : '';
    } catch {
      // 非 JSON 错误响应时回退到状态码文案
    }
    throw new AttachmentApiError(detail || friendlyStatusMessage(res.status), res.status);
  }

  return (await res.json()) as ApiAttachmentUploadResponse;
}
