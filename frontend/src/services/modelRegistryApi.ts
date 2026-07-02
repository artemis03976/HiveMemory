/**
 * Model Registry API Client
 *
 * 封装 /api/v1/models 的 CRUD 请求。与 configApi 不同，注册表变更
 * 即时生效并持久化到后端 configs/models.yaml，不走设置面板的草稿机制。
 */

import type {
  ModelCreatePayload,
  ModelUpdatePayload,
  RegisteredModel,
} from '@/types/model';

const BASE = '/api/v1/models';

/** 从响应中提取后端返回的错误详情（FastAPI 的 detail 字段） */
async function extractError(res: Response, fallback: string): Promise<string> {
  try {
    const body = await res.json();
    if (body && typeof body.detail === 'string') return body.detail;
  } catch {
    // 响应体非 JSON，忽略
  }
  return `${fallback} (${res.status})`;
}

export async function fetchModels(): Promise<RegisteredModel[]> {
  const res = await fetch(BASE);
  if (!res.ok) throw new Error(await extractError(res, 'fetchModels failed'));
  return res.json();
}

export async function createModel(payload: ModelCreatePayload): Promise<RegisteredModel> {
  const res = await fetch(BASE, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await extractError(res, 'createModel failed'));
  return res.json();
}

export async function updateModel(
  modelId: string,
  payload: ModelUpdatePayload,
): Promise<RegisteredModel> {
  const res = await fetch(`${BASE}/${encodeURIComponent(modelId)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await extractError(res, 'updateModel failed'));
  return res.json();
}

export async function deleteModel(modelId: string): Promise<void> {
  const res = await fetch(`${BASE}/${encodeURIComponent(modelId)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(await extractError(res, 'deleteModel failed'));
}
