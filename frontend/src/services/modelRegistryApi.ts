/**
 * Model Registry API Client
 *
 * 封装 /api/v1/models 的 CRUD 请求。与 configApi 不同，注册表变更
 * 即时生效并持久化到后端 configs/models.yaml，不走设置面板的草稿机制。
 */
import { apiErrorMessage } from '@/services/apiError';

import type {
  ModelCreatePayload,
  ModelUpdatePayload,
  RegisteredModel,
} from '@/types/model';

const BASE = '/api/v1/models';

export async function fetchModels(): Promise<RegisteredModel[]> {
  const res = await fetch(BASE);
  if (!res.ok) throw new Error(await apiErrorMessage(res, 'fetchModels failed'));
  return res.json();
}

export async function createModel(payload: ModelCreatePayload): Promise<RegisteredModel> {
  const res = await fetch(BASE, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await apiErrorMessage(res, 'createModel failed'));
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
  if (!res.ok) throw new Error(await apiErrorMessage(res, 'updateModel failed'));
  return res.json();
}

export async function deleteModel(modelId: string): Promise<void> {
  const res = await fetch(`${BASE}/${encodeURIComponent(modelId)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(await apiErrorMessage(res, 'deleteModel failed'));
}
