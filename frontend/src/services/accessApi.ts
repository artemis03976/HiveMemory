/**
 * 接入登记与 Workspace 访问记录的管理接口。
 *
 * 接入登记属于 System，与提供商、模型接口一样不携带身份请求头；访问记录经 root API，
 * 以统一的身份请求头进入当前 Workspace。
 */
import { readApiError, type ApiError } from '@/services/apiError';
import { identityHeaders } from '@/services/identity';
import { apiUrl } from '@/transports/backendUrl';
import type {
  AccessPrincipal,
  ActorRecord,
  ActorRecordPayload,
  PrincipalCreatePayload,
  PrincipalCredential,
  PrincipalSettingsPayload,
} from '@/types/access';

const BASE = '/api/v1/access';

/** 登记被拒绝的原因换成页面上能做的事；后端的说明对所有原因相同。 */
const CONFLICT_MESSAGES: Record<string, string> = {
  protected: '这条登记关系到管理接口、对话或导入本身，只能经登记文件修改',
  duplicate: '已经有同名的接入来源',
  credential_from_env: '凭据来自环境变量，请修改 .env 后重启后端',
  file_modified: '登记文件在运行中被手工修改过，请重启后端装载这些修改后再编辑',
};

async function accessError(response: Response, fallback: string): Promise<ApiError> {
  const error = await readApiError(response, fallback);
  const message = error.code === 'access.registration.conflict' && error.reason
    ? CONFLICT_MESSAGES[error.reason]
    : undefined;
  if (message) error.message = message;
  return error;
}

async function json<T>(response: Response, fallback: string): Promise<T> {
  if (!response.ok) throw await accessError(response, fallback);
  return response.json();
}

async function noContent(response: Response, fallback: string): Promise<void> {
  if (!response.ok) throw await accessError(response, fallback);
}

const principalUrl = (principalId: string) =>
  apiUrl(`${BASE}/principals/${encodeURIComponent(principalId)}`);

export async function fetchPrincipals(): Promise<AccessPrincipal[]> {
  return json(await fetch(apiUrl(`${BASE}/principals`)), '读取接入登记失败');
}

export async function registerPrincipal(payload: PrincipalCreatePayload): Promise<PrincipalCredential> {
  return json(
    await fetch(apiUrl(`${BASE}/principals`), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
    '登记接入来源失败',
  );
}

export async function updatePrincipal(
  principalId: string,
  payload: PrincipalSettingsPayload,
): Promise<AccessPrincipal> {
  return json(
    await fetch(principalUrl(principalId), {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
    '修改接入来源失败',
  );
}

export async function rotatePrincipalCredential(principalId: string): Promise<PrincipalCredential> {
  return json(
    await fetch(`${principalUrl(principalId)}/credential`, { method: 'POST' }),
    '轮换凭据失败',
  );
}

export async function deletePrincipal(principalId: string): Promise<void> {
  return noContent(await fetch(principalUrl(principalId), { method: 'DELETE' }), '删除接入来源失败');
}

/** 用户级记录与精确记录各有路径：用户级记录没有 agent_id。 */
const recordUrl = (agentId: string | null) =>
  agentId === null
    ? apiUrl(`${BASE}/actors/user`)
    : apiUrl(`${BASE}/actors/agents/${encodeURIComponent(agentId)}`);

export async function fetchActorRecords(): Promise<ActorRecord[]> {
  return json(
    await fetch(apiUrl(`${BASE}/actors`), { headers: identityHeaders() }),
    '读取访问记录失败',
  );
}

export async function putActorRecord(
  agentId: string | null,
  payload: ActorRecordPayload,
): Promise<ActorRecord> {
  return json(
    await fetch(recordUrl(agentId), {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json', ...identityHeaders() },
      body: JSON.stringify(payload),
    }),
    '保存访问记录失败',
  );
}

export async function deleteActorRecord(agentId: string | null): Promise<void> {
  return noContent(
    await fetch(recordUrl(agentId), { method: 'DELETE', headers: identityHeaders() }),
    '删除访问记录失败',
  );
}
