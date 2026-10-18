/** 会话管理客户端：正文只来自后端历史，浏览器只保存当前会话选择。 */
import { apiErrorMessage } from '@/services/apiError';
import { identityHeaders } from '@/services/identity';

export type TurnOutcome = 'completed' | 'cancelled' | 'failed';
export interface SessionSummary {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  turn_count: number;
  active_process_id: string | null;
}
export interface SessionTurnEvent {
  kind: string;
  sequence: number;
  content: string;
  action_id?: string | null;
  tool_kind?: string | null;
  target?: string | null;
  tool_args?: Record<string, unknown> | null;
  status?: string | null;
}
export interface ConversationTurn {
  interaction_id: string;
  agent_id: string;
  topic_id: string;
  created_at: string;
  payload: {
    outcome: TurnOutcome;
    user_message: string;
    assistant_final_text: string | null;
    turn_events: SessionTurnEvent[];
  };
}
export interface ConversationSession extends SessionSummary {
  turns: ConversationTurn[];
  compaction_checkpoint?: SessionCompactionCheckpoint | null;
}
/** 后端接受的历史视图；游标只标记原文范围，不替换浏览器展示的历史。 */
export interface SessionCompactionCheckpoint {
  producer: string;
  format_version: number;
  reduced_through: string | null;
  summarized_through: string | null;
  summary: string;
  references: string[];
  created_at: string;
}
export class SessionApiError extends Error {
  status: number;
  constructor(status: number, message: string) { super(message); this.status = status; }
}
async function request<T>(path = '', method = 'GET', title?: string): Promise<T> {
  const response = await fetch(`/api/v1/sessions${path}`, {
    method,
    headers: { ...identityHeaders(), 'Content-Type': 'application/json' },
    ...(title !== undefined ? { body: JSON.stringify({ title }) } : {}),
  });
  if (!response.ok) {
    throw new SessionApiError(response.status, await apiErrorMessage(response, '会话操作失败'));
  }
  return response.status === 204 ? undefined as T : response.json();
}
export const listSessions = () => request<SessionSummary[]>();
export const getSession = (id: string) => request<ConversationSession>(`/${encodeURIComponent(id)}`);
export const createSession = () => request<ConversationSession>('', 'POST', '');
export const renameSession = (id: string, title: string) => request<ConversationSession>(`/${encodeURIComponent(id)}`, 'PATCH', title);
export const deleteSession = (id: string) => request<void>(`/${encodeURIComponent(id)}`, 'DELETE');
