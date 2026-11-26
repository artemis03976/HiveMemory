/**
 * Topic API Client
 *
 * Handles HTTP requests to the backend topic management API.
 *
 * 身份约定：Topic 请求统一携带 `user_id + workspace_id`
 * 基础选择（x-user-id / x-workspace-id 请求头），后端不接受 query 中的身份字段。
 */
import { apiErrorMessage } from '@/services/apiError';

import { identityHeaders } from '@/services/identity';
import type { Topic, TopicProgressEntry } from '@/types';
import { apiUrl } from '@/transports/backendUrl';
import { backendFetch } from '@/transports/connectionStatus';

export interface ApiTopicSnapshot {
  topic_id: string;
  topic_title?: string;
  topic_summary?: string;
  fold_summary?: string;
  last_turn?: { user: string; assistant: string } | null;
  block_count?: number;
  /** 话题最近一次写入的 Unix 时间戳（秒）。 */
  last_update: number;
  total_tokens?: number;
  model_used?: string;
  /** 是否有还没有沉淀为长期记忆的交互。 */
  has_unsettled_interactions: boolean;
  /** 进展记录，按写下的顺序。 */
  progress_record: ApiProgressEntry[];
  /** 进展记录还没有覆盖到的交互轮数。 */
  unrecorded_turn_count: number;
  /** 还没有整理的交互中最近至多 3 轮的节选。 */
  unrecorded_turns: { turn_number: number; user: string; assistant: string }[];
}

interface ApiProgressEntry {
  text: string;
  /** 这一条覆盖到话题的第几轮（从 1 数起）。 */
  covered_until: number;
  /** 这一条覆盖的轮数。 */
  turn_count: number;
  /** 写下这一条的 Unix 时间戳（秒）。 */
  written_at: number;
}

interface ActiveTopicListResponse {
  topics: ApiTopicSnapshot[];
}

function mapProgressEntry(raw: ApiProgressEntry): TopicProgressEntry {
  return {
    text: raw.text,
    fromTurn: raw.covered_until - raw.turn_count + 1,
    toTurn: raw.covered_until,
    writtenAt: raw.written_at * 1000,
  };
}

export function mapTopic(raw: ApiTopicSnapshot): Topic {
  return {
    id: raw.topic_id,
    title: raw.topic_title ?? '',
    topicSummary: raw.topic_summary || undefined,
    foldSummary: raw.fold_summary || undefined,
    lastTurn: raw.last_turn ?? undefined,
    model: raw.model_used || undefined,
    lastActive: raw.last_update * 1000,
    blockCount: raw.block_count ?? 0,
    totalTokens: raw.total_tokens ?? 0,
    hasUnsettledInteractions: raw.has_unsettled_interactions,
    progressRecord: raw.progress_record.map(mapProgressEntry),
    unrecordedTurnCount: raw.unrecorded_turn_count,
    unrecordedTurns: raw.unrecorded_turns.map((turn) => ({
      turnNumber: turn.turn_number,
      user: turn.user,
      assistant: turn.assistant,
    })),
  };
}

export async function fetchTopics(): Promise<Topic[]> {
  const res = await backendFetch(apiUrl('/api/v1/topics'), {
    headers: identityHeaders(),
  });
  if (!res.ok) throw new Error(await apiErrorMessage(res, 'fetchTopics failed'));
  const data: ActiveTopicListResponse = await res.json();
  return data.topics.map(mapTopic);
}

export async function settleTopic(topicId: string): Promise<void> {
  const res = await backendFetch(apiUrl(`/api/v1/topics/${encodeURIComponent(topicId)}/settle`), {
    method: 'POST',
    headers: identityHeaders(),
  });
  if (!res.ok) throw new Error(await apiErrorMessage(res, 'settleTopic failed'));
}

export async function deleteTopic(topicId: string): Promise<void> {
  const res = await backendFetch(apiUrl(`/api/v1/topics/${encodeURIComponent(topicId)}`), {
    method: 'DELETE',
    headers: identityHeaders(),
  });
  if (!res.ok) throw new Error(await apiErrorMessage(res, 'deleteTopic failed'));
}
