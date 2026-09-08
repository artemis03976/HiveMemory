/**
 * Chat SSE Client Service
 *
 * Handles Server-Sent Events (SSE) connection to the backend chat API.
 * Chat uses POST + request body, so it is backed by the fetch SSE transport.
 */

import { DEFAULT_AGENT_ID } from '@/constants/identity';
import { identityHeaders } from '@/services/identity';
import { FetchSseClient } from '@/transports/sse/fetchSseClient';
import type { ParsedSseEvent } from '@/transports/sse/parseSse';
import type {
  ChatRequestParams,
  SSECallbacks,
  ChatTokenEvent,
  MTPStartEvent,
  MTPResultEvent,
  TopicInfoEvent,
  MemoryRefsEvent,
  ChatDoneEvent,
  ChatErrorEvent,
  SubAgentStartEvent,
  SubAgentEndEvent,
  GenerationIdEvent,
  ChatRunStatusEvent,
  CommandResultEvent,
} from '@/types';

export class ChatSSEClient {
  private client = new FetchSseClient();

  async connect(params: ChatRequestParams, callbacks: SSECallbacks): Promise<void> {
    this.disconnect();

    // Chat 是 Agent action：请求体只携带具体 agent_id；user_id + workspace_id
    // 基础身份选择统一由请求头承载，不在 body 中重复传递。
    const requestBody = {
      message: params.message,
      agent_id: params.agent_id || DEFAULT_AGENT_ID,
      session_id: params.session_id || null,
      enable_memory_retrieval: params.enable_memory_retrieval ?? true,
      generation_options: params.generation_options,
    };

    await this.client.connect(
      '/api/v1/chat',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
          ...identityHeaders(),
        },
        body: JSON.stringify(requestBody),
      },
      {
        onEvent: (event) => this.handleParsedEvent(event, callbacks),
        onError: (error) => callbacks.onConnectionError(error),
      },
    );
  }

  disconnect(): void {
    this.client.disconnect();
  }

  isConnected(): boolean {
    return this.client.isConnected();
  }

  private handleParsedEvent(event: ParsedSseEvent, callbacks: SSECallbacks): void {
    if (!event.data) return;
    try {
      this.handleEvent(event.event, JSON.parse(event.data), callbacks);
    } catch (err) {
      console.error('[ChatSSEClient] Failed to parse SSE data:', err, event.data);
    }
  }

  private handleEvent(eventType: string, data: unknown, callbacks: SSECallbacks): void {
    const isSubScoped = (payload: unknown): boolean => {
      if (!payload || typeof payload !== 'object') return false;
      return (payload as { scope?: string }).scope === 'sub';
    };

    switch (eventType) {
      case 'token': {
        const tokenData = data as ChatTokenEvent;
        if (isSubScoped(tokenData)) {
          callbacks.onSubAgentToken(tokenData);
        } else {
          callbacks.onToken(tokenData);
        }
        break;
      }

      case 'mtp_start': {
        const mtpStartData = data as MTPStartEvent;
        if (isSubScoped(mtpStartData)) {
          callbacks.onSubAgentMTPStart(mtpStartData);
        } else {
          callbacks.onMTPStart(mtpStartData);
        }
        break;
      }

      case 'mtp_result': {
        const mtpResultData = data as MTPResultEvent;
        if (isSubScoped(mtpResultData)) {
          callbacks.onSubAgentMTPResult(mtpResultData);
        } else {
          callbacks.onMTPResult(mtpResultData);
        }
        break;
      }

      case 'topic_info':
        callbacks.onTopicInfo(data as TopicInfoEvent);
        break;

      case 'memory_refs':
        callbacks.onMemoryRefs(data as MemoryRefsEvent);
        break;

      case 'command_result':
        callbacks.onCommandResult(data as CommandResultEvent);
        break;

      case 'done':
        callbacks.onDone(data as ChatDoneEvent);
        break;

      case 'error':
        callbacks.onError(data as ChatErrorEvent);
        break;

      case 'sub_agent_start':
        callbacks.onSubAgentStart(data as SubAgentStartEvent);
        break;

      case 'sub_agent_end':
        callbacks.onSubAgentEnd(data as SubAgentEndEvent);
        break;

      case 'generation_id':
        callbacks.onGenerationId(data as GenerationIdEvent);
        break;

      case 'run_status':
        callbacks.onRunStatus(data as ChatRunStatusEvent);
        break;

      default:
        console.warn('[ChatSSEClient] Unknown SSE event type:', eventType);
    }
  }
}

export async function stopGeneration(generationId: string): Promise<void> {
  try {
    // 取消不是 Agent action：只携带 generation_id 与基础身份选择；
    // 后端通过 generation registry 复用创建时冻结的原始 scope 执行取消。
    await fetch('/api/v1/chat/stop', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...identityHeaders(),
      },
      body: JSON.stringify({ generation_id: generationId }),
    });
  } catch {
    // fire-and-forget: network errors should not block the local stop flow
  }
}
