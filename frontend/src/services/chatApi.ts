/**
 * Chat SSE Client Service
 *
 * Handles Server-Sent Events (SSE) connection to the backend chat API.
 * Chat uses POST + request body, so it is backed by the fetch SSE transport.
 */

import { DEFAULT_USER_ID, DEFAULT_AGENT_ID } from '@/constants/identity';
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

    const requestBody = {
      message: params.message,
      user_id: params.user_id || DEFAULT_USER_ID,
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
    await fetch('/api/v1/chat/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ generation_id: generationId }),
    });
  } catch {
    // fire-and-forget: network errors should not block the local stop flow
  }
}
