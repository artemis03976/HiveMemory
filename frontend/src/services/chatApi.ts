/**
 * Chat SSE Client Service
 *
 * Handles Server-Sent Events (SSE) connection to the backend chat API.
 * Uses fetch + ReadableStream for manual SSE parsing with better control.
 */

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
} from '@/types';

export class ChatSSEClient {
  private abortController: AbortController | null = null;

  /**
   * Connect to the chat SSE endpoint and stream responses
   */
  async connect(params: ChatRequestParams, callbacks: SSECallbacks): Promise<void> {
    // Cleanup existing connection
    this.disconnect();

    try {
      // Create abort controller for cleanup
      this.abortController = new AbortController();

      // Make POST request to initiate SSE stream
      const response = await fetch('/api/v1/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify({
          message: params.message,
          user_id: params.user_id || 'default',
          agent_id: params.agent_id || 'default',
          session_id: params.session_id || null,
          enable_memory_retrieval: params.enable_memory_retrieval ?? true,
          generation_options: params.generation_options,
        }),
        signal: this.abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      if (!response.body) {
        throw new Error('Response body is null');
      }

      // Parse SSE stream manually
      await this.parseSSEStream(response.body, callbacks);
    } catch (error) {
      // Only report non-abort errors
      if (error instanceof Error && error.name !== 'AbortError') {
        callbacks.onConnectionError(error);
      }
    }
  }

  /**
   * Parse SSE stream from ReadableStream
   */
  private async parseSSEStream(
    body: ReadableStream<Uint8Array>,
    callbacks: SSECallbacks
  ): Promise<void> {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let currentEvent = '';

    try {
      while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        // Decode chunk and add to buffer
        buffer += decoder.decode(value, { stream: true });

        // Process complete lines
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
          // Parse SSE format: "event: <type>" and "data: <json>"
          if (line.startsWith('event:')) {
            currentEvent = line.substring(6).trim();
          } else if (line.startsWith('data:')) {
            const dataStr = line.substring(5).trim();

            if (dataStr) {
              try {
                const data = JSON.parse(dataStr);
                this.handleEvent(currentEvent, data, callbacks);
              } catch (err) {
                console.error('[ChatSSEClient] Failed to parse SSE data:', err, dataStr);
              }
            }

            // Reset event type after processing
            currentEvent = '';
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  /**
   * Route SSE events to appropriate callbacks
   */
  private handleEvent(
    eventType: string,
    data: unknown,
    callbacks: SSECallbacks
  ): void {
    switch (eventType) {
      case 'token':
        callbacks.onToken(data as ChatTokenEvent);
        break;

      case 'mtp_start':
        callbacks.onMTPStart(data as MTPStartEvent);
        break;

      case 'mtp_result':
        callbacks.onMTPResult(data as MTPResultEvent);
        break;

      case 'topic_info':
        callbacks.onTopicInfo(data as TopicInfoEvent);
        break;

      case 'memory_refs':
        callbacks.onMemoryRefs(data as MemoryRefsEvent);
        break;

      case 'done':
        callbacks.onDone(data as ChatDoneEvent);
        break;

      case 'error':
        callbacks.onError(data as ChatErrorEvent);
        break;

      default:
        console.warn('[ChatSSEClient] Unknown SSE event type:', eventType);
    }
  }

  /**
   * Disconnect and cleanup
   */
  disconnect(): void {
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
  }

  /**
   * Check if connection is active
   */
  isConnected(): boolean {
    return this.abortController !== null;
  }
}
