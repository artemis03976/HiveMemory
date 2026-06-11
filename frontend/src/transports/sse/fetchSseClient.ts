import { SseStreamParser, type ParsedSseEvent } from './parseSse';

export interface FetchSseClientOptions {
  onEvent: (event: ParsedSseEvent) => void;
  onError?: (error: Error) => void;
}

export class FetchSseClient {
  private abortController: AbortController | null = null;

  async connect(input: RequestInfo | URL, init: RequestInit, options: FetchSseClientOptions): Promise<void> {
    this.disconnect();
    this.abortController = new AbortController();

    try {
      const response = await fetch(input, {
        ...init,
        signal: this.abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      if (!response.body) {
        throw new Error('Response body is null');
      }

      await this.readStream(response.body, options.onEvent);
    } catch (error) {
      if (error instanceof Error && error.name !== 'AbortError') {
        options.onError?.(error);
      }
    }
  }

  disconnect(): void {
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
  }

  isConnected(): boolean {
    return this.abortController !== null;
  }

  private async readStream(body: ReadableStream<Uint8Array>, onEvent: (event: ParsedSseEvent) => void): Promise<void> {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    const parser = new SseStreamParser();

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        for (const event of parser.push(chunk)) {
          onEvent(event);
        }
      }

      for (const event of parser.flush()) {
        onEvent(event);
      }
    } finally {
      reader.releaseLock();
    }
  }
}
