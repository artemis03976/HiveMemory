import type { TransportStatus } from '@/transports/state';

export interface EventSourceClientOptions<TEvent> {
  url: string;
  lastEventId?: string | null;
  eventName: string;
  parseEvent: (data: string) => TEvent;
  onOpen?: () => void;
  onEvent: (event: TEvent, raw: MessageEvent<string>) => void;
  onStatus?: (status: TransportStatus, error: string | null) => void;
  onError?: (error: string) => void;
}

export class EventSourceClient<TEvent> {
  private eventSource: EventSource | null = null;

  connect(options: EventSourceClientOptions<TEvent>): EventSource {
    this.disconnect();

    const url = new URL(options.url);
    if (options.lastEventId) {
      url.searchParams.set('last_event_id', options.lastEventId);
    }

    const eventSource = new EventSource(url.toString());
    this.eventSource = eventSource;

    eventSource.onopen = () => {
      options.onStatus?.('connected', null);
      options.onOpen?.();
    };

    eventSource.addEventListener(options.eventName, (message) => {
      try {
        options.onEvent(options.parseEvent(message.data), message as MessageEvent<string>);
      } catch (error) {
        console.error('[EventSourceClient] Failed to parse event:', error);
      }
    });

    eventSource.onerror = () => {
      const status = eventSource.readyState === EventSource.CLOSED ? 'disabled' : 'error';
      const message = status === 'disabled' ? 'EventSource stream disabled' : 'EventSource connection error';
      options.onStatus?.(status, message);
      options.onError?.(message);
    };

    return eventSource;
  }

  disconnect(): void {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
  }

  get current(): EventSource | null {
    return this.eventSource;
  }
}
