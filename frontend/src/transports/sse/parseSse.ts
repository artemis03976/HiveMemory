export interface ParsedSseEvent {
  id?: string;
  event: string;
  data: string;
}

interface SseDraft {
  id?: string;
  event?: string;
  dataLines: string[];
}

export class SseStreamParser {
  private buffer = '';
  private draft: SseDraft = { dataLines: [] };

  push(chunk: string): ParsedSseEvent[] {
    this.buffer += chunk;
    const lines = this.buffer.split(/\r?\n/);
    this.buffer = lines.pop() ?? '';
    return this.processLines(lines);
  }

  flush(): ParsedSseEvent[] {
    const events = this.processLines(this.buffer ? [this.buffer, ''] : ['']);
    this.buffer = '';
    return events;
  }

  private processLines(lines: string[]): ParsedSseEvent[] {
    const events: ParsedSseEvent[] = [];

    for (const line of lines) {
      if (line === '') {
        const event = this.commitDraft();
        if (event) events.push(event);
        continue;
      }

      if (line.startsWith(':')) continue;

      const separatorIndex = line.indexOf(':');
      const field = separatorIndex === -1 ? line : line.slice(0, separatorIndex);
      const rawValue = separatorIndex === -1 ? '' : line.slice(separatorIndex + 1);
      const value = rawValue.startsWith(' ') ? rawValue.slice(1) : rawValue;

      switch (field) {
        case 'id':
          this.draft.id = value;
          break;
        case 'event':
          this.draft.event = value;
          break;
        case 'data':
          this.draft.dataLines.push(value);
          break;
      }
    }

    return events;
  }

  private commitDraft(): ParsedSseEvent | null {
    if (!this.draft.event && this.draft.dataLines.length === 0 && !this.draft.id) {
      return null;
    }

    const event: ParsedSseEvent = {
      id: this.draft.id,
      event: this.draft.event || 'message',
      data: this.draft.dataLines.join('\n'),
    };

    this.draft = { dataLines: [] };
    return event;
  }
}

export function parseSseText(text: string): ParsedSseEvent[] {
  const parser = new SseStreamParser();
  return [...parser.push(text), ...parser.flush()];
}
