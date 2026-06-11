import type { LogEntry, TraceGroup } from '@/types/kernel';

export function appendBounded<T>(current: T[], incoming: T[], maxSize: number): T[] {
  const next = [...current, ...incoming];
  return next.length > maxSize ? next.slice(-maxSize) : next;
}

export function groupLogsByTrace(logs: LogEntry[]): Map<string, TraceGroup> {
  const groups = new Map<string, TraceGroup>();

  for (const log of logs) {
    const { trace_id, span_name, task_type } = log;

    if (!groups.has(trace_id)) {
      groups.set(trace_id, {
        trace_id,
        spans: new Map(),
        task_type,
        collapsed: task_type === 'background',
      });
    }

    const trace = groups.get(trace_id)!;
    if (!trace.spans.has(span_name)) {
      trace.spans.set(span_name, {
        span_name,
        logs: [],
        collapsed: false,
        task_type,
        first_timestamp: log.timestamp,
        last_timestamp: log.timestamp,
      });
    }

    const span = trace.spans.get(span_name)!;
    span.logs.push(log);
    span.last_timestamp = log.timestamp;
  }

  return groups;
}

export function toggleTraceCollapseInGroups(
  groups: Map<string, TraceGroup>,
  traceId: string,
): Map<string, TraceGroup> {
  const nextGroups = new Map(groups);
  const trace = nextGroups.get(traceId);
  if (trace) {
    nextGroups.set(traceId, { ...trace, collapsed: !trace.collapsed });
  }
  return nextGroups;
}

export function toggleSpanCollapseInGroups(
  groups: Map<string, TraceGroup>,
  traceId: string,
  spanName: string,
): Map<string, TraceGroup> {
  const nextGroups = new Map(groups);
  const trace = nextGroups.get(traceId);
  const span = trace?.spans.get(spanName);
  if (trace && span) {
    const nextSpans = new Map(trace.spans);
    nextSpans.set(spanName, { ...span, collapsed: !span.collapsed });
    nextGroups.set(traceId, { ...trace, spans: nextSpans });
  }
  return nextGroups;
}
