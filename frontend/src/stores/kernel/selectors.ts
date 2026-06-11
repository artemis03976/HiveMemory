import type { FilterState, LogEntry } from '@/types/kernel';

export function filterLogs(logs: LogEntry[], filters: FilterState): LogEntry[] {
  return logs.filter((log) => {
    if (filters.logLevel && log.level !== filters.logLevel) {
      return false;
    }

    if (filters.loggerNamespace && !log.logger.startsWith(filters.loggerNamespace)) {
      return false;
    }

    if (filters.searchText) {
      const search = filters.searchText.toLowerCase();
      return log.message.toLowerCase().includes(search) || log.logger.toLowerCase().includes(search);
    }

    return true;
  });
}
