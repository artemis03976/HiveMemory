export function appendBounded<T>(current: T[], incoming: T[], maxSize: number): T[] {
  const next = [...current, ...incoming];
  return next.length > maxSize ? next.slice(-maxSize) : next;
}
