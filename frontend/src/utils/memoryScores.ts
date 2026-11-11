export function formatVitalityScore(score: number): string {
  if (!Number.isFinite(score)) return '0';
  return Math.max(0, Math.min(100, score)).toFixed(0);
}
