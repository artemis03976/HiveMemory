import { AlertTriangle } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ConfidenceIndicatorProps {
  score: number;
  showLabel?: boolean;
}

export function ConfidenceIndicator({ score, showLabel = false }: ConfidenceIndicatorProps) {
  const percentage = Math.round(score * 100);
  const isLow = score < 0.5;

  return (
    <div className="flex items-center gap-2">
      {/* Progress bar */}
      <div className="flex-1 h-1.5 bg-muted/30 rounded-full overflow-hidden min-w-[60px]">
        <div
          className={cn(
            'h-full transition-all duration-300 rounded-full',
            isLow ? 'bg-destructive' : 'bg-primary'
          )}
          style={{ width: `${percentage}%` }}
        />
      </div>

      {/* Percentage label */}
      {showLabel && (
        <span
          className={cn(
            'text-xs font-medium',
            isLow ? 'text-destructive' : 'text-muted-foreground'
          )}
        >
          {percentage}%
        </span>
      )}

      {/* Warning icon for low confidence */}
      {isLow && (
        <AlertTriangle className="w-3 h-3 text-destructive" aria-label="Low confidence score" />
      )}
    </div>
  );
}
