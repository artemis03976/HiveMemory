import React, { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

interface CategorySectionProps {
  title: string;
  paramCount: number;
  accentColor?: string;
  defaultExpanded?: boolean;
  children: React.ReactNode;
}

export const CategorySection: React.FC<CategorySectionProps> = ({
  title,
  paramCount,
  accentColor = 'hsl(var(--foreground) / 0.8)',
  defaultExpanded = false,
  children,
}) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  return (
    <div className="border-b border-white/5 last:border-b-0">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between px-6 py-4 bg-muted/10 backdrop-blur-lg hover:bg-muted/20 transition-all duration-300"
      >
        <div className="flex items-center gap-3">
          {isExpanded ? (
            <ChevronDown className="w-5 h-5" style={{ color: accentColor }} />
          ) : (
            <ChevronRight className="w-5 h-5" style={{ color: accentColor }} />
          )}
          <h3 className="text-base font-semibold text-foreground">{title}</h3>
        </div>
        <span
          className="px-2 py-0.5 rounded-full text-xs font-medium backdrop-blur-lg"
          style={{
            backgroundColor: `${accentColor}20`,
            borderColor: `${accentColor}30`,
            color: accentColor,
            border: '1px solid',
          }}
        >
          {paramCount} 个参数
        </span>
      </button>
      {isExpanded && (
        <div className="px-6 py-4 space-y-4 bg-background/5 backdrop-blur-sm">
          {children}
        </div>
      )}
    </div>
  );
};
