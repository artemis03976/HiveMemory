import { useState } from 'react';

interface RangeSliderProps {
  label: string;
  min: number;
  max: number;
  step: number;
  defaultValue: number;
  value?: number;
  onChange?: (value: number) => void;
}

export default function RangeSlider({ label, min, max, step, defaultValue, value, onChange }: RangeSliderProps) {
  const [internalValue, setInternalValue] = useState(defaultValue);
  const currentValue = value ?? internalValue;
  const percentage = ((currentValue - min) / (max - min)) * 100;

  return (
    <div>
      <div className="flex justify-between text-xs mb-2">
        <span className="text-slate-400">{label}</span>
        <span className="text-primary font-mono bg-primary/10 px-1.5 py-0.5 rounded">{currentValue}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={currentValue}
        onChange={(e) => {
          const nextValue = Number(e.target.value);
          if (value === undefined) {
            setInternalValue(nextValue);
          }
          onChange?.(nextValue);
        }}
        className="w-full accent-primary h-1 rounded-lg appearance-none cursor-pointer"
        style={{
          background: `linear-gradient(to right, #c59aff ${percentage}%, rgba(255,255,255,0.1) ${percentage}%)`
        }}
      />
    </div>
  );
}
