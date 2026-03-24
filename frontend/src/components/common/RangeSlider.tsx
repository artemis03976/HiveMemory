import { useState } from 'react';

interface RangeSliderProps {
  label: string;
  min: number;
  max: number;
  step: number;
  defaultValue: number;
}

export default function RangeSlider({ label, min, max, step, defaultValue }: RangeSliderProps) {
  const [value, setValue] = useState(defaultValue);
  const percentage = ((value - min) / (max - min)) * 100;

  return (
    <div>
      <div className="flex justify-between text-xs mb-2">
        <span className="text-slate-400">{label}</span>
        <span className="text-primary font-mono bg-primary/10 px-1.5 py-0.5 rounded">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => setValue(Number(e.target.value))}
        className="w-full accent-primary h-1 rounded-lg appearance-none cursor-pointer"
        style={{
          background: `linear-gradient(to right, #c59aff ${percentage}%, rgba(255,255,255,0.1) ${percentage}%)`
        }}
      />
    </div>
  );
}