import type { InputHTMLAttributes } from 'react';

export function Toggle({ checked, onChange, disabled }: { checked: boolean, onChange?: (c: boolean) => void, disabled?: boolean }) {
  return (
    <button 
      disabled={disabled}
      onClick={() => onChange?.(!checked)}
      className={`relative w-11 h-6 rounded-full transition-colors ${checked ? 'bg-primary' : 'bg-ink/10'} ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
    >
      <div className={`absolute top-1 left-1 w-4 h-4 rounded-full bg-white transition-transform ${checked ? 'translate-x-5' : 'translate-x-0'}`} />
    </button>
  );
}

type BaseInputProps = {
  placeholder?: string;
  className?: string;
  step?: number | string;
  disabled?: boolean;
  error?: string;
};

type NumberInputProps = BaseInputProps & {
  type: 'number';
  value?: number | null;
  onChange?: (v: number) => void;
};

type TextInputProps = BaseInputProps & {
  type?: Exclude<InputHTMLAttributes<HTMLInputElement>['type'], 'number'>;
  value?: string | null;
  onChange?: (v: string) => void;
};

type InputProps = NumberInputProps | TextInputProps;

export function Input({ type = 'text', value, onChange, placeholder, className = 'w-48', step, disabled, error }: InputProps) {
  return (
    <div className="flex flex-col gap-1 items-end">
      <input 
        type={type} 
        value={value ?? ''}
        onChange={(e) => {
          if (type === 'number') {
            (onChange as NumberInputProps['onChange'])?.(Number(e.target.value));
            return;
          }
          (onChange as TextInputProps['onChange'])?.(e.target.value);
        }}
        placeholder={placeholder}
        step={step}
        disabled={disabled}
        className={`bg-field border ${error ? 'border-magic-fire/50 focus:ring-magic-fire/50 focus:border-magic-fire/50' : 'border-line focus:ring-primary/50 focus:border-primary/50'} rounded-lg px-3 py-1.5 text-sm text-fg focus:outline-none focus:ring-1 transition-all font-mono ${disabled ? 'opacity-50 cursor-not-allowed' : ''} ${className}`}
      />
      {error && <span className="text-xs text-magic-fire max-w-50 text-right">{error}</span>}
    </div>
  );
}
