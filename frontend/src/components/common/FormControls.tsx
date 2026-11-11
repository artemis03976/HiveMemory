import type { InputHTMLAttributes } from 'react';

export function SettingSection({ title, children }: { title: string, children: React.ReactNode }) {
  return (
    <section className="mb-8">
      <h2 className="text-lg font-bold text-fg mb-4 flex items-center gap-2">{title}</h2>
      <div className="bg-surface-container-low border border-line rounded-2xl overflow-hidden border border-line">
        {children}
      </div>
    </section>
  );
}

export function SettingRow({ label, description, children }: { label: string, description?: string, children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between p-5 border-b border-line last:border-0 hover:bg-ink/2 transition-colors">
      <div className="pr-8">
        <div className="text-sm font-medium text-fg">{label}</div>
        {description && <div className="text-xs text-fg-3 mt-1 leading-relaxed">{description}</div>}
      </div>
      <div className="shrink-0">
        {children}
      </div>
    </div>
  );
}

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

/** 允许留空的数字输入：清空时回传 null，由调用方解释为“不覆盖、沿用默认值”。 */
export function NullableNumberInput({
  value,
  onChange,
  placeholder,
  className = 'w-48',
  step,
  error,
}: Omit<BaseInputProps, 'disabled'> & {
  value: number | null;
  onChange: (v: number | null) => void;
}) {
  return (
    <div className="flex flex-col gap-1 items-end">
      <input
        type="number"
        value={value ?? ''}
        onChange={(e) => onChange(e.target.value === '' ? null : Number(e.target.value))}
        placeholder={placeholder}
        step={step}
        className={`bg-field border ${error ? 'border-magic-fire/50 focus:ring-magic-fire/50 focus:border-magic-fire/50' : 'border-line focus:ring-primary/50 focus:border-primary/50'} rounded-lg px-3 py-1.5 text-sm text-fg focus:outline-none focus:ring-1 transition-all font-mono ${className}`}
      />
      {error && <span className="text-xs text-magic-fire max-w-50 text-right">{error}</span>}
    </div>
  );
}

export function Select({ options, value, onChange, error }: { options: {label: string, value: string}[], value?: string, onChange?: (v: string) => void, error?: string }) {
  return (
    <div className="flex flex-col gap-1 items-end">
      <select 
        value={value}
        onChange={(e) => onChange?.(e.target.value)}
        className={`bg-field border ${error ? 'border-magic-fire/50 focus:ring-magic-fire/50 focus:border-magic-fire/50' : 'border-line focus:ring-primary/50 focus:border-primary/50'} rounded-lg px-3 py-1.5 text-sm text-fg focus:outline-none focus:ring-1 transition-all cursor-pointer font-mono outline-none`}
      >
        {options.map(o => <option key={o.value} value={o.value} className="bg-surface-container">{o.label}</option>)}
      </select>
      {error && <span className="text-xs text-magic-fire max-w-50 text-right">{error}</span>}
    </div>
  );
}
