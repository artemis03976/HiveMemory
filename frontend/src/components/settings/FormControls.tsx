import React from 'react';

interface TextInputProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  error?: string;
  readOnly?: boolean;
  hint?: string;
}

export const TextInput: React.FC<TextInputProps> = ({
  label,
  value,
  onChange,
  placeholder,
  error,
  readOnly,
  hint,
}) => {
  return (
    <div className="space-y-1.5">
      <label className="text-sm font-medium text-foreground">{label}</label>
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        readOnly={readOnly}
        className={`w-full px-3 py-2 glass-input rounded-lg text-sm transition-all duration-300 ${
          error ? 'border-destructive/50' : ''
        } ${readOnly ? 'opacity-60 cursor-not-allowed' : ''}`}
      />
      {error && <p className="text-xs text-destructive mt-1">{error}</p>}
    </div>
  );
};

interface NumberInputProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  error?: string;
  readOnly?: boolean;
  hint?: string;
}

export const NumberInput: React.FC<NumberInputProps> = ({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  error,
  readOnly,
  hint,
}) => {
  return (
    <div className="space-y-1.5">
      <label className="text-sm font-medium text-foreground">{label}</label>
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        min={min}
        max={max}
        step={step}
        readOnly={readOnly}
        className={`w-full px-3 py-2 glass-input rounded-lg text-sm transition-all duration-300 ${
          error ? 'border-destructive/50' : ''
        } ${readOnly ? 'opacity-60 cursor-not-allowed' : ''}`}
      />
      {error && <p className="text-xs text-destructive mt-1">{error}</p>}
    </div>
  );
};

interface PasswordInputProps {
  label: string;
  value: string | null;
  onChange: (value: string | null) => void;
  placeholder?: string;
  error?: string;
  hint?: string;
}

export const PasswordInput: React.FC<PasswordInputProps> = ({
  label,
  value,
  onChange,
  placeholder,
  error,
  hint,
}) => {
  return (
    <div className="space-y-1.5">
      <label className="text-sm font-medium text-foreground">{label}</label>
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
      <input
        type="password"
        value={value || ''}
        onChange={(e) => onChange(e.target.value || null)}
        placeholder={placeholder}
        className={`w-full px-3 py-2 glass-input rounded-lg text-sm transition-all duration-300 ${
          error ? 'border-destructive/50' : ''
        }`}
      />
      {error && <p className="text-xs text-destructive mt-1">{error}</p>}
    </div>
  );
};

interface ToggleSwitchProps {
  label: string;
  value: boolean;
  onChange: (value: boolean) => void;
  hint?: string;
}

export const ToggleSwitch: React.FC<ToggleSwitchProps> = ({
  label,
  value,
  onChange,
  hint,
}) => {
  return (
    <div className="flex items-center justify-between py-2">
      <div className="flex-1">
        <label className="text-sm font-medium text-foreground">{label}</label>
        {hint && <p className="text-xs text-muted-foreground mt-0.5">{hint}</p>}
      </div>
      <button
        onClick={() => onChange(!value)}
        className={`relative w-11 h-6 rounded-full transition-all duration-300 ${
          value
            ? 'bg-primary/30 border-primary/50'
            : 'bg-muted/20 border-white/10'
        } border backdrop-blur-lg`}
      >
        <div
          className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow-lg transition-transform duration-300 ${
            value ? 'translate-x-5' : 'translate-x-0'
          }`}
        />
      </button>
    </div>
  );
};

interface SelectDropdownProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
  error?: string;
  hint?: string;
}

export const SelectDropdown: React.FC<SelectDropdownProps> = ({
  label,
  value,
  onChange,
  options,
  error,
  hint,
}) => {
  return (
    <div className="space-y-1.5">
      <label className="text-sm font-medium text-foreground">{label}</label>
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={`w-full px-3 py-2 glass-input rounded-lg text-sm transition-all duration-300 ${
          error ? 'border-destructive/50' : ''
        }`}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      {error && <p className="text-xs text-destructive mt-1">{error}</p>}
    </div>
  );
};

interface SliderInputProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
  error?: string;
  hint?: string;
}

export const SliderInput: React.FC<SliderInputProps> = ({
  label,
  value,
  onChange,
  min,
  max,
  step,
  error,
  hint,
}) => {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium text-foreground">{label}</label>
        <span className="text-sm text-muted-foreground">{value}</span>
      </div>
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
      <input
        type="range"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        min={min}
        max={max}
        step={step}
        className={`w-full h-2 rounded-lg appearance-none cursor-pointer bg-muted/20 backdrop-blur-lg ${
          error ? 'accent-destructive' : 'accent-primary'
        }`}
      />
      {error && <p className="text-xs text-destructive mt-1">{error}</p>}
    </div>
  );
};

interface TextAreaProps {
  label: string;
  value: string | null;
  onChange: (value: string | null) => void;
  placeholder?: string;
  rows?: number;
  error?: string;
  hint?: string;
}

export const TextArea: React.FC<TextAreaProps> = ({
  label,
  value,
  onChange,
  placeholder,
  rows = 4,
  error,
  hint,
}) => {
  return (
    <div className="space-y-1.5">
      <label className="text-sm font-medium text-foreground">{label}</label>
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
      <textarea
        value={value || ''}
        onChange={(e) => onChange(e.target.value || null)}
        placeholder={placeholder}
        rows={rows}
        className={`w-full px-3 py-2 glass-input rounded-lg text-sm transition-all duration-300 resize-none ${
          error ? 'border-destructive/50' : ''
        }`}
      />
      {error && <p className="text-xs text-destructive mt-1">{error}</p>}
    </div>
  );
};

interface FilePathInputProps {
  label: string;
  value: string | null;
  onChange: (value: string | null) => void;
  placeholder?: string;
  error?: string;
  hint?: string;
}

export const FilePathInput: React.FC<FilePathInputProps> = ({
  label,
  value,
  onChange,
  placeholder,
  error,
  hint,
}) => {
  return (
    <div className="space-y-1.5">
      <label className="text-sm font-medium text-foreground">{label}</label>
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
      <div className="flex gap-2">
        <input
          type="text"
          value={value || ''}
          onChange={(e) => onChange(e.target.value || null)}
          placeholder={placeholder}
          className={`flex-1 px-3 py-2 glass-input rounded-lg text-sm transition-all duration-300 ${
            error ? 'border-destructive/50' : ''
          }`}
        />
        <button
          className="px-3 py-2 bg-muted/20 hover:bg-muted/30 border border-white/10 rounded-lg text-sm transition-all duration-300 backdrop-blur-lg"
          onClick={() => {
            // File picker functionality would be implemented here
            console.log('文件选择器未实现');
          }}
        >
          浏览
        </button>
      </div>
      {error && <p className="text-xs text-destructive mt-1">{error}</p>}
    </div>
  );
};

interface TagInputProps {
  label: string;
  value: string[];
  onChange: (value: string[]) => void;
  placeholder?: string;
  hint?: string;
}

export const TagInput: React.FC<TagInputProps> = ({
  label,
  value,
  onChange,
  placeholder,
  hint,
}) => {
  const [inputValue, setInputValue] = React.useState('');

  const addTag = () => {
    if (inputValue.trim() && !value.includes(inputValue.trim())) {
      onChange([...value, inputValue.trim()]);
      setInputValue('');
    }
  };

  const removeTag = (tag: string) => {
    onChange(value.filter((t) => t !== tag));
  };

  return (
    <div className="space-y-1.5">
      <label className="text-sm font-medium text-foreground">{label}</label>
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
      <div className="flex flex-wrap gap-2 mb-2">
        {value.map((tag) => (
          <span
            key={tag}
            className="inline-flex items-center gap-1 px-2 py-1 bg-primary/20 border border-primary/30 rounded text-xs backdrop-blur-lg"
          >
            {tag}
            <button
              onClick={() => removeTag(tag)}
              className="hover:text-destructive transition-colors"
            >
              ×
            </button>
          </span>
        ))}
      </div>
      <div className="flex gap-2">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault();
              addTag();
            }
          }}
          placeholder={placeholder}
          className="flex-1 px-3 py-2 glass-input rounded-lg text-sm transition-all duration-300"
        />
        <button
          onClick={addTag}
          className="px-3 py-2 bg-primary/20 hover:bg-primary/30 border border-primary/30 rounded-lg text-sm transition-all duration-300 backdrop-blur-lg"
        >
          添加
        </button>
      </div>
    </div>
  );
};
