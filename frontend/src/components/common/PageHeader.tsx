import type { ReactNode } from 'react';

interface PageTitleProps {
  title: string;
  description?: string;
  /** page 用于整页的头部，section 用于侧栏或紧凑的页头。 */
  size?: 'page' | 'section';
}

/** 页面标题：标题加一条短的品牌渐变线，下面一行说明。 */
export function PageTitle({ title, description, size = 'page' }: PageTitleProps) {
  return (
    <div className="min-w-0">
      <h1 className={`flex items-center gap-3 font-manrope font-extrabold text-fg ${size === 'page' ? 'text-[26px]' : 'text-xl'}`}>
        {title}
        <span aria-hidden="true" className={`mt-1.5 h-[3px] rounded-full bg-brand ${size === 'page' ? 'w-10' : 'w-7'}`} />
      </h1>
      {description && <p className="mt-1 text-sm text-fg-2">{description}</p>}
    </div>
  );
}

interface PageHeaderProps {
  title: string;
  description?: string;
  /** 本页的操作，排在右侧。 */
  children?: ReactNode;
}

/** 各页统一的头部：左侧是标题与说明，右侧是本页的操作。 */
export default function PageHeader({ title, description, children }: PageHeaderProps) {
  return (
    <header className="flex shrink-0 flex-wrap items-end gap-4 border-b border-line bg-surface-container-low px-8 py-5">
      <PageTitle title={title} description={description} />
      <div className="flex-1" />
      {children}
    </header>
  );
}
