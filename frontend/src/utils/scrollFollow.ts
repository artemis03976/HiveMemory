/** 离底部在这个距离以内都算停在底部，免得像素级的取整让跟随意外中断。 */
const NEAR_BOTTOM_PX = 48;

/** 列表是否停在底部：停在底部时新内容自动跟随，用户向上翻过之后不再跟随。 */
export function isNearBottom(scrollTop: number, scrollHeight: number, clientHeight: number): boolean {
  return scrollHeight - scrollTop - clientHeight <= NEAR_BOTTOM_PX;
}
