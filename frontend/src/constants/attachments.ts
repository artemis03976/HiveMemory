/**
 * Chat 附件上传的共享常量
 *
 * 与后端 `system/services/attachments/formats.py` 冻结的批准格式集合
 * 保持一致（计划 7.1 节：前端 accept、上传白名单与 parser 分派共用
 * 同一格式集合）；两侧修改时必须同步。客户端不做本地格式校验，格式
 * 与大小以后端稳定错误为准。
 */

/** 批准的附件扩展名（小写，含点），用于 file input 的 accept 提示 */
export const ACCEPTED_ATTACHMENT_EXTENSIONS = ['.txt', '.md', '.markdown', '.docx'] as const;

/** 传递给 <input type="file" accept> 的属性值 */
export const ATTACHMENT_ACCEPT_ATTR = ACCEPTED_ATTACHMENT_EXTENSIONS.join(',');

/** 单个附件的默认大小上限（10 MiB，与后端 AttachmentsConfig 默认值一致） */
export const MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024;
