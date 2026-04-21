/** MTP 协议的 5 个指令动词 */
export type MTPVerb = 'SEARCH' | 'READ' | 'RUN' | 'WRITE' | 'UPDATE';

export const ALL_MTP_VERBS: MTPVerb[] = ['SEARCH', 'READ', 'RUN', 'WRITE', 'UPDATE'];
