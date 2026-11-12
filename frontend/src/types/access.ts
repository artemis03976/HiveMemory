/** 外部入口的 adapter：运行时登记的来源只能列出它们 */
export type ExternalAdapter = 'mcp' | 'import';

/** 六项 actor 操作，与访问记录白名单中的取值一致 */
export type WorkspaceOperationName =
  | 'memory.read'
  | 'memory.search'
  | 'memory.run'
  | 'memory.write'
  | 'memory.update'
  | 'agent.call';

/** 一条接入登记；进程内来源 editable 为 false，只能经登记文件修改 */
export interface AccessPrincipal {
  principal_id: string;
  kind: string;
  enabled: boolean;
  adapters: string[];
  /** 可声明的 Agent；null 表示不限定 */
  allowed_agent_ids: string[] | null;
  allow_system_actor: boolean;
  editable: boolean;
  /** 生效凭据的来源；外部来源为 null 时没有可用凭据，认证一律被拒绝 */
  credential_source: 'env' | 'generated' | null;
  credential_env: string | null;
  credential_masked: string | null;
}

/** 外部来源可以修改的内容 */
export interface PrincipalSettingsPayload {
  adapters: ExternalAdapter[];
  allowed_agent_ids: string[] | null;
  enabled: boolean;
}

/** 一条 Workspace 访问记录；agent_id 为 null 表示用户级记录 */
export interface ActorRecord {
  agent_id: string | null;
  enabled: boolean;
  allowed_operations: WorkspaceOperationName[];
}

export interface ActorRecordPayload {
  enabled: boolean;
  allowed_operations: WorkspaceOperationName[];
}
