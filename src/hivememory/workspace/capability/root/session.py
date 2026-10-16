"""root 会话管理 API：先授权，再访问目标 Workspace 的会话资源。"""

from hivememory.core.access import WorkspaceAccessContext
from hivememory.core.models import WorkspaceIdentity
from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
from hivememory.workspace.sessions import (
    ConversationSession,
    ConversationSessionService,
    SessionSummary,
)


class RootSessionService:
    """会话管理只供 system，不新增 actor operation。"""

    def __init__(
        self,
        sessions: ConversationSessionService,
        *,
        operation_authorizer: WorkspaceOperationAuthorizer,
    ) -> None:
        self._sessions = sessions
        self._authorizer = operation_authorizer

    def create_session(
        self,
        *,
        target_workspace: WorkspaceIdentity,
        access: WorkspaceAccessContext,
        title: str = "",
    ) -> ConversationSession:
        """创建目标 Workspace 的空会话。"""
        scope = self._authorizer.authorize_root(access, target_workspace)
        return self._sessions.create(scope.workspace_identity, title)

    def list_sessions(
        self, *, target_workspace: WorkspaceIdentity, access: WorkspaceAccessContext
    ) -> tuple[SessionSummary, ...]:
        """列出目标 Workspace 的会话摘要。"""
        scope = self._authorizer.authorize_root(access, target_workspace)
        return self._sessions.list_sessions(scope.workspace_identity)

    def get_session(
        self,
        session_id: str,
        *,
        target_workspace: WorkspaceIdentity,
        access: WorkspaceAccessContext,
    ) -> ConversationSession:
        """加载全部历史，跨 Workspace 查找与不存在统一拒绝。"""
        scope = self._authorizer.authorize_root(access, target_workspace)
        return self._sessions.get(scope.workspace_identity, session_id)

    def rename_session(
        self,
        session_id: str,
        *,
        title: str,
        target_workspace: WorkspaceIdentity,
        access: WorkspaceAccessContext,
    ) -> ConversationSession:
        """修改会话标题，不改动历史。"""
        scope = self._authorizer.authorize_root(access, target_workspace)
        return self._sessions.rename(scope.workspace_identity, session_id, title)

    def delete_session(
        self,
        session_id: str,
        *,
        target_workspace: WorkspaceIdentity,
        access: WorkspaceAccessContext,
    ) -> None:
        """删除空闲会话，运行中的会话显式拒绝。"""
        scope = self._authorizer.authorize_root(access, target_workspace)
        self._sessions.delete(scope.workspace_identity, session_id)
