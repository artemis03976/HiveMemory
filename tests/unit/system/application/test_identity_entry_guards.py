"""身份入口收敛守卫测试。

守卫目标（v0.7.0 A1 访问边界返工后的入口形态）：

1. ``system/application`` 公共服务方法签名不再出现 ``user_id: str`` 裸参数；
2. workspace 能力层与任务进程入口不再接收预先组装的 ``identity_scope``：
   能力层公共方法以 ``access`` + ``target_workspace`` 显式配对进入授权点，
   任务进程以 actor/workspace 声明进入注册入口；
3. ``PassiveIngressService``（``/ingest``）不经网关，仍以
   ``identity_scope`` 一次性冻结身份——这是不变量 1 的已知例外；
4. 任务进程注册入口在认证前拒绝保留 ``system`` actor，且不创建进程；
5. 取消路径以请求方 context 与进程记录比对驻留坐标：跨 user/workspace
   的取消不可见（``not_found``）；未签发的请求方 context 是接线缺陷；
6. server 的 ``authenticate_request_access`` 以请求级 RunBinding 经网关
   签发请求级 context。

server 声明解析的合并/冲突规则由 ``tests/unit/server/routers/
test_identity_resolution.py`` 覆盖，此处不重复。
"""

import inspect
from unittest.mock import AsyncMock

import pytest

from hivememory.core.access import AccessRunType
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.errors import ScopeRequiredError, WorkspaceDomainError
from hivememory.core.models import ActorIdentity
from hivememory.server import deps
from hivememory.system.application.passive_ingress_service import PassiveIngressService
from hivememory.workspace.capability.agent_profiles import AgentApplicationService
from hivememory.workspace.capability.assets import WorkspaceAssetApplicationService
from hivememory.workspace.capability.memory import MemoryApplicationService
from hivememory.workspace.capability.memory_tasks import MemoryTaskApplicationService
from hivememory.workspace.capability.topic import TopicApplicationService
from hivememory.workspace.process.service import TaskProcessService
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.process import make_task_process_service
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_identity_scope,
    make_management_identity_scope,
    make_workspace_identity,
)

_APPLICATION_SERVICES = (
    AgentApplicationService,
    TaskProcessService,
    MemoryApplicationService,
    PassiveIngressService,
    TopicApplicationService,
)

#: 经统一认证网关接入的服务：能力层以 access + target_workspace 授权，
#: 任务进程以 actor/workspace 声明注册；均不得再接收 identity_scope。
_GATEWAY_WIRED_SERVICES = (
    AgentApplicationService,
    MemoryApplicationService,
    MemoryTaskApplicationService,
    TopicApplicationService,
    WorkspaceAssetApplicationService,
    TaskProcessService,
)

#: 显式接收目标 workspace 的能力层服务：每个目标都必须配对访问 context。
_CAPABILITY_SERVICES = (
    AgentApplicationService,
    MemoryApplicationService,
    MemoryTaskApplicationService,
    TopicApplicationService,
    WorkspaceAssetApplicationService,
)

MAIN = make_workspace_identity(owner_user_id="owner", workspace_id="main_workspace")
ISOLATION = make_workspace_identity(owner_user_id="owner", workspace_id="isolation_workspace")
OTHER_USER = make_workspace_identity(owner_user_id="other", workspace_id="main_workspace")
U1_MAIN = make_workspace_identity(owner_user_id="u1", workspace_id="main_workspace")


def _public_methods(cls) -> list[str]:
    return [
        name for name, fn in inspect.getmembers(cls, inspect.isfunction) if not name.startswith("_")
    ]


class TestServiceSignatureGuard:
    """应用服务入口不得再以裸 user_id 字符串作为公共签名。"""

    @pytest.mark.parametrize("service_cls", _APPLICATION_SERVICES)
    def test_public_methods_do_not_accept_bare_user_id(self, service_cls):
        for name in _public_methods(service_cls):
            signature = inspect.signature(getattr(service_cls, name))
            assert "user_id" not in signature.parameters, (
                f"{service_cls.__name__}.{name} 仍接受裸 user_id 参数，"
                "身份应经 access/target_workspace 授权点或声明入口进入"
            )

    @pytest.mark.parametrize("service_cls", _GATEWAY_WIRED_SERVICES)
    def test_gateway_wired_services_do_not_accept_identity_scope(self, service_cls):
        """经网关接入的服务不再接收预先组装的 IdentityScope（不变量 1）。"""
        for name in _public_methods(service_cls):
            signature = inspect.signature(getattr(service_cls, name))
            assert "identity_scope" not in signature.parameters, (
                f"{service_cls.__name__}.{name} 仍接收 identity_scope："
                "能力层应改为 access + target_workspace，任务进程应改为 "
                "actor/workspace 声明注册"
            )

    @pytest.mark.parametrize("service_cls", _CAPABILITY_SERVICES)
    def test_target_workspace_methods_are_paired_with_access(self, service_cls):
        """凡显式接收目标 workspace 的公共方法必须同时接收访问 context。"""
        for name in _public_methods(service_cls):
            params = inspect.signature(getattr(service_cls, name)).parameters
            if "target_workspace" in params:
                assert "access" in params, (
                    f"{service_cls.__name__}.{name} 接收目标 workspace 却没有 "
                    "access 授权入口：调用方可以绕过 operation 授权"
                )

    def test_passive_ingress_keeps_identity_scope_as_known_exception(self):
        """/ingest 被动摄入不经网关：身份仍以 identity_scope 一次性冻结。"""
        for name in ("ingest_event", "flush_conversation"):
            params = inspect.signature(getattr(PassiveIngressService, name)).parameters
            assert "identity_scope" in params, (
                f"PassiveIngressService.{name} 丢失 identity_scope 入口："
                "/ingest 是不变量 1 的已知例外，不经认证网关"
            )
            assert "access" not in params, (
                f"PassiveIngressService.{name} 出现 access 参数："
                "被动摄入不经统一认证网关，不应接收访问 context"
            )


class TestChatIdentityGuard:
    """任务进程必须由具体 Agent 执行；保留 system actor 在注册前被拒绝。"""

    @staticmethod
    def _make_service(composition) -> TaskProcessService:
        return make_task_process_service(
            AsyncMock(),  # system actor 在认证与总线触达前即被拒绝
            cpu=ScriptedCPU(result=make_cpu_result()),
            access_gateway=composition.gateway,
            operation_authorizer=composition.authorizer,
        )

    @pytest.mark.asyncio
    async def test_register_process_rejects_system_actor(self):
        """保留 system actor 的声明在注册入口认证前显式失败。"""
        composition = make_access_composition(
            [make_actor_access_record(owner_user_id="u1", agent_id=SYSTEM_AGENT_ID)],
            default_workspace=U1_MAIN,
        )
        service = self._make_service(composition)

        with pytest.raises(WorkspaceDomainError) as exc_info:
            await service.register_process(
                adapter="local",
                principal=composition.principal,
                actor=ActorIdentity(user_id="u1", agent_id=SYSTEM_AGENT_ID),
                workspace=U1_MAIN,
                process_id="process-system-1",
                message="hello",
            )

        assert exc_info.value.details["agent_id"] == SYSTEM_AGENT_ID

    @pytest.mark.asyncio
    async def test_rejected_system_actor_registration_creates_no_process(self):
        """被拒绝的 system actor 注册不创建、不登记进程（两阶段认证之前失败）。"""
        composition = make_access_composition(
            [make_actor_access_record(owner_user_id="u1", agent_id=SYSTEM_AGENT_ID)],
            default_workspace=U1_MAIN,
        )
        service = self._make_service(composition)
        with pytest.raises(WorkspaceDomainError):
            await service.register_process(
                adapter="local",
                principal=composition.principal,
                actor=ActorIdentity(user_id="u1", agent_id=SYSTEM_AGENT_ID),
                workspace=U1_MAIN,
                process_id="process-system-2",
                message="hello",
            )

        requestor = await composition.authenticate(agent_id=SYSTEM_AGENT_ID, user_id="u1")

        assert service.process_status("process-system-2", access=requestor) is None


class TestCancelUsesRequestorContext:
    """取消路径：请求方 context 只用于与进程记录比对驻留坐标（P-7）。"""

    @staticmethod
    def _make_composition():
        return make_access_composition(
            [
                # 进程 actor：owner 在 main_workspace 的具体 Agent。
                make_actor_access_record(owner_user_id="owner", agent_id="omni_doll"),
                # 同 workspace 的管理语义请求方（user + system 声明）。
                make_actor_access_record(owner_user_id="owner", agent_id=SYSTEM_AGENT_ID),
                # 跨 workspace 请求方：同 owner 的隔离 workspace。
                make_actor_access_record(
                    owner_user_id="owner",
                    agent_id=SYSTEM_AGENT_ID,
                    workspace_id="isolation_workspace",
                ),
                # 跨 user 请求方：另一用户的 workspace。
                make_actor_access_record(owner_user_id="other", agent_id=SYSTEM_AGENT_ID),
            ],
            default_workspace=MAIN,
        )

    @staticmethod
    def _make_service(composition) -> TaskProcessService:
        return make_task_process_service(
            AsyncMock(),
            cpu=ScriptedCPU(result=make_cpu_result()),
            access_gateway=composition.gateway,
            operation_authorizer=composition.authorizer,
        )

    @staticmethod
    async def _register_running_process(service, composition, process_id: str):
        """注册一个已启动的流式进程并消费 process_id 事件，保持记录存活。"""
        handle = await service.register_process(
            adapter="local",
            principal=composition.principal,
            actor=ActorIdentity(user_id="owner", agent_id="omni_doll"),
            workspace=MAIN,
            process_id=process_id,
            message="hello",
        )
        stream = service.run_process(handle, stream=True)
        first = await stream.__anext__()
        assert first["event"] == "process_id"
        return stream

    @pytest.mark.asyncio
    async def test_cancel_across_users_returns_not_found(self):
        composition = self._make_composition()
        service = self._make_service(composition)
        stream = await self._register_running_process(service, composition, "process-owner-1")

        requestor = await composition.authenticate(
            agent_id=SYSTEM_AGENT_ID, user_id="other", workspace=OTHER_USER
        )
        result = service.cancel_process("process-owner-1", access=requestor)

        assert result.cancelled is False
        assert result.status == "not_found"

        await stream.aclose()

    @pytest.mark.asyncio
    async def test_cancel_across_workspaces_returns_not_found(self):
        composition = self._make_composition()
        service = self._make_service(composition)
        stream = await self._register_running_process(service, composition, "process-owner-2")

        requestor = await composition.authenticate(
            agent_id=SYSTEM_AGENT_ID, user_id="owner", workspace=ISOLATION
        )
        result = service.cancel_process("process-owner-2", access=requestor)

        assert result.cancelled is False
        assert result.status == "not_found"

        await stream.aclose()

    @pytest.mark.asyncio
    async def test_same_workspace_management_context_is_authorized_for_control(self):
        """取消不携带 agent 选择：与进程同 owner/workspace 的请求方可控进程。

        请求方是管理语义的 (user, ``system``) 声明，agent 维度与进程 actor
        不同也不影响驻留坐标比对（P-7 只比对 owner + workspace）。
        """
        composition = self._make_composition()
        service = self._make_service(composition)
        stream = await self._register_running_process(service, composition, "process-owner-3")

        requestor = await composition.authenticate(
            agent_id=SYSTEM_AGENT_ID, user_id="owner", workspace=MAIN
        )
        snapshot = service.process_status("process-owner-3", access=requestor)

        assert snapshot is not None
        assert snapshot.process_id == "process-owner-3"

        await stream.aclose()

    @pytest.mark.asyncio
    async def test_cancel_from_same_workspace_with_management_context_succeeds(self):
        """同 owner/workspace 的管理请求方取消进程成功（not_found 语义之外的正路径）。"""
        composition = self._make_composition()
        service = self._make_service(composition)
        stream = await self._register_running_process(service, composition, "process-owner-5")

        requestor = await composition.authenticate(
            agent_id=SYSTEM_AGENT_ID, user_id="owner", workspace=MAIN
        )
        result = service.cancel_process("process-owner-5", access=requestor)

        assert result.cancelled is True

        await stream.aclose()

    @pytest.mark.asyncio
    async def test_cancel_with_revoked_requestor_context_is_a_wiring_defect(self):
        """已撤销的请求方 context 在取消入口显式失败（server 必须持有有效的请求级 context）。"""
        composition = self._make_composition()
        service = self._make_service(composition)
        stream = await self._register_running_process(service, composition, "process-owner-4")
        requestor = await composition.authenticate(
            agent_id=SYSTEM_AGENT_ID, user_id="owner", workspace=MAIN
        )
        composition.gateway.invalidate_context(requestor)

        with pytest.raises(ScopeRequiredError) as exc_info:
            service.cancel_process("process-owner-4", access=requestor)

        assert exc_info.value.details["reason"] == "context_not_issued"

        await stream.aclose()


class TestRequestAccessEntryGuard:
    """server 请求级访问入口：声明进入认证，签发绑定请求级运行。"""

    def test_request_identity_claims_carry_only_declarations(self):
        """认证前声明载体只有 actor 与 workspace 两个声明字段。"""
        assert tuple(deps.RequestIdentityClaims.__dataclass_fields__) == ("actor", "workspace")

    @pytest.mark.asyncio
    async def test_authenticate_request_access_binds_request_run(self):
        """请求级认证经网关签发：授予内容携带 server principal 与请求运行类型。"""
        composition = make_access_composition(
            [
                make_actor_access_record(owner_user_id="u1", agent_id=None),
                make_actor_access_record(owner_user_id="u1", agent_id=SYSTEM_AGENT_ID),
            ],
            adapters=("http",),
        )
        claims = deps.resolve_request_identity_claims(
            deps.RequestIdentitySelection(user_id=None, workspace_id=None),
            explicit_user_id="u1",
        )

        access = await deps.authenticate_request_access(
            claims,
            gateway=composition.gateway,
            principal_id=composition.principal.principal_id,
        )

        summary = composition.gateway.describe_context(access)
        assert summary is not None
        assert summary.actor_user_id == "u1"
        assert summary.agent_id == SYSTEM_AGENT_ID
        assert summary.workspace_id == claims.workspace.workspace_id
        assert summary.principal_id == composition.principal.principal_id
        assert summary.run_type == AccessRunType.REQUEST.value


class TestManagementScopeUsesSystemActor:
    """/ingest 例外的测试 scope 构造器必须注入保留 system actor。"""

    def test_management_scope_carries_system_agent(self):
        scope = make_management_identity_scope(user_id="u1")
        assert scope.actor_identity.agent_id == SYSTEM_AGENT_ID
        assert scope.workspace_identity.owner_user_id == "u1"

    def test_management_scope_differs_from_agent_scope(self):
        assert (
            make_identity_scope(user_id="u1", agent_id="omni_doll").actor_identity.agent_id
            != SYSTEM_AGENT_ID
        )
