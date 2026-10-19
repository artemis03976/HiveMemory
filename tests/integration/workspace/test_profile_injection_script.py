"""Profile 初始化脚本通过真实认证网关和 root API 注入，重复执行保持幂等。"""

from types import SimpleNamespace

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.config.app import HiveMemoryConfig
from hivememory.core.errors import AdmissionDeniedError
from hivememory.core.models import IndexLayer, MemoryAtom, MemoryType, PayloadLayer
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.workspace.capability.root.agent_profiles import RootAgentProfileService
from scripts import inject_profiles
from tests.helpers.memory import make_memory_metadata
from tests.helpers.workspace import make_access_composition, make_actor_access_record


class _ScriptSystems:
    """替换脚本构建的系统：真实网关与 root API，只在 Patchouli 路由边界保存 Profile。"""

    def __init__(self) -> None:
        self.atoms = {}
        self.stopped = 0
        self.gateways = []

    def build(self, *, config):
        composition = make_access_composition(
            [make_actor_access_record(owner_user_id="u1", agent_id="system")],
            principal_id="hivememory:profile-injector",
        )
        bus = GlobalSystemBus()

        async def list_profiles(**_kwargs):
            return list(self.atoms.values())

        async def create_profile(identity_scope, creation):
            atom = MemoryAtom(
                meta=make_memory_metadata(
                    user_id=identity_scope.workspace_identity.owner_user_id,
                    source_agent_id=identity_scope.actor_identity.agent_id,
                ),
                index=IndexLayer(
                    title=creation.title,
                    memory_type=MemoryType.AGENT_PROFILE,
                    alias=creation.alias,
                ),
                payload=PayloadLayer(content=creation.content, agent_config=creation.agent_config),
            )
            self.atoms[atom.index.alias] = atom
            return atom

        bus.register(PatchouliRoutes.AGENT_PROFILE_LIST, list_profiles)
        bus.register(PatchouliRoutes.AGENT_PROFILE_CREATE, create_profile)
        issued = []
        authenticate = composition.gateway.authenticate

        async def recording_authenticate(**kwargs):
            context = await authenticate(**kwargs)
            issued.append(context)
            return context

        composition.gateway.authenticate = recording_authenticate
        self.gateways.append((composition.gateway, issued))

        async def start():
            return None

        async def stop():
            self.stopped += 1
            composition.gateway.close()
            composition.gateway.revoke_all_contexts()

        return SimpleNamespace(
            access_gateway=composition.gateway,
            root_agent_service=RootAgentProfileService(
                bus, operation_authorizer=composition.authorizer
            ),
            start=start,
            stop=stop,
        )


@pytest.fixture
def script_systems(monkeypatch):
    systems = _ScriptSystems()
    monkeypatch.setattr(inject_profiles.HiveMemorySystem, "build", systems.build)
    monkeypatch.setattr(inject_profiles, "load_app_config", HiveMemoryConfig)
    return systems


@pytest.mark.asyncio
async def test_profile_injection_is_idempotent_and_releases_each_context(script_systems):
    """登记的本地来源以 system 经 root API 注入；重复执行不重写，每次都撤销 context。"""
    await inject_profiles.main("u1", "hivememory:profile-injector")
    original_ids = {alias: atom.id for alias, atom in script_systems.atoms.items()}
    await inject_profiles.main("u1", "hivememory:profile-injector")

    assert set(script_systems.atoms) == {"coder_doll", "reviewer_doll"}
    assert {alias: atom.id for alias, atom in script_systems.atoms.items()} == original_ids
    assert {atom.meta.provenance.source_agent_id for atom in script_systems.atoms.values()} == {
        "system"
    }
    assert {
        atom.meta.workspace_identity.owner_user_id for atom in script_systems.atoms.values()
    } == {"u1"}
    assert script_systems.stopped == 2
    for gateway, issued in script_systems.gateways:
        (context,) = issued
        assert gateway.describe_context(context) is None


@pytest.mark.asyncio
async def test_profile_injection_from_unknown_principal_is_denied_and_stops_system(
    script_systems,
):
    """未登记来源不能进入 root API；认证失败不写入 Profile，系统仍被关闭。"""
    with pytest.raises(AdmissionDeniedError) as error:
        await inject_profiles.main("u1", "unknown-injector")

    assert error.value.details["reason"] == "unknown_principal"
    assert script_systems.atoms == {}
    assert script_systems.stopped == 1
    ((_gateway, issued),) = script_systems.gateways
    assert issued == []
