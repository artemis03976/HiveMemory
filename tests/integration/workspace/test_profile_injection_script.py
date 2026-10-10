"""Profile 初始化脚本通过真实认证网关和 root API 注入，重复执行保持幂等。"""

from types import SimpleNamespace

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.config.app import HiveMemoryConfig
from hivememory.core.contracts.routes import GlobalRoutes
from hivememory.core.errors import AdmissionDeniedError
from hivememory.workspace.capability.root.agent_profiles import RootAgentProfileService
from scripts import inject_profiles
from tests.helpers.workspace import make_access_composition, make_actor_access_record


@pytest.mark.asyncio
@pytest.mark.parametrize("principal_id", ["hivememory:profile-injector", "unknown-injector"])
async def test_profile_injection_authenticates_and_releases_context(monkeypatch, principal_id):
    """重复初始化不重复写入；未知来源不能进入 root API，认证失败也关闭系统。"""
    atoms = {}
    stopped = []
    bindings = []

    def build_system(*, config):
        composition = make_access_composition(
            [make_actor_access_record(owner_user_id="u1", agent_id="system")],
            principal_id="hivememory:profile-injector",
        )
        bus = GlobalSystemBus()

        async def list_profiles(**_kwargs):
            return list(atoms.values())

        async def create_profile(scope, atom):
            atoms[atom.index.alias] = atom
            return atom

        bus.register(GlobalRoutes.PATCHOULI_AGENT_PROFILE_LIST, list_profiles)
        bus.register(GlobalRoutes.PATCHOULI_AGENT_PROFILE_CREATE, create_profile)
        issued = []
        authenticate = composition.gateway.authenticate

        async def recording_authenticate(**kwargs):
            context = await authenticate(**kwargs)
            issued.append(context)
            return context

        monkeypatch.setattr(composition.gateway, "authenticate", recording_authenticate)
        bindings.append((composition.gateway, issued))

        async def start():
            return None

        async def stop():
            stopped.append(True)
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

    monkeypatch.setattr(inject_profiles.HiveMemorySystem, "build", build_system)
    monkeypatch.setattr(inject_profiles, "load_app_config", HiveMemoryConfig)
    if principal_id == "unknown-injector":
        with pytest.raises(AdmissionDeniedError) as error:
            await inject_profiles.main("u1", principal_id)
        assert error.value.details["reason"] == "unknown_principal"
        assert atoms == {}
        assert stopped == [True]
        assert bindings[0][1] == []
    else:
        await inject_profiles.main("u1", principal_id)
        original_ids = {alias: atom.id for alias, atom in atoms.items()}
        await inject_profiles.main("u1", principal_id)
        assert set(atoms) == {"coder_doll", "reviewer_doll"}
        assert {alias: atom.id for alias, atom in atoms.items()} == original_ids
        assert {atom.meta.provenance.source_agent_id for atom in atoms.values()} == {"system"}
        assert {atom.meta.workspace_identity.owner_user_id for atom in atoms.values()} == {"u1"}
        assert stopped == [True, True]
        for gateway, issued in bindings:
            (context,) = issued
            assert gateway.describe_context(context) is None
