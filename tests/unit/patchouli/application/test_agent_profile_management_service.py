"""AgentProfileManagementService 单元测试。

被测对象：Patchouli 的 Profile 管理应用服务，读写真实的 AgentProfileStore（临时目录）。
守护本层的判断：新建不覆盖同名 Profile、编辑只改给出的字段且 agent_id 不变、提交的
字段与读取文件走同一套校验、预设只写尚不存在的 Profile。
"""

from unittest.mock import AsyncMock

import pytest

from hivememory.core.errors import (
    AgentProfileExistsError,
    InvalidAgentProfileError,
    ResourceNotFoundError,
)
from hivememory.core.models import AgentProfile
from hivememory.memory_library.agent_profiles import AgentProfileStore
from hivememory.patchouli.application import AgentProfileManagementService
from hivememory.patchouli.application.preset_agent_profiles import PRESET_AGENT_PROFILES


@pytest.fixture
def store(tmp_path):
    return AgentProfileStore(tmp_path)


@pytest.fixture
def service(store):
    return AgentProfileManagementService(store=store, bus=AsyncMock())


@pytest.mark.asyncio
async def test_created_profile_is_saved_with_persona_as_body(service, store):
    created = await service.create_agent_profile(
        "worker", {"title": "工人", "persona": "认真干活", "temperature": 0.4}
    )

    assert created.profile == AgentProfile(
        agent_id="worker", title="工人", persona="认真干活", temperature=0.4
    )
    assert store.get("worker").profile == created.profile


@pytest.mark.asyncio
async def test_create_does_not_overwrite_existing_profile(service, store):
    """同名文件已存在时拒绝新建，即使那个文件不合格也不覆盖。"""
    await service.create_agent_profile("worker", {"title": "原来的"})

    with pytest.raises(AgentProfileExistsError):
        await service.create_agent_profile("worker", {"title": "新的"})

    assert store.get("worker").profile.title == "原来的"


@pytest.mark.asyncio
async def test_update_changes_only_given_fields(service, store):
    await service.create_agent_profile(
        "worker", {"title": "工人", "persona": "认真干活", "temperature": 0.4}
    )

    updated = await service.update_agent_profile("worker", {"summary": "搬砖", "temperature": None})

    assert updated.profile == AgentProfile(
        agent_id="worker", title="工人", summary="搬砖", persona="认真干活", temperature=None
    )
    assert store.get("worker").profile == updated.profile


@pytest.mark.asyncio
async def test_update_of_missing_profile_is_not_found(service):
    with pytest.raises(ResourceNotFoundError):
        await service.update_agent_profile("nobody", {"title": "x"})


@pytest.mark.parametrize(
    ("changes", "reason"),
    [({"top_p": 2.0}, "invalid_value"), ({"agent_id": "renamed"}, "unknown_field")],
)
@pytest.mark.asyncio
async def test_invalid_update_is_rejected_without_writing(service, store, changes, reason):
    """提交的字段与读取文件走同一套校验；agent_id 不是可编辑的字段，改名被拒绝。"""
    await service.create_agent_profile("worker", {"title": "工人"})

    with pytest.raises(InvalidAgentProfileError) as exc_info:
        await service.update_agent_profile("worker", changes)

    assert exc_info.value.details["reason"] == reason
    assert store.get("worker").profile == AgentProfile(agent_id="worker", title="工人")


@pytest.mark.asyncio
async def test_presets_are_written_only_where_no_profile_exists(service, store):
    """已有同名文件的预设跳过，不覆盖用户改过的 Profile。"""
    kept = PRESET_AGENT_PROFILES[0].model_copy(update={"persona": "用户改过的人设"})
    store.save(kept)

    created = await service.seed_preset_agent_profiles()

    assert created == [preset.agent_id for preset in PRESET_AGENT_PROFILES[1:]]
    assert store.get(kept.agent_id).profile.persona == "用户改过的人设"
    assert [stored.profile for stored in store.list()] == sorted(
        [kept, *PRESET_AGENT_PROFILES[1:]], key=lambda profile: profile.agent_id
    )
