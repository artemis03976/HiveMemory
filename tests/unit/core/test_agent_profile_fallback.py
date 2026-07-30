from hivememory.agent_runtime.mtp.syscalls.registry import build_kernel_registry
from hivememory.core.models.agent import (
    OMNI_DOLL_ALLOWED_MTP_VERBS,
    OMNI_DOLL_ALLOWED_SYS_TOOLS,
    OMNI_DOLL_PROFILE,
)
from hivememory.core.mtp.models import MTPVerb


def test_omni_doll_fallback_uses_explicit_current_capability_lists():
    assert OMNI_DOLL_PROFILE.allowed_mtp_verbs is not None
    assert OMNI_DOLL_PROFILE.allowed_sys_tools is not None
    assert set(OMNI_DOLL_ALLOWED_MTP_VERBS) == {verb.value for verb in MTPVerb}
    assert set(OMNI_DOLL_ALLOWED_SYS_TOOLS) == set(build_kernel_registry())
    assert OMNI_DOLL_PROFILE.allowed_mtp_verbs == OMNI_DOLL_ALLOWED_MTP_VERBS
    assert OMNI_DOLL_PROFILE.allowed_sys_tools == OMNI_DOLL_ALLOWED_SYS_TOOLS
