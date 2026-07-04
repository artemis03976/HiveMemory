from hivememory.patchouli.eye import TheEye as LegacyTheEye
from hivememory.system.gateway.eye import TheEye


def test_patchouli_eye_is_compatibility_shim():
    assert LegacyTheEye is TheEye
