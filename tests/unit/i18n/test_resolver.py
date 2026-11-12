"""hivememory.i18n.resolver：一个语言设置，显式值优先。"""

import pytest

from hivememory.i18n import Language, get_language, resolve_language, set_language


def test_without_explicit_value_follows_the_setting():
    set_language(Language.EN)

    assert get_language() == Language.EN
    assert resolve_language() == Language.EN


def test_explicit_value_wins_and_accepts_aliases():
    set_language(Language.ZH)

    assert resolve_language(explicit="english") == Language.EN


def test_unrecognized_explicit_value_falls_back_to_the_setting():
    set_language(Language.EN)

    assert resolve_language(explicit="fr") == Language.EN


def test_setting_rejects_unknown_language():
    with pytest.raises(ValueError):
        set_language("fr")  # type: ignore[arg-type]
