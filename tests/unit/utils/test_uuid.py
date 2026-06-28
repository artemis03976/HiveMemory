from uuid import UUID, uuid4

from hivememory.utils.uuid import normalize_uuid


def test_normalize_uuid_returns_existing_uuid():
    value = uuid4()

    assert normalize_uuid(value) is value


def test_normalize_uuid_converts_string_to_uuid():
    value = uuid4()

    normalized = normalize_uuid(str(value))

    assert isinstance(normalized, UUID)
    assert normalized == value
