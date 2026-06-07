from hivememory.utils.token_estimator import TokenEstimator, estimate_tokens


def test_estimate_handles_empty_and_none():
    assert TokenEstimator.estimate(None) == 0
    assert TokenEstimator.estimate("") == 0


def test_estimate_counts_cjk_and_other_characters():
    assert TokenEstimator.count_cjk_chars("Hello世界かな한글") == 6
    assert TokenEstimator.estimate("Hello世界") == 2


def test_estimate_with_ratio_uses_custom_ratios():
    assert TokenEstimator.estimate_with_ratio("abcd世界", chinese_ratio=1.0, other_ratio=2.0) == 4
    assert TokenEstimator.estimate_with_ratio("abcd世界", chinese_ratio=0, other_ratio=0) == 0


def test_estimate_messages_uses_configurable_content_key():
    messages = [
        {"role": "user", "content": "abcd"},
        {"role": "assistant", "content": "世界"},
        {"role": "tool", "other": "ignored"},
    ]

    assert TokenEstimator.estimate_messages(messages) == 2
    assert TokenEstimator.estimate_messages([{"text": "abcd"}], content_key="text") == 1


def test_estimate_dict_defaults_to_all_stringifiable_values():
    data = {"summary": "abcd", "count": 1234, "missing": None}

    assert TokenEstimator.estimate_dict(data) == 2
    assert TokenEstimator.estimate_dict(data, keys=["summary"]) == 1
    assert TokenEstimator.estimate_dict({}, keys=["summary"]) == 0


def test_estimate_tokens_is_backward_compatible_alias():
    assert estimate_tokens("abcd") == TokenEstimator.estimate("abcd")
