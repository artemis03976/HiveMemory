from hivememory.prompts.system_prompt import SystemPromptBuilder


def test_system_prompt_default_chinese_texts():
    prompt = (
        SystemPromptBuilder()
        .with_storage_offline_notice()
        .with_persona("负责代码实现。")
        .with_topic_state("当前正在迁移 prompt i18n。")
        .build()
    )

    assert "[系统通知]" in prompt
    assert "### 角色设定 ###" in prompt
    assert "[话题状态]" in prompt
    assert "[Topic State]" not in prompt


def test_system_prompt_english_texts():
    prompt = (
        SystemPromptBuilder(language="en")
        .with_storage_offline_notice()
        .with_persona("Owns code implementation.")
        .with_topic_state("Migrating prompt i18n.")
        .build()
    )

    assert "[SYSTEM NOTICE]" in prompt
    assert "### PERSONA ###" in prompt
    assert "[Topic State]" in prompt


def test_system_prompt_language_alias_fallback():
    prompt = (
        SystemPromptBuilder(language="en-US")
        .with_storage_offline_notice()
        .with_persona("Owns code implementation.")
        .with_topic_state("Migrating prompt i18n.")
        .build()
    )

    assert "[SYSTEM NOTICE]" in prompt
    assert "### PERSONA ###" in prompt
