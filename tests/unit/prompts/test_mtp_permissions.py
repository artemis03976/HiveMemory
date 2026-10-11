"""MTP 教学按 frame 动词集合裁剪，系统工具不设权限过滤。"""

import pytest

from hivememory.prompts.mtp import MTPPromptBuilder


@pytest.mark.parametrize("language", ["zh", "en"])
def test_read_search_run_prompt_omits_other_verbs_from_all_teaching(language):
    """规格、行为准则与演示都不能教导未获准的 WRITE、UPDATE 或 CALL。"""
    output = MTPPromptBuilder(
        language=language,
        allowed_verbs={"READ", "SEARCH", "RUN"},
    ).build()

    assert "- READ:" in output
    assert "- SEARCH:" in output
    assert "- RUN:" in output
    assert "WRITE" not in output
    assert "UPDATE" not in output
    assert "CALL" not in output
    assert 'SEARCH | * | query="login deploy script"' in output


@pytest.mark.parametrize("language", ["zh", "en"])
def test_run_only_prompt_omits_resource_teaching_and_preserves_all_system_tools(language):
    """空操作集合对应的 RUN 仍能展示完整系统菜单，其它教学不泄露禁用动词。"""
    output = MTPPromptBuilder(language=language, allowed_verbs={"RUN"}).build()

    assert "- RUN:" in output
    assert "SEARCH" not in output
    assert "READ" not in output
    assert "WRITE" not in output
    assert "UPDATE" not in output
    assert "CALL" not in output
    assert "sys_clock" in output
    assert "sys_web_search" in output
    assert "sys_read_file" in output
    assert "sys_write_file" in output
    assert "sys_python_repl" in output
    assert "tool_deploy_login" not in output


@pytest.mark.parametrize("language", ["zh", "en"])
@pytest.mark.parametrize("verb", ["WRITE", "UPDATE"])
def test_write_update_instruction_does_not_imply_read_permission(language, verb):
    """允许写入不隐含 READ，动词定义不能建议随后执行未获准的读取。"""
    output = MTPPromptBuilder(language=language, allowed_verbs={verb, "RUN"}).build()

    assert f"- {verb}:" in output
    assert "pending alias" in output
    assert "READ" not in output
    assert "SEARCH" not in output
    assert "CALL" not in output


def test_no_verb_filter_renders_all_current_protocol_verbs():
    """独立构建器省略过滤条件时完整渲染当前协议。"""
    output = MTPPromptBuilder(language="en").build()

    assert "- SEARCH:" in output
    assert "- READ:" in output
    assert "- RUN:" in output
    assert "- WRITE:" in output
    assert "- UPDATE:" in output
    assert "- CALL:" in output


def test_empty_verb_list_preserves_protocol_structure_and_system_menu():
    """显式空教学集合不渲染动词定义，工具列表仍由工具注册表提供。"""
    output = MTPPromptBuilder(language="en", allowed_verbs=[]).build()

    assert "3. VERBS:" in output
    assert "- SEARCH:" not in output
    assert "- READ:" not in output
    assert "- RUN:" not in output
    assert "- WRITE:" not in output
    assert "- UPDATE:" not in output
    assert "- CALL:" not in output
    assert "[RUNTIME TOOLS]" in output
    assert "sys_clock" in output


def test_lowercase_verb_filter_normalizes_teaching_and_protocol_list():
    """大小写规范化同时应用到规格和行为教学。"""
    output = MTPPromptBuilder(language="en", allowed_verbs=["read", "search", "run"]).build()

    assert "- READ:" in output
    assert "- SEARCH:" in output
    assert "- RUN:" in output
    assert "WRITE" not in output
    assert "UPDATE" not in output
    assert "CALL" not in output


@pytest.mark.parametrize(
    "permitted_verbs, includes_demo",
    [
        ({"SEARCH", "READ", "RUN"}, True),
        ({"READ", "RUN"}, False),
        ({"SEARCH", "RUN"}, False),
        ({"SEARCH", "READ"}, False),
    ],
    ids=["all-demo-verbs", "without-search", "without-read", "without-run"],
)
@pytest.mark.parametrize("language", ["zh", "en"])
def test_demo_requires_every_verb_used_by_the_example(permitted_verbs, includes_demo, language):
    """示例作为完整流程保留或省略，不裁剪成无法执行的半条教学链。"""
    output = MTPPromptBuilder(language=language, allowed_verbs=permitted_verbs).build()

    assert ("tool_deploy_login" in output) == includes_demo


def test_filtered_prompt_keeps_core_sections_and_error_recovery():
    """缺少示例所需动词时，核心结构和通用错误恢复仍保留。"""
    output = MTPPromptBuilder(language="en", allowed_verbs={"READ", "RUN"}).build()

    assert "HIVE MEMORY KERNEL CONTEXT" in output
    assert "PROTOCOL RULES" in output
    assert "CONSTRAINTS" in output
    assert "BEHAVIORAL GUIDELINES" in output
    assert "[RUNTIME TOOLS]" in output
    assert "ERROR RECOVERY" in output
    assert "[Syntax Error]" in output
    assert "ONE-SHOT DEMONSTRATION" not in output


def test_custom_tool_registry_is_rendered_completely():
    """完整菜单由实际传入的工具列表决定，无 agent 工具白名单入口。"""
    output = MTPPromptBuilder(
        language="en",
        runtime_tools=[("tool_a", "Tool A"), ("tool_b", "Tool B"), ("tool_c", "Tool C")],
        allowed_verbs={"RUN"},
    ).build()

    assert "- `tool_a`: Tool A" in output
    assert "- `tool_b`: Tool B" in output
    assert "- `tool_c`: Tool C" in output
    assert "sys_clock" not in output


def test_unknown_verbs_are_ignored_without_enabling_other_verbs():
    """未知标记不进入教学，也不扩大到已知但未提供的动词。"""
    output = MTPPromptBuilder(language="en", allowed_verbs=["READ", "UNKNOWN_VERB"]).build()

    assert "- READ:" in output
    assert "UNKNOWN_VERB" not in output
    assert "SEARCH" not in output
    assert "WRITE" not in output


def test_duplicate_verbs_are_only_listed_once():
    """重复输入不能产生重复的动词定义。"""
    output = MTPPromptBuilder(language="en", allowed_verbs=["READ", "READ", "SEARCH"]).build()

    assert output.count("- READ:") == 1
    assert output.count("- SEARCH:") == 1


def test_guidance_filter_detects_verbs_adjacent_to_chinese_text(monkeypatch):
    """中文教学中紧贴汉字的禁用动词同样被裁剪，普通单词中的同名片段不受影响。"""
    import hivememory.prompts.mtp as mtp_prompt

    original = mtp_prompt.get_mtp_prompt_text
    guidance = "\n".join(
        [
            "- 回答前先SEARCH再作答",
            "- 已用READ读取的内容不要重复读取",
            "- ALREADY_DONE 标记保持原样",
            "- 保持回答简洁",
        ]
    )

    def template(key, language):
        return guidance if key == "behavioral_guidelines" else original(key, language)

    monkeypatch.setattr(mtp_prompt, "get_mtp_prompt_text", template)
    output = MTPPromptBuilder(language="zh", allowed_verbs={"RUN"}).build()

    assert "先SEARCH再作答" not in output
    assert "已用READ读取" not in output
    assert "ALREADY_DONE 标记保持原样" in output
    assert "保持回答简洁" in output
