"""MemoryCompiler 单元测试。"""

import pytest
from datetime import datetime, timedelta

from hivememory.core.models import (
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    IndexLayer,
    MetaData,
    VerificationStatus,
)
from hivememory.engines.memory_compiler import (
    MemoryCompiler,
    MemoryCompileTarget,
    MemoryCompileOptions,
    CompiledMemoryArtifact,
    MemoryEnvelopeSection,
    MemoryEnvelopeTarget,
)
from hivememory.engines.memory_compiler.envelopes import compile_envelope_from_ir
from hivememory.engines.memory_compiler.ir import MemoryBundleIR, MemorySectionIR
from hivememory.i18n import set_default_language


@pytest.fixture(autouse=True)
def reset_i18n():
    set_default_language("zh")
    yield
    set_default_language("zh")


@pytest.fixture
def compiler():
    return MemoryCompiler()


@pytest.fixture
def sample_atom():
    return MemoryAtom(
        index=IndexLayer(
            title="Python parse_date 函数",
            summary="基于 datetime 库的日期解析工具",
            memory_type=MemoryType.CODE_SNIPPET,
            tags=["python", "datetime", "utils"],
        ),
        payload=PayloadLayer(content="def parse_date(s):\n    return datetime.strptime(s, '%Y-%m-%d')"),
        meta=MetaData(
            source_agent_id="test",
            user_id="u1",
            updated_at=datetime.now() - timedelta(hours=2),
            confidence_score=0.95,
            verification_status=VerificationStatus.VERIFIED,
        ),
    )


@pytest.fixture
def agent_profile_atom():
    return MemoryAtom(
        index=IndexLayer(
            title="代码分析师",
            summary="擅长代码审查和重构建议",
            memory_type=MemoryType.AGENT_PROFILE,
            tags=["code", "review"],
        ),
        payload=PayloadLayer(content=""),
        meta=MetaData(
            source_agent_id="system",
            user_id="u1",
            updated_at=datetime.now(),
            confidence_score=1.0,
        ),
    )


class TestMemoryAtomCompilation:
    """测试 MemoryAtom 各目标编译。"""

    def test_prompt_full(self, compiler, sample_atom):
        artifact = compiler.compile(sample_atom, MemoryCompileTarget.PROMPT_FULL)
        assert isinstance(artifact, CompiledMemoryArtifact)
        assert artifact.target == MemoryCompileTarget.PROMPT_FULL
        assert artifact.source_kind == "atom"
        assert "parse_date" in artifact.text
        assert "<memory" in artifact.text
        assert "**类型**:" in artifact.text
        assert "**存档于**:" in artifact.text
        assert "**置信度**:" in artifact.text
        assert "**标签**:" in artifact.text
        assert "[完整内容]:" in artifact.text
        assert artifact.memory_id == str(sample_atom.id)

    def test_prompt_full_english(self, sample_atom):
        set_default_language("en")
        compiler = MemoryCompiler()
        artifact = compiler.compile(sample_atom, MemoryCompileTarget.PROMPT_FULL)

        assert "**Type**:" in artifact.text
        assert "**Archived At**:" in artifact.text
        assert "**Confidence**:" in artifact.text
        assert "**Tags**:" in artifact.text
        assert "[Full Content]:" in artifact.text
        assert "hours ago" in artifact.text
        assert "95% (High) [Verified]" in artifact.text
        assert "**类型**:" not in artifact.text

    def test_prompt_full_options_language_overrides_default(self, sample_atom):
        set_default_language("zh")
        compiler = MemoryCompiler()
        artifact = compiler.compile(
            sample_atom,
            MemoryCompileTarget.PROMPT_FULL,
            MemoryCompileOptions(language="en"),
        )

        assert "**Type**:" in artifact.text
        assert "[Full Content]:" in artifact.text
        assert "**类型**:" not in artifact.text

    def test_prompt_index(self, compiler, sample_atom):
        artifact = compiler.compile(sample_atom, MemoryCompileTarget.PROMPT_INDEX)
        assert artifact.target == MemoryCompileTarget.PROMPT_INDEX
        assert "<memory_index" in artifact.text
        assert "**类型**:" in artifact.text
        assert "**存档于**:" in artifact.text
        assert "**置信度**:" in artifact.text
        assert "**标签**:" in artifact.text
        assert "**内容摘要**:" in artifact.text
        assert "基于 datetime 库" in artifact.text

    def test_prompt_index_english(self, sample_atom):
        set_default_language("en")
        compiler = MemoryCompiler()
        artifact = compiler.compile(sample_atom, MemoryCompileTarget.PROMPT_INDEX)

        assert "**Type**:" in artifact.text
        assert "**Archived At**:" in artifact.text
        assert "**Confidence**:" in artifact.text
        assert "**Tags**:" in artifact.text
        assert "**Summary**:" in artifact.text
        assert "hours ago" in artifact.text
        assert "95% (High) [Verified]" in artifact.text
        assert "**内容摘要**:" not in artifact.text

    def test_prompt_index_options_language_overrides_default(self, sample_atom):
        set_default_language("zh")
        compiler = MemoryCompiler()
        artifact = compiler.compile(
            sample_atom,
            MemoryCompileTarget.PROMPT_INDEX,
            MemoryCompileOptions(language="en"),
        )

        assert "**Summary**:" in artifact.text
        assert "**内容摘要**:" not in artifact.text

    def test_prompt_index_empty_tags_i18n(self, sample_atom):
        sample_atom.index.tags = []

        set_default_language("zh")
        zh_artifact = MemoryCompiler().compile(
            sample_atom,
            MemoryCompileTarget.PROMPT_INDEX,
        )
        set_default_language("en")
        en_artifact = MemoryCompiler().compile(
            sample_atom,
            MemoryCompileTarget.PROMPT_INDEX,
        )

        assert "(无标签)" in zh_artifact.text
        assert "(No tags)" in en_artifact.text

    def test_dense_embedding(self, compiler, sample_atom):
        artifact = compiler.compile(sample_atom, MemoryCompileTarget.DENSE_EMBEDDING)
        assert artifact.target == MemoryCompileTarget.DENSE_EMBEDDING
        assert "Python parse_date" in artifact.text
        assert "CODE_SNIPPET" in artifact.text
        assert "python" in artifact.text

    def test_sparse_embedding(self, compiler, sample_atom):
        artifact = compiler.compile(sample_atom, MemoryCompileTarget.SPARSE_EMBEDDING)
        assert artifact.target == MemoryCompileTarget.SPARSE_EMBEDDING
        assert artifact.text.count("Python parse_date 函数") == 2

    def test_agent_profile_menu(self, compiler, agent_profile_atom):
        artifact = compiler.compile(agent_profile_atom, MemoryCompileTarget.AGENT_PROFILE_MENU)
        assert artifact.target == MemoryCompileTarget.AGENT_PROFILE_MENU
        assert "代码分析师" in artifact.text
        assert "**角色**:" in artifact.text
        assert "**能力特长**:" in artifact.text
        assert "<agent_profile" in artifact.text

    def test_agent_profile_menu_english(self, agent_profile_atom):
        set_default_language("en")
        compiler = MemoryCompiler()
        artifact = compiler.compile(agent_profile_atom, MemoryCompileTarget.AGENT_PROFILE_MENU)

        assert "**Role**:" in artifact.text
        assert "**Capabilities**:" in artifact.text
        assert "**角色**:" not in artifact.text

    def test_agent_profile_menu_options_language_overrides_default(self, agent_profile_atom):
        set_default_language("zh")
        compiler = MemoryCompiler()
        artifact = compiler.compile(
            agent_profile_atom,
            MemoryCompileTarget.AGENT_PROFILE_MENU,
            MemoryCompileOptions(language="en"),
        )

        assert "**Role**:" in artifact.text
        assert "**角色**:" not in artifact.text

    def test_agent_profile_untitled_i18n(self, agent_profile_atom):
        agent_profile_atom.index.title = ""

        set_default_language("zh")
        zh_artifact = MemoryCompiler().compile(
            agent_profile_atom,
            MemoryCompileTarget.AGENT_PROFILE_MENU,
        )
        set_default_language("en")
        en_artifact = MemoryCompiler().compile(
            agent_profile_atom,
            MemoryCompileTarget.AGENT_PROFILE_MENU,
        )

        assert "(未命名子代理)" in zh_artifact.text
        assert "(Untitled sub-agent)" in en_artifact.text

    def test_mtp_read(self, compiler, sample_atom):
        artifact = compiler.compile(sample_atom, MemoryCompileTarget.MTP_READ)
        alias = sample_atom.get_alias()
        assert f'<memory alias="{alias}">' in artifact.text
        assert "[完整内容]:" in artifact.text
        assert "parse_date" in artifact.text

    def test_mtp_read_english_full_item(self, sample_atom):
        set_default_language("en")
        compiler = MemoryCompiler()
        artifact = compiler.compile(sample_atom, MemoryCompileTarget.MTP_READ)

        assert "[Full Content]:" in artifact.text
        assert "**Confidence**:" in artifact.text

    def test_mtp_read_with_requested_alias(self, compiler, sample_atom):
        opts = MemoryCompileOptions(requested_alias="my_alias")
        artifact = compiler.compile(sample_atom, MemoryCompileTarget.MTP_READ, opts)
        assert f'<memory alias="{sample_atom.get_alias()}">' in artifact.text
        assert artifact.metadata["requested_alias"] == "my_alias"

    def test_shared_context(self, compiler, sample_atom):
        artifact = compiler.compile(sample_atom, MemoryCompileTarget.SHARED_CONTEXT)
        alias = sample_atom.get_alias()
        assert f'<memory alias="{alias}">' in artifact.text
        assert "[完整内容]:" in artifact.text

    def test_shared_context_english_full_item(self, sample_atom):
        set_default_language("en")
        compiler = MemoryCompiler()
        artifact = compiler.compile(sample_atom, MemoryCompileTarget.SHARED_CONTEXT)

        assert "[Full Content]:" in artifact.text
        assert "**Tags**:" in artifact.text

    def test_runnable_tool_raises(self, compiler, sample_atom):
        with pytest.raises(ValueError, match="reserved"):
            compiler.compile(sample_atom, MemoryCompileTarget.RUNNABLE_TOOL)

    def test_options_max_content_length(self, compiler, sample_atom):
        opts = MemoryCompileOptions(max_content_length=10)
        artifact = compiler.compile(sample_atom, MemoryCompileTarget.PROMPT_FULL, opts)
        assert "截断" in artifact.text or len(artifact.text) < 500

    def test_prompt_full_english_truncation_notice(self, sample_atom):
        set_default_language("en")
        compiler = MemoryCompiler()
        artifact = compiler.compile(
            sample_atom,
            MemoryCompileTarget.PROMPT_FULL,
            MemoryCompileOptions(max_content_length=10),
        )

        assert "content truncated" in artifact.text

    def test_prompt_full_empty_tags_i18n(self, sample_atom):
        sample_atom.index.tags = []

        set_default_language("zh")
        zh_artifact = MemoryCompiler().compile(
            sample_atom,
            MemoryCompileTarget.PROMPT_FULL,
        )
        set_default_language("en")
        en_artifact = MemoryCompiler().compile(
            sample_atom,
            MemoryCompileTarget.PROMPT_FULL,
        )

        assert "(无标签)" in zh_artifact.text
        assert "(No tags)" in en_artifact.text

    def test_list_input(self, compiler, sample_atom, agent_profile_atom):
        artifacts = compiler.compile(
            [sample_atom, agent_profile_atom],
            MemoryCompileTarget.DENSE_EMBEDDING,
        )
        assert isinstance(artifacts, list)
        assert len(artifacts) == 2
        assert all(isinstance(a, CompiledMemoryArtifact) for a in artifacts)


class TestPendingAtomCompilation:
    """测试 PendingAtom 编译。"""

    @pytest.fixture
    def write_pending(self):
        from hivememory.core.models import PendingAtom, PendingAtomStatus, WriteFocus

        return PendingAtom(
            pending_alias="draft_001",
            intent_id="intent_001",
            status=PendingAtomStatus.PENDING,
            source_verb="WRITE",
            focus=WriteFocus(content="Hello world", title="Test Write"),
        )

    @pytest.fixture
    def update_pending(self):
        from hivememory.core.models import PendingAtom, PendingAtomStatus, UpdateFocus

        return PendingAtom(
            pending_alias="rev_001",
            intent_id="intent_002",
            status=PendingAtomStatus.PENDING,
            source_verb="UPDATE",
            focus=UpdateFocus(
                base_alias="fact_api",
                base_uuid="uuid-123",
                instruction="Update the API endpoint",
                content="New content here",
            ),
        )

    def test_pending_mtp_read_draft(self, compiler, write_pending):
        artifact = compiler.compile(write_pending, MemoryCompileTarget.MTP_READ)
        assert artifact.source_kind == "pending"
        assert "draft_001" in artifact.text
        assert "Hello world" in artifact.text
        assert "pending" in artifact.text.lower()

    def test_materializing_mtp_read_draft(self, compiler, write_pending):
        from hivememory.core.models import PendingAtomStatus

        write_pending.status = PendingAtomStatus.MATERIALIZING
        artifact = compiler.compile(write_pending, MemoryCompileTarget.MTP_READ)
        assert "draft_001" in artifact.text
        assert "materializing" in artifact.text.lower()

    def test_pending_mtp_read_revision(self, compiler, update_pending):
        artifact = compiler.compile(update_pending, MemoryCompileTarget.MTP_READ)
        assert "rev_001" in artifact.text
        assert "pending revision" in artifact.text.lower()

    def test_pending_mtp_read_english_option(self, compiler, write_pending):
        artifact = compiler.compile(
            write_pending,
            MemoryCompileTarget.MTP_READ,
            MemoryCompileOptions(language="en"),
        )
        assert "runtime pending atom" in artifact.text
        assert "draft_001" in artifact.text

    def test_pending_shared_context(self, compiler, write_pending):
        artifact = compiler.compile(write_pending, MemoryCompileTarget.SHARED_CONTEXT)
        assert "draft_001" in artifact.text
        assert "Hello world" in artifact.text

    def test_pending_unsupported_target(self, compiler, write_pending):
        with pytest.raises(ValueError):
            compiler.compile(write_pending, MemoryCompileTarget.DENSE_EMBEDDING)

    @pytest.fixture
    def settled_pending(self):
        from hivememory.core.models import (
            PendingAtom, PendingAtomStatus, WriteFocus,
            PendingAtomSettlement, PendingAtomResolution,
        )
        atom = PendingAtom(
            pending_alias="draft_settled",
            intent_id="intent_s",
            status=PendingAtomStatus.SETTLED,
            source_verb="WRITE",
            focus=WriteFocus(content="Hello", title="T"),
        )
        atom.settlement = PendingAtomSettlement(
            pending_alias="draft_settled",
            intent_id="intent_s",
            resolution=PendingAtomResolution.CREATED,
            canonical_alias="fact_hello",
            canonical_uuid="uuid-s",
        )
        return atom

    @pytest.fixture
    def failed_pending(self):
        from hivememory.core.models import (
            PendingAtom, PendingAtomStatus, WriteFocus,
            PendingAtomSettlement, PendingAtomResolution,
        )
        atom = PendingAtom(
            pending_alias="draft_failed",
            intent_id="intent_f",
            status=PendingAtomStatus.FAILED,
            source_verb="WRITE",
            focus=WriteFocus(content="X"),
        )
        atom.settlement = PendingAtomSettlement(
            pending_alias="draft_failed",
            intent_id="intent_f",
            resolution=PendingAtomResolution.CREATED,
            error="generation pipeline error",
        )
        return atom

    @pytest.fixture
    def cancelled_pending(self):
        from hivememory.core.models import PendingAtom, PendingAtomStatus, WriteFocus
        return PendingAtom(
            pending_alias="draft_cancelled",
            intent_id="intent_c",
            status=PendingAtomStatus.CANCELLED,
            source_verb="WRITE",
            focus=WriteFocus(content="X"),
        )

    @pytest.fixture
    def expired_pending(self):
        from hivememory.core.models import PendingAtom, PendingAtomStatus, WriteFocus
        return PendingAtom(
            pending_alias="draft_expired",
            intent_id="intent_e",
            status=PendingAtomStatus.EXPIRED,
            source_verb="WRITE",
            focus=WriteFocus(content="X"),
        )

    def test_settled_mtp_read_shows_canonical(self, compiler, settled_pending):
        artifact = compiler.compile(settled_pending, MemoryCompileTarget.MTP_READ)
        assert "settled" in artifact.text.lower()
        assert "fact_hello" in artifact.text

    def test_settled_mtp_read_empty_canonical_uses_placeholder(self, compiler, settled_pending):
        settled_pending.settlement = settled_pending.settlement.model_copy(
            update={"canonical_alias": None}
        )

        zh_artifact = compiler.compile(settled_pending, MemoryCompileTarget.MTP_READ)
        en_artifact = compiler.compile(
            settled_pending,
            MemoryCompileTarget.MTP_READ,
            MemoryCompileOptions(language="en"),
        )

        assert "\u6b63\u5f0f\u540d\u79f0\uff1a\u65e0" in zh_artifact.text
        assert "Canonical alias: None" in en_artifact.text

    def test_pending_mtp_read_empty_title_uses_placeholder(self, compiler, write_pending):
        write_pending.focus = write_pending.focus.model_copy(update={"title": ""})

        zh_artifact = compiler.compile(write_pending, MemoryCompileTarget.MTP_READ)
        en_artifact = compiler.compile(
            write_pending,
            MemoryCompileTarget.MTP_READ,
            MemoryCompileOptions(language="en"),
        )

        assert "\u6807\u9898\uff1a\u65e0" in zh_artifact.text
        assert "title: None" in en_artifact.text

    def test_failed_mtp_read_shows_error(self, compiler, failed_pending):
        artifact = compiler.compile(failed_pending, MemoryCompileTarget.MTP_READ)
        assert "failed" in artifact.text.lower()
        assert "generation pipeline error" in artifact.text

    def test_cancelled_mtp_read(self, compiler, cancelled_pending):
        artifact = compiler.compile(cancelled_pending, MemoryCompileTarget.MTP_READ)
        assert "cancelled" in artifact.text.lower()

    def test_expired_mtp_read_shows_reclaimed(self, compiler, expired_pending):
        artifact = compiler.compile(expired_pending, MemoryCompileTarget.MTP_READ)
        assert "expired" in artifact.text.lower()
        assert "reclaimed" in artifact.text.lower()


class TestResolveResultCompilation:
    """测试 ResolveResult 编译。"""

    @pytest.fixture
    def atom_resolve(self, sample_atom):
        from hivememory.agent_runtime.resolver import ResolveResult

        return ResolveResult(
            kind="atom",
            requested_alias="fact_api",
            atom=sample_atom,
        )

    @pytest.fixture
    def redirect_resolve(self, sample_atom):
        from hivememory.agent_runtime.resolver import ResolveResult
        from hivememory.core.models import (
            PendingAtomResolution,
            PendingAtomSettlement,
        )

        return ResolveResult(
            kind="redirect",
            requested_alias="draft_old",
            canonical_alias="fact_api",
            atom=sample_atom,
            settlement=PendingAtomSettlement(
                pending_alias="draft_old",
                intent_id="intent-1",
                resolution=PendingAtomResolution.CREATED,
                canonical_alias="fact_api",
            ),
        )

    @pytest.fixture
    def discarded_resolve(self):
        from hivememory.agent_runtime.resolver import ResolveResult
        from hivememory.core.models import (
            PendingAtomResolution,
            PendingAtomSettlement,
        )

        return ResolveResult(
            kind="discarded",
            requested_alias="draft_bad",
            settlement=PendingAtomSettlement(
                pending_alias="draft_bad",
                intent_id="intent-2",
                resolution=PendingAtomResolution.DISCARDED,
                message="Duplicate content",
                reason="Merged with existing memory",
            ),
        )

    @pytest.fixture
    def failed_resolve(self):
        from hivememory.agent_runtime.resolver import ResolveResult
        from hivememory.core.models import (
            PendingAtomResolution,
            PendingAtomSettlement,
        )

        return ResolveResult(
            kind="failed",
            requested_alias="draft_failed",
            settlement=PendingAtomSettlement(
                pending_alias="draft_failed",
                intent_id="intent-3",
                resolution=PendingAtomResolution.CREATED,
                error="generation failed",
            ),
        )

    @pytest.fixture
    def expired_resolve(self):
        from hivememory.agent_runtime.resolver import ResolveResult

        return ResolveResult(kind="expired", requested_alias="draft_expired")

    @pytest.fixture
    def pending_resolve(self):
        from hivememory.agent_runtime.resolver import ResolveResult
        from hivememory.core.models import PendingAtom, PendingAtomStatus, WriteFocus

        pending = PendingAtom(
            pending_alias="draft_002",
            intent_id="intent_003",
            status=PendingAtomStatus.PENDING,
            source_verb="WRITE",
            focus=WriteFocus(content="Pending content", title="Pending"),
        )
        return ResolveResult(
            kind="pending",
            requested_alias="draft_002",
            pending=pending,
        )

    def test_atom_resolve_mtp_read(self, compiler, atom_resolve):
        artifact = compiler.compile(atom_resolve, MemoryCompileTarget.MTP_READ)
        assert "parse_date" in artifact.text

    def test_atom_resolve_prompt_full(self, compiler, atom_resolve):
        artifact = compiler.compile(atom_resolve, MemoryCompileTarget.PROMPT_FULL)
        assert "<memory" in artifact.text

    def test_redirect_mtp_read(self, compiler, redirect_resolve):
        opts = MemoryCompileOptions(requested_alias="draft_old")
        artifact = compiler.compile(redirect_resolve, MemoryCompileTarget.MTP_READ, opts)
        assert "Alias Redirected" in artifact.text
        assert "draft_old" in artifact.text
        assert "fact_api" in artifact.text

    def test_redirect_shared_context(self, compiler, redirect_resolve):
        artifact = compiler.compile(redirect_resolve, MemoryCompileTarget.SHARED_CONTEXT)
        assert artifact.alias == "fact_api"
        assert "parse_date" in artifact.text
        assert "<memory alias=" in artifact.text

    def test_discarded_mtp_read(self, compiler, discarded_resolve):
        opts = MemoryCompileOptions(requested_alias="draft_bad")
        artifact = compiler.compile(discarded_resolve, MemoryCompileTarget.MTP_READ, opts)
        assert "discarded" in artifact.text.lower()
        assert "draft_bad" in artifact.text

    def test_failed_mtp_read(self, compiler, failed_resolve):
        opts = MemoryCompileOptions(requested_alias="draft_failed")
        artifact = compiler.compile(failed_resolve, MemoryCompileTarget.MTP_READ, opts)
        assert "failed" in artifact.text.lower()
        assert "generation failed" in artifact.text
        assert "\u6d88\u606f\uff1a\u65e0" in artifact.text
        assert "\u539f\u56e0\uff1a\u65e0" in artifact.text

    def test_expired_mtp_read(self, compiler, expired_resolve):
        opts = MemoryCompileOptions(requested_alias="draft_expired")
        artifact = compiler.compile(expired_resolve, MemoryCompileTarget.MTP_READ, opts)
        assert "expired" in artifact.text.lower()
        assert "reclaimed" in artifact.text.lower()
        assert "Alias Not Found" not in artifact.text

    def test_expired_mtp_read_english_option(self, compiler, expired_resolve):
        artifact = compiler.compile(
            expired_resolve,
            MemoryCompileTarget.MTP_READ,
            MemoryCompileOptions(requested_alias="draft_expired", language="en"),
        )
        assert "expired" in artifact.text.lower()
        assert "draft_expired" in artifact.text

    def test_pending_resolve_mtp_read(self, compiler, pending_resolve):
        artifact = compiler.compile(pending_resolve, MemoryCompileTarget.MTP_READ)
        assert "draft_002" in artifact.text
        assert "Pending content" in artifact.text

    def test_not_found_raises(self, compiler):
        from hivememory.agent_runtime.resolver import ResolveResult

        resolve = ResolveResult(kind="not_found", requested_alias="missing")
        with pytest.raises(ValueError, match="not_found"):
            compiler.compile(resolve, MemoryCompileTarget.MTP_READ)


class TestInvalidInputs:
    """测试无效输入。"""

    def test_unsupported_source_type(self, compiler):
        with pytest.raises(TypeError, match="Unsupported source type"):
            compiler.compile("not a memory", MemoryCompileTarget.MTP_READ)

    def test_none_source(self, compiler):
        with pytest.raises(TypeError):
            compiler.compile(None, MemoryCompileTarget.MTP_READ)

    def test_compiled_memory_source_rejected(self, compiler, sample_atom):
        artifact = compiler.compile(sample_atom, MemoryCompileTarget.PROMPT_FULL)

        with pytest.raises(TypeError, match="internal MemoryCompiler intermediates"):
            compiler.compile(artifact, MemoryEnvelopeTarget.RETRIEVAL_CONTEXT)


class TestEnvelopeCompilation:
    """测试 envelope 编译。"""

    def test_retrieval_context_compile_atoms(self, compiler, sample_atom, agent_profile_atom):
        envelope = compiler.compile(
            [sample_atom, agent_profile_atom],
            MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
        )

        assert envelope.target == MemoryEnvelopeTarget.RETRIEVAL_CONTEXT
        assert "<memory_context>" in envelope.text
        assert "相关记忆" in envelope.text
        assert "可用子代理" in envelope.text
        assert "Python parse_date" in envelope.text
        assert "代码分析师" in envelope.text
        assert isinstance(envelope.sections[0], MemoryEnvelopeSection)

    def test_retrieval_context_compile_agent_profile_only(self, compiler, agent_profile_atom):
        envelope = compiler.compile(
            agent_profile_atom,
            MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
        )

        assert "可用子代理" in envelope.text
        assert "代码分析师" in envelope.text
        assert "相关记忆" not in envelope.text

    def test_compile_envelope_from_ir_empty_section_hint(self, compiler, agent_profile_atom):
        agent_artifact = compiler.compile(agent_profile_atom, MemoryCompileTarget.AGENT_PROFILE_MENU)
        bundle = MemoryBundleIR(
            purpose=MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
            sections=[
                MemorySectionIR(kind="memories", empty_text="No memories"),
                MemorySectionIR(kind="agent_profiles", artifacts=[agent_artifact]),
            ],
        )
        envelope = compile_envelope_from_ir(
            bundle,
            options=MemoryCompileOptions(language="zh"),
        )

        assert "相关记忆" in envelope.text
        assert "No memories" in envelope.text
        assert "可用子代理" in envelope.text

    def test_retrieval_context_compile_uses_english_default_language(self, sample_atom, agent_profile_atom):
        set_default_language("en")
        compiler = MemoryCompiler()

        envelope = compiler.compile(
            [sample_atom, agent_profile_atom],
            MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
        )

        assert "Patchouli, the memory library manager" in envelope.text
        assert "### Relevant Memories" in envelope.text
        assert "### Available Sub-Agents" in envelope.text
        assert "Use common sense when judging them" in envelope.text

    def test_retrieval_context_compile_options_language_overrides_default(self, sample_atom):
        set_default_language("zh")
        compiler = MemoryCompiler()

        envelope = compiler.compile(
            sample_atom,
            MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
            options=MemoryCompileOptions(language="en"),
        )

        assert "Patchouli, the memory library manager" in envelope.text
        assert "相关记忆" not in envelope.text
        assert envelope.sections[0].kind == "memories"
        assert isinstance(envelope.sections[0], MemoryEnvelopeSection)

    def test_mtp_read_response_compile(self, compiler, sample_atom):
        envelope = compiler.compile(
            sample_atom,
            MemoryEnvelopeTarget.MTP_READ_RESPONSE,
        )

        assert envelope.text.startswith("[MTP READ Result]")
        assert "Python parse_date" in envelope.text

    def test_mtp_read_response_compile_english(self, sample_atom):
        set_default_language("en")
        compiler = MemoryCompiler()

        envelope = compiler.compile(
            sample_atom,
            MemoryEnvelopeTarget.MTP_READ_RESPONSE,
        )

        assert envelope.text.startswith("[MTP READ Result]")
        assert "Python parse_date" in envelope.text

    def test_shared_context_injection_compile(self, compiler, sample_atom):
        envelope = compiler.compile(
            sample_atom,
            MemoryEnvelopeTarget.SHARED_CONTEXT_INJECTION,
        )

        assert envelope.text.startswith("[Shared Context from Parent Agent]")
        assert "READ" in envelope.text
        assert "Python parse_date" in envelope.text

    def test_shared_context_injection_compile_english(self, sample_atom):
        set_default_language("en")
        compiler = MemoryCompiler()

        envelope = compiler.compile(
            sample_atom,
            MemoryEnvelopeTarget.SHARED_CONTEXT_INJECTION,
        )

        assert envelope.text.startswith("[Shared Context from Parent Agent]")
        assert "The parent agent shared" in envelope.text
        assert "Use READ" in envelope.text
        assert "Python parse_date" in envelope.text

    def test_shared_context_injection_empty_default_chinese(self, compiler):
        envelope = compiler.compile(
            [],
            MemoryEnvelopeTarget.SHARED_CONTEXT_INJECTION,
        )

        assert envelope.text.startswith("[Shared Context from Parent Agent]")
        assert "没有共享的记忆材料" in envelope.text
        assert envelope.sections == []

    def test_shared_context_injection_empty_english(self):
        set_default_language("en")
        compiler = MemoryCompiler()

        envelope = compiler.compile(
            [],
            MemoryEnvelopeTarget.SHARED_CONTEXT_INJECTION,
        )

        assert envelope.text.startswith("[Shared Context from Parent Agent]")
        assert "No shared memory artifacts." in envelope.text

    def test_compile_envelope_accepts_retrieval_bundle_ir(self, compiler, sample_atom):
        artifact = compiler.compile(sample_atom, MemoryCompileTarget.PROMPT_FULL)
        bundle = MemoryBundleIR(
            purpose=MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
            sections=[
                MemorySectionIR(kind="memories", artifacts=[artifact]),
                MemorySectionIR(kind="agent_profiles", empty_text="No agents"),
            ],
        )

        envelope = compile_envelope_from_ir(bundle, options=MemoryCompileOptions(language="en"))

        assert envelope.target == MemoryEnvelopeTarget.RETRIEVAL_CONTEXT
        assert "### Relevant Memories" in envelope.text
        assert "No agents" in envelope.text
        assert isinstance(envelope.sections[0], MemoryEnvelopeSection)

    def test_compile_envelope_accepts_mtp_read_bundle_ir(self, compiler, sample_atom):
        artifact = compiler.compile(sample_atom, MemoryCompileTarget.MTP_READ)
        bundle = MemoryBundleIR(
            purpose=MemoryEnvelopeTarget.MTP_READ_RESPONSE,
            sections=[MemorySectionIR(kind="default", artifacts=[artifact])],
        )

        envelope = compile_envelope_from_ir(bundle)

        assert envelope.text.startswith("[MTP READ Result]")
        assert "Python parse_date" in envelope.text

    def test_compile_envelope_accepts_shared_context_bundle_ir(self):
        bundle = MemoryBundleIR(
            purpose=MemoryEnvelopeTarget.SHARED_CONTEXT_INJECTION,
            sections=[],
        )

        envelope = compile_envelope_from_ir(bundle, options=MemoryCompileOptions(language="en"))

        assert envelope.text.startswith("[Shared Context from Parent Agent]")
        assert "No shared memory artifacts." in envelope.text
