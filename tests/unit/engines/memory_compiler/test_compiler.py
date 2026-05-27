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
        assert artifact.memory_id == str(sample_atom.id)

    def test_prompt_index(self, compiler, sample_atom):
        artifact = compiler.compile(sample_atom, MemoryCompileTarget.PROMPT_INDEX)
        assert artifact.target == MemoryCompileTarget.PROMPT_INDEX
        assert "<memory_index" in artifact.text
        assert "基于 datetime 库" in artifact.text

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
        assert "<agent_profile" in artifact.text

    def test_mtp_read(self, compiler, sample_atom):
        artifact = compiler.compile(sample_atom, MemoryCompileTarget.MTP_READ)
        alias = sample_atom.get_alias()
        assert f'<memory alias="{alias}">' in artifact.text
        assert "[完整内容]:" in artifact.text
        assert "parse_date" in artifact.text

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

    def test_runnable_tool_raises(self, compiler, sample_atom):
        with pytest.raises(ValueError, match="reserved"):
            compiler.compile(sample_atom, MemoryCompileTarget.RUNNABLE_TOOL)

    def test_options_max_content_length(self, compiler, sample_atom):
        opts = MemoryCompileOptions(max_content_length=10)
        artifact = compiler.compile(sample_atom, MemoryCompileTarget.PROMPT_FULL, opts)
        assert "截断" in artifact.text or len(artifact.text) < 500

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
        from hivememory.alice.runtime.models import PendingAtom, PendingAtomStatus
        from hivememory.engines.generation.models import WriteFocus

        return PendingAtom(
            pending_alias="draft_001",
            status=PendingAtomStatus.PENDING,
            source_verb="WRITE",
            focus=WriteFocus(content="Hello world", title="Test Write"),
        )

    @pytest.fixture
    def update_pending(self):
        from hivememory.alice.runtime.models import PendingAtom, PendingAtomStatus
        from hivememory.engines.generation.models import UpdateFocus

        return PendingAtom(
            pending_alias="rev_001",
            status=PendingAtomStatus.REVISION,
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

    def test_pending_mtp_read_revision(self, compiler, update_pending):
        artifact = compiler.compile(update_pending, MemoryCompileTarget.MTP_READ)
        assert "rev_001" in artifact.text
        assert "revision" in artifact.text.lower()

    def test_pending_mtp_ack_write(self, compiler, write_pending):
        artifact = compiler.compile(write_pending, MemoryCompileTarget.MTP_ACK)
        assert artifact.target == MemoryCompileTarget.MTP_ACK
        assert "draft_001" in artifact.text
        assert "pending" in artifact.text.lower()

    def test_pending_mtp_ack_update(self, compiler, update_pending):
        artifact = compiler.compile(update_pending, MemoryCompileTarget.MTP_ACK)
        assert "rev_001" in artifact.text
        assert "revision" in artifact.text.lower()

    def test_pending_shared_context(self, compiler, write_pending):
        artifact = compiler.compile(write_pending, MemoryCompileTarget.SHARED_CONTEXT)
        assert "draft_001" in artifact.text
        assert "Hello world" in artifact.text

    def test_pending_unsupported_target(self, compiler, write_pending):
        with pytest.raises(ValueError):
            compiler.compile(write_pending, MemoryCompileTarget.DENSE_EMBEDDING)


class TestResolveResultCompilation:
    """测试 ResolveResult 编译。"""

    @pytest.fixture
    def atom_resolve(self, sample_atom):
        from hivememory.alice.runtime.resolver import ResolveResult

        return ResolveResult(
            kind="atom",
            requested_alias="fact_api",
            atom=sample_atom,
        )

    @pytest.fixture
    def redirect_resolve(self, sample_atom):
        from hivememory.alice.runtime.resolver import ResolveResult
        from hivememory.engines.generation.models import PendingAtomSettlement

        return ResolveResult(
            kind="redirect",
            requested_alias="draft_old",
            canonical_alias="fact_api",
            atom=sample_atom,
            settlement=PendingAtomSettlement(
                pending_alias="draft_old",
                intent_id="intent-1",
                status="COMMITTED",
                canonical_alias="fact_api",
            ),
        )

    @pytest.fixture
    def discarded_resolve(self):
        from hivememory.alice.runtime.resolver import ResolveResult
        from hivememory.engines.generation.models import PendingAtomSettlement

        return ResolveResult(
            kind="discarded",
            requested_alias="draft_bad",
            settlement=PendingAtomSettlement(
                pending_alias="draft_bad",
                intent_id="intent-2",
                status="DISCARDED",
                message="Duplicate content",
                reason="Merged with existing memory",
            ),
        )

    @pytest.fixture
    def pending_resolve(self):
        from hivememory.alice.runtime.resolver import ResolveResult
        from hivememory.alice.runtime.models import PendingAtom, PendingAtomStatus
        from hivememory.engines.generation.models import WriteFocus

        pending = PendingAtom(
            pending_alias="draft_002",
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

    def test_redirect_notice(self, compiler, redirect_resolve):
        opts = MemoryCompileOptions(requested_alias="draft_old")
        artifact = compiler.compile(redirect_resolve, MemoryCompileTarget.MTP_REDIRECT_NOTICE, opts)
        assert "Alias Redirected" in artifact.text
        assert "RUN" in artifact.text

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

    def test_pending_resolve_mtp_read(self, compiler, pending_resolve):
        artifact = compiler.compile(pending_resolve, MemoryCompileTarget.MTP_READ)
        assert "draft_002" in artifact.text
        assert "Pending content" in artifact.text

    def test_not_found_raises(self, compiler):
        from hivememory.alice.runtime.resolver import ResolveResult

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


class TestEnvelopeCompilation:
    """测试 envelope 包装。"""

    def test_retrieval_context_wrap(self, compiler, sample_atom, agent_profile_atom):
        memory_artifact = compiler.compile(sample_atom, MemoryCompileTarget.PROMPT_FULL)
        agent_artifact = compiler.compile(agent_profile_atom, MemoryCompileTarget.AGENT_PROFILE_MENU)

        envelope = compiler.wrap(
            envelope_target=MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
            sections=[
                MemoryEnvelopeSection(kind="memories", artifacts=[memory_artifact]),
                MemoryEnvelopeSection(kind="agent_profiles", artifacts=[agent_artifact]),
            ],
        )

        assert envelope.target == MemoryEnvelopeTarget.RETRIEVAL_CONTEXT
        assert "<memory_context>" in envelope.text
        assert "相关记忆" in envelope.text
        assert "可用子代理" in envelope.text
        assert "Python parse_date" in envelope.text
        assert "代码分析师" in envelope.text

    def test_retrieval_context_empty_section_hint(self, compiler, agent_profile_atom):
        agent_artifact = compiler.compile(agent_profile_atom, MemoryCompileTarget.AGENT_PROFILE_MENU)

        envelope = compiler.wrap(
            envelope_target=MemoryEnvelopeTarget.RETRIEVAL_CONTEXT,
            sections=[
                MemoryEnvelopeSection(kind="memories", empty_text="No memories"),
                MemoryEnvelopeSection(kind="agent_profiles", artifacts=[agent_artifact]),
            ],
        )

        assert "相关记忆" in envelope.text
        assert "No memories" in envelope.text
        assert "可用子代理" in envelope.text

    def test_mtp_read_response_wrap(self, compiler, sample_atom):
        artifact = compiler.compile(sample_atom, MemoryCompileTarget.MTP_READ)
        envelope = compiler.wrap(
            artifact,
            envelope_target=MemoryEnvelopeTarget.MTP_READ_RESPONSE,
        )

        assert envelope.text.startswith("[MTP READ Result]")
        assert "Python parse_date" in envelope.text

    def test_shared_context_injection_wrap(self, compiler, sample_atom):
        artifact = compiler.compile(sample_atom, MemoryCompileTarget.SHARED_CONTEXT)
        envelope = compiler.wrap(
            artifact,
            envelope_target=MemoryEnvelopeTarget.SHARED_CONTEXT_INJECTION,
        )

        assert envelope.text.startswith("[Shared Context from Parent Agent]")
        assert "READ" in envelope.text
        assert "Python parse_date" in envelope.text
