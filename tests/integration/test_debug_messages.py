import pytest
import asyncio
from unittest.mock import patch, MagicMock
from hivememory.patchouli.service import PatchouliService
from hivememory.core.protocol.models import RetrievalResponse
from hivememory.system.config import load_app_config
from hivememory.system import HiveMemorySystem

@pytest.fixture
def patch_assemble_messages():
    """
    一个可以挂载到任意测试中的 fixture，
    自动拦截 _assemble_messages_from_context 方法并打印 system_prompt 和 user_prompt
    """
    original_assemble = PatchouliService._assemble_messages_from_context

    def _intercept(self, topic_context, retrieval_result, user_message, profile=None, current_agent_id="omni_doll"):
        messages = original_assemble(
            self,
            topic_context,
            retrieval_result,
            user_message,
            profile=profile,
            current_agent_id=current_agent_id,
        )
        
        print("\n" + "="*50)
        print("🚀 [TEST DEBUG] Intercepted Messages Before Kernel Loop")
        print("="*50)
        
        system_prompts = [msg['content'] for msg in messages if msg['role'] == 'system']
        if system_prompts:
            print("\n🔹 [System Prompt]:")
            print("-" * 40)
            print(system_prompts[0])
            print("-" * 40)
        
        user_prompts = [msg['content'] for msg in messages if msg['role'] == 'user']
        if user_prompts:
            print("\n🔸 [Current User Prompt]:")
            print("-" * 40)
            print(user_prompts[-1])
            print("-" * 40)
            
        print("="*50 + "\n")
        return messages

    with patch.object(PatchouliService, '_assemble_messages_from_context', new=_intercept):
        yield

# 示例测试：
@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_debug_messages_with_system(patch_assemble_messages):
    """
    只要使用 patch_assemble_messages fixture，
    调用任何触发 chat 或 chat_stream 的代码都会在控制台打印提示词。
    """
    # Mock 存储层和检索层，避免真实调用 Qdrant
    with patch('hivememory.infrastructure.storage.QdrantMemoryStore') as mock_store_cls, \
         patch('hivememory.patchouli.services.retrieval.RetrievalFamiliar.retrieve') as mock_retrieve:

        # Mock 存储实例
        mock_store = MagicMock()
        mock_store.health_check.return_value = True
        mock_store.embedding_service.is_loaded.return_value = True
        mock_store.embedding_service.warmup.return_value = None
        mock_store_cls.return_value = mock_store

        # Mock 检索返回空结果
        mock_retrieve.return_value = RetrievalResponse(
            memories=[],
            rendered_context=""
        )

        # 实例化顶层系统，使用 Phase D 主入口
        system = HiveMemorySystem.build(config=load_app_config())
        await system.start()

        try:
            # 这里的调用会被拦截并打印
            result = await system.chat(
                user_message="测试拦截功能",
                user_id="test_user"
            )
        finally:
            await system.stop()

        assert result is not None
