import pytest
import asyncio
from unittest.mock import patch, MagicMock
from hivememory.patchouli.system import PatchouliSystem
from hivememory.patchouli.protocol.models import RetrievalResponse

@pytest.fixture
def patch_assemble_messages():
    """
    一个可以挂载到任意测试中的 fixture，
    自动拦截 _assemble_messages_from_context 方法并打印 system_prompt 和 user_prompt
    """
    original_assemble = PatchouliSystem._assemble_messages_from_context

    def _intercept(self, topic_context, hot_result, user_message):
        messages = original_assemble(self, topic_context, hot_result, user_message)
        
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

    with patch.object(PatchouliSystem, '_assemble_messages_from_context', new=_intercept):
        yield

# 示例测试：
@pytest.mark.asyncio
async def test_debug_messages_with_system(patch_assemble_messages):
    """
    只要使用 patch_assemble_messages fixture，
    调用任何触发 chat 或 chat_stream 的代码都会在控制台打印提示词。
    """
    # Mock 存储层和检索层，避免真实调用 Qdrant
    with patch('hivememory.infrastructure.storage.QdrantMemoryStore') as mock_store_cls, \
         patch('hivememory.patchouli.kernel.retrieval_familiar.RetrievalFamiliar.retrieve') as mock_retrieve:

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

        # 实例化 PatchouliSystem，自动加载默认配置
        system = PatchouliSystem()

        # 这里的调用会被拦截并打印
        result = await system.chat(
            user_message="测试拦截功能",
            user_id="test_user"
        )

        assert result is not None
