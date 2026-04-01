import asyncio
from unittest.mock import patch

from hivememory.patchouli.system import PatchouliSystem

# 1. 保存原始方法
original_assemble = PatchouliSystem._assemble_messages_from_context

def debug_assemble_messages(self, topic_context, hot_result, user_message):
    """
    拦截 _assemble_messages_from_context 方法，
    在返回组装好的 messages 前打印出 system_prompt 和 user_prompt
    """
    # 调用原始方法获取真实的 messages 列表
    messages = original_assemble(self, topic_context, hot_result, user_message)
    
    print("\n" + "="*60)
    print("🚀 [DEBUG] Intercepted Messages Before Kernel Loop")
    print("="*60)
    
    # 提取 System Prompt (通常在索引 0 的位置)
    system_messages = [msg['content'] for msg in messages if msg['role'] == 'system']
    if system_messages:
        print("\n🔹 [System Prompt]:")
        print("-" * 40)
        print(system_messages[0])
        print("-" * 40)
    else:
        print("\n🔹 [System Prompt]: None found")
        
    # 提取所有历史对话记录作为辅助信息(可选)
    history_messages = [msg for msg in messages if msg['role'] not in ('system',)]
    if len(history_messages) > 1:
        print(f"\n[History]: {len(history_messages) - 1} turns of conversation included.")
    
    # 提取当前 User Prompt (最后一个 user message)
    user_messages = [msg['content'] for msg in messages if msg['role'] == 'user']
    if user_messages:
        print("\n🔸 [Current User Prompt]:")
        print("-" * 40)
        print(user_messages[-1])
        print("-" * 40)
        
    print("="*60 + "\n")
    
    return messages

async def main():
    print("初始化系统...")
    # 实例化 PatchouliSystem，自动加载默认配置
    system = PatchouliSystem()
    
    print("\n开始测试对话 (使用拦截器)...")
    # 2. 使用 patch.object 临时替换方法
    with patch.object(PatchouliSystem, '_assemble_messages_from_context', new=debug_assemble_messages):
        # 触发 chat 或 chat_stream
        # 这里演示调用普通的 chat，chat_stream 同理，因为底层都调用 _assemble_messages_from_context
        try:
            result = await system.chat(
                user_message="我是一个正在学习做饭的新手，我想了解一下怎么样才能做出一份标准又好吃的红烧羊肉？",
                user_id="test_user",
                agent_id="koakuma"
            )
            print(f"\n✅ [Response]: {result.final_text}")
        except Exception as e:
            print(f"执行出错 (可能是没有配置正确的环境变量): {e}")

if __name__ == "__main__":
    asyncio.run(main())
