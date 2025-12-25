"""
HiveMemory ChatBot Streamlit Web UI

简单的聊天界面，用于测试帕秋莉的记忆生成功能

运行方式:
    streamlit run examples/chatbot_ui.py

功能:
1. 类 ChatGPT 的对话界面
2. 会话管理（支持清空会话）
3. 实时显示当前配置
4. 自动将对话推送给帕秋莉进行记忆生成
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import streamlit as st
import uuid
import redis
from datetime import datetime

from hivememory.core.config import get_config
from hivememory.memory.storage import QdrantMemoryStore
from hivememory.agents.patchouli import PatchouliAgent
from hivememory.agents.chatbot import ChatBotAgent
from hivememory.agents.session_manager import SessionManager


# 页面配置
st.set_page_config(
    page_title="HiveMemory ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def initialize_system():
    """初始化系统组件（缓存，避免重复初始化）"""
    # 加载配置
    config = get_config()

    # 初始化 Redis
    redis_client = redis.Redis(
        **config.redis.model_dump(),
        socket_connect_timeout=5
    )

    # 初始化 Qdrant Storage
    storage = QdrantMemoryStore(
        qdrant_config=config.qdrant,
        embedding_config=config.embedding
    )

    # 初始化 Patchouli Agent（图书管理员）
    patchouli = PatchouliAgent(storage=storage)

    # 初始化 Session Manager
    session_manager = SessionManager(
        redis_client=redis_client,
        key_prefix="hivememory:session",
        ttl_days=7
    )

    return config, patchouli, session_manager, storage


def init_session_state():
    """初始化 Streamlit session state"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]

    if "user_id" not in st.session_state:
        st.session_state.user_id = "demo_user"

    if "messages_displayed" not in st.session_state:
        st.session_state.messages_displayed = []

    if "chatbot_agent" not in st.session_state:
        config, patchouli, session_manager, _ = initialize_system()

        # 获取 Worker LLM 配置
        worker_llm_config = config.get_worker_llm_config()

        # 创建 ChatBot Agent
        st.session_state.chatbot_agent = ChatBotAgent(
            patchouli=patchouli,
            session_manager=session_manager,
            user_id=st.session_state.user_id,
            agent_id="streamlit_chatbot",
            llm_config={
                "model": worker_llm_config.model,
                "api_key": worker_llm_config.api_key,
                "api_base": worker_llm_config.api_base,
                "temperature": worker_llm_config.temperature,
                "max_tokens": worker_llm_config.max_tokens
            }
        )


def load_session_history():
    """从 SessionManager 加载历史消息"""
    config, _, session_manager, _ = initialize_system()

    history = session_manager.get_history(st.session_state.session_id)

    # 转换为 Streamlit 显示格式
    st.session_state.messages_displayed = [
        {"role": msg.role, "content": msg.content}
        for msg in history
    ]


def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.title("⚙️ 系统设置")

        # 会话信息
        st.subheader("📋 当前会话")
        st.text(f"Session ID: {st.session_state.session_id}")
        st.text(f"User ID: {st.session_state.user_id}")

        # 会话统计
        agent: ChatBotAgent = st.session_state.chatbot_agent
        session_info = agent.get_session_info(st.session_state.session_id)
        st.metric("消息数量", session_info["message_count"])

        # 清空会话按钮
        if st.button("🗑️ 清空会话", use_container_width=True):
            agent.clear_session(st.session_state.session_id)
            st.session_state.messages_displayed = []
            st.rerun()

        # 新建会话按钮
        if st.button("➕ 新建会话", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.session_state.messages_displayed = []
            st.rerun()

        st.divider()

        # LLM 配置信息
        st.subheader("🤖 LLM 配置")
        config, _, _, _ = initialize_system()
        worker_llm = config.get_worker_llm_config()

        st.text(f"模型: {worker_llm.model}")
        st.text(f"温度: {worker_llm.temperature}")
        st.text(f"最大 Tokens: {worker_llm.max_tokens}")

        st.divider()

        # 帕秋莉配置
        st.subheader("📚 帕秋莉配置")
        st.text(f"触发阈值: {config.memory.buffer.max_messages} 条消息")
        st.text(f"空闲触发: {config.memory.buffer.timeout_seconds // 60} 分钟")
        st.text(f"最低置信度: {config.memory.extraction.min_confidence}")

        st.divider()

        # 说明
        st.caption("""
        **💡 使用说明**

        1. 在下方输入框发送消息
        2. ChatBot 会自动回复
        3. 对话会被推送给帕秋莉
        4. 每 5 条消息或 15 分钟空闲后，帕秋莉会自动提取记忆
        5. 记忆将存储到 Qdrant 数据库
        """)


def render_chat_interface():
    """渲染聊天界面"""
    st.title("🤖 HiveMemory ChatBot")
    st.caption("与 AI 助手对话，帕秋莉会自动提取并存储有价值的记忆")

    # 显示历史消息
    for message in st.session_state.messages_displayed:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 用户输入
    if prompt := st.chat_input("输入消息..."):
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages_displayed.append({"role": "user", "content": prompt})

        # 调用 ChatBot Agent
        agent: ChatBotAgent = st.session_state.chatbot_agent

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    # 生成回复
                    response = agent.chat(
                        session_id=st.session_state.session_id,
                        user_message=prompt,
                        record_to_patchouli=True  # 推送给帕秋莉
                    )

                    st.markdown(response)
                    st.session_state.messages_displayed.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_msg = f"❌ 调用失败: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages_displayed.append({"role": "assistant", "content": error_msg})


def main():
    """主函数"""
    # 初始化
    init_session_state()

    # 首次加载时从 Redis 恢复历史
    if not st.session_state.messages_displayed:
        load_session_history()

    # 渲染界面
    render_sidebar()
    render_chat_interface()

    # 底部信息
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("🔗 [GitHub](https://github.com/yourusername/HiveMemory)")
    with col2:
        st.caption("📖 [文档](https://docs.hivememory.com)")
    with col3:
        st.caption("🐛 [报告问题](https://github.com/yourusername/HiveMemory/issues)")


if __name__ == "__main__":
    main()
