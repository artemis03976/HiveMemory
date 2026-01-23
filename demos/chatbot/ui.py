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
import logging

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# 配置日志 - 确保在导入 hivememory 之前或尽早配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
# 显式设置 hivememory 的日志级别
logging.getLogger("hivememory").setLevel(logging.INFO)

import streamlit as st
import uuid
import redis
from datetime import datetime

from hivememory.patchouli.config import load_app_config
from hivememory.infrastructure.storage import QdrantMemoryStore
from hivememory.patchouli.librarian_core import PatchouliAgent
# ChatBotAgent and SessionManager are now local (moved to demos/chatbot/)
from .chatbot import ChatBotAgent
from .session_manager import SessionManager


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
    # 加载配置 (使用工厂函数)
    config = load_app_config()

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
    # 使用依赖注入传入配置
    patchouli = PatchouliAgent(
        storage=storage,
        perception_config=config.perception,
        generation_config=config.generation
    )

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

        # 创建 ChatBot Agent
        st.session_state.chatbot_agent = ChatBotAgent(
            patchouli=patchouli,
            session_manager=session_manager,
            user_id=st.session_state.user_id,
            agent_id="streamlit_chatbot",
            config=config,  # 依赖注入：传递全局配置
            enable_memory_retrieval=False,  # 默认关闭，后续由侧边栏控制
            enable_lifecycle_management=False  # 示例中暂不启用生命周期管理
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


def render_memory_inspector(storage, user_id):
    """渲染记忆库检查器"""
    st.subheader("🧠 记忆原子流")
    st.caption("实时展示生成的记忆原子 (Top 50)")

    try:
        # 获取所有记忆
        memories = storage.get_all_memories(
            filters={"meta.user_id": user_id},
            limit=50
        )

        if not memories:
            st.info("暂无记忆生成。尝试与 ChatBot 多聊聊！")
            return

        # 按创建时间倒序排列 (最新的在最前)
        # 注意：这里假设 meta.created_at 是 datetime 对象或可比较的字符串
        memories.sort(key=lambda x: x.meta.created_at, reverse=True)

        st.metric("记忆总数", len(memories))

        # 遍历展示
        for mem in memories:
            # 确定图标
            icon = "📝"
            mem_type = str(mem.index.memory_type)
            if "CODE" in mem_type:
                icon = "💻"
            elif "FACT" in mem_type:
                icon = "💡"
            elif "URL" in mem_type:
                icon = "🔗"
            elif "REFLECTION" in mem_type:
                icon = "🤔"

            # 格式化时间
            created_at = mem.meta.created_at
            if isinstance(created_at, str):
                try:
                    # 尝试解析 ISO 格式
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    time_str = created_at
            elif isinstance(created_at, datetime):
                time_str = created_at.strftime("%Y-%m-%d %H:%M:%S")
            else:
                time_str = str(created_at)

            # 展开器标题
            title = f"{icon} {mem.index.title}"
            
            with st.expander(title):
                st.caption(f"🕒 {time_str}")
                
                # 标签
                if mem.index.tags:
                    st.markdown(f"🏷️ **Tags**: `{'`, `'.join(mem.index.tags)}`")
                
                # 摘要
                st.markdown(f"**摘要**: {mem.index.summary}")
                
                # 类型
                st.caption(f"类型: {mem_type}")

    except Exception as e:
        st.error(f"加载记忆失败: {e}")


def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.title("HiveMemory")
        
        # 使用 Tabs 分离设置和记忆查看
        tab_settings, tab_memories = st.tabs(["⚙️ 设置", "🧠 记忆库"])
        
        with tab_settings:
            st.subheader("📋 当前会话")
            st.text(f"Session ID: {st.session_state.session_id}")
            st.text(f"User ID: {st.session_state.user_id}")

            # 会话统计
            agent: ChatBotAgent = st.session_state.chatbot_agent
            session_info = agent.get_session_info(st.session_state.session_id)
            st.metric("消息数量", session_info["message_count"])

            st.divider()

            # 功能控制
            st.subheader("🎛️ 功能控制")
            enable_retrieval = st.toggle(
                "启用记忆检索",
                value=agent.enable_memory_retrieval,
                help="开启后，ChatBot 会在回答前检索相关的历史记忆作为上下文。"
            )
            # 更新 Agent 状态
            if enable_retrieval != agent.enable_memory_retrieval:
                agent.enable_memory_retrieval = enable_retrieval
                st.rerun()

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
            st.text(f"感知层类型: {config.perception.layer_type}")
            if config.perception.layer_type == "semantic_flow":
                st.text(f"空闲超时: {config.perception.semantic_flow.idle_timeout_seconds // 60} 分钟")
                st.text(f"语义阈值: {config.perception.semantic_flow.semantic_threshold}")
            else:
                st.text(f"消息阈值: {config.perception.simple.message_threshold} 条")
                st.text(f"空闲超时: {config.perception.simple.timeout_seconds // 60} 分钟")
            st.text(f"高相似阈值: {config.generation.deduplicator.high_similarity_threshold}")
            st.text(f"低相似阈值: {config.generation.deduplicator.low_similarity_threshold}")

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
            
        with tab_memories:
            config, _, _, storage = initialize_system()
            render_memory_inspector(storage, st.session_state.user_id)


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
