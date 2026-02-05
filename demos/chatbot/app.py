import streamlit as st
import logging
import uuid
import os
import sys
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

# Adjust path to include src if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../src"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Also add the demos directory to path so we can import from chatbot package
demos_dir = os.path.abspath(os.path.join(current_dir, "../.."))
if demos_dir not in sys.path:
    sys.path.insert(0, demos_dir)

from hivememory.patchouli.system import PatchouliSystem
from hivememory.patchouli.config import load_app_config
from demos.chatbot.chatbot import ChatBotAgent
from demos.chatbot.session_manager import SessionManager
from demos.chatbot.config import load_chatbot_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
USER_ID = "default_user"  # In a real app, this would come from auth
PAGE_TITLE = "HiveMemory ChatBot"
PAGE_ICON = "🤖"

def init_page():
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

@st.cache_resource
def init_system():
    """Initialize the backend system once"""
    try:
        # Load Config
        chatbot_config = load_chatbot_config()
        main_config = load_app_config()
        
        # Initialize Patchouli System
        patchouli = PatchouliSystem(config=main_config)
        
        # Initialize Redis Client (assuming default localhost:6379 for demo)
        import redis
        redis_client = redis.Redis(
            host=main_config.redis.host,
            port=main_config.redis.port,
            db=main_config.redis.db,
            password=main_config.redis.password,
            decode_responses=False # SessionManager expects bytes/handling decoding itself or not?
            # Checking SessionManager: it does key.decode("utf-8") so it handles bytes.
            # But let's check init: redis_client is passed in.
        )
        
        # Initialize Session Manager
        session_manager = SessionManager(redis_client=redis_client)
        
        # Initialize Agent
        agent = ChatBotAgent(
            patchouli_system=patchouli,
            session_manager=session_manager,
            user_id=USER_ID,
            config=main_config,
            chatbot_config=chatbot_config
        )
        
        return agent
    except Exception as e:
        st.error(f"System initialization failed: {e}")
        logger.exception("Initialization error")
        return None

def init_session_state():
    """Initialize session state variables"""
    if "current_view" not in st.session_state:
        st.session_state.current_view = "chat"
    
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
        
    if "messages" not in st.session_state:
        st.session_state.messages = []

def main():
    init_page()
    init_session_state()
    
    # Initialize System
    agent = init_system()
    if not agent:
        st.stop()
        
    # Render UI
    load_css(st.session_state.current_view)
    render_sidebar()
    
    # Render Main Content
    if st.session_state.current_view == "chat":
        render_chat_view(agent)
    elif st.session_state.current_view == "memory":
        render_memory_view(agent)
    elif st.session_state.current_view == "config":
        render_config_view()

def load_css(current_view):
    # Base CSS
    base_css = """
        <style>
        /* Global Dark Theme Overrides */
        .stApp {
            background-color: #1E1E1E;
            color: #CCCCCC;
        }
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            width: 80px !important; /* Force narrow sidebar */
            min-width: 80px !important;
            max-width: 80px !important;
            background-color: #252526;
        }
        
        /* Hide sidebar collapse button and header */
        [data-testid="collapsedControl"] { display: none; }
        header[data-testid="stHeader"] { display: none; }
        
        /* Sidebar Buttons */
        .stButton button {
            background-color: transparent;
            border: none;
            color: #CCCCCC;
            font-size: 24px;
            padding: 10px 0;
            width: 100%;
            border-radius: 0;
        }
        .stButton button:hover {
            background-color: #2D2D2D;
            color: #FFFFFF;
        }
        .stButton button:focus {
            background-color: #37373D;
            color: #FFFFFF;
            border: none;
            box-shadow: none;
        }
        
        /* Chat Message Styling */
        .stChatMessage {
            background-color: transparent;
        }
        .stChatMessage [data-testid="stChatMessageContent"] {
            background-color: #2D2D2D;
            border-radius: 10px;
            padding: 10px;
            color: #E1E1E1;
        }
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #1E1E1E; 
        }
        ::-webkit-scrollbar-thumb {
            background: #424242; 
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #4F4F4F; 
        }
        </style>
    """
    
    # View-specific CSS
    view_css = ""
    if current_view == "chat":
         view_css = """
         <style>
         /* Session List Border (Explicit targetting) */
         div[data-testid="column"]:has(#session-list-marker) {
             background-color: #252526;
             border: 2px solid #555; /* Full border */
             border-radius: 8px;
             padding: 15px;
             margin-right: 10px; /* Spacing */
         }
         
         /* Chat Window Border */
         /* Target the container that holds the chat messages */
         .stChatFloatingInputContainer {
             bottom: 20px;
         }
         
         /* Wrap the chat area */
         [data-testid="stVerticalBlockBorderWrapper"] {
             border: 2px solid #555 !important;
             border-radius: 8px;
             padding: 10px;
             background-color: #1E1E1E;
         }
         
         /* Specific fix for chat input container if needed */
         </style>
         """
        
    st.markdown(base_css + view_css, unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown("<div style='text-align: center; padding: 10px 0;'>🤖</div>", unsafe_allow_html=True)
        
        # Navigation Buttons
        # Use columns to center buttons if needed, or just full width
        
        # Chat Button
        if st.button("💬", key="nav_chat", help="对话窗口"):
            st.session_state.current_view = "chat"
            st.rerun()
            
        # Memory Button
        if st.button("🧠", key="nav_memory", help="记忆库"):
            st.session_state.current_view = "memory"
            st.rerun()
            
        # Config Button
        if st.button("⚙️", key="nav_config", help="系统配置"):
            st.session_state.current_view = "config"
            st.rerun()
            
        # Spacer to push settings to bottom
        st.markdown("<div style='height: 50vh;'></div>", unsafe_allow_html=True)
        
        # Bottom Settings (Icon only)
        st.button("🔧", key="nav_settings", help="设置")

def render_chat_view(agent):
    # Layout: Sessions | Chat
    col_sessions, col_chat = st.columns([1, 4])
    
    with col_sessions:
        st.markdown('<div id="session-list-marker"></div>', unsafe_allow_html=True)
        st.markdown("### 会话列表")
        if st.button("➕ 新建对话", use_container_width=True):
            new_id = str(uuid.uuid4())
            st.session_state.current_session_id = new_id
            st.rerun()
            
        st.markdown("---")
        
        # List sessions
        try:
            sessions = agent.session_manager.get_all_sessions(USER_ID)
            if not sessions:
                st.caption("暂无历史会话")
                
            for sess in sessions:
                sess_id = sess["session_id"]
                # Display shortened ID or timestamp
                label = f"Session {sess_id[:6]}..."
                is_active = st.session_state.current_session_id == sess_id
                
                if st.button(label, key=f"sess_{sess_id}", use_container_width=True, 
                             type="primary" if is_active else "secondary"):
                    st.session_state.current_session_id = sess_id
                    st.rerun()
        except Exception as e:
            st.error(f"加载会话列表失败: {e}")

    with col_chat:
        # Wrap chat area in a container with border styling via st.container(border=True)
        # Streamlit 1.30+ supports border=True
        chat_wrapper = st.container(border=True)
        with chat_wrapper:
            if not st.session_state.current_session_id:
                # If no session selected but we have sessions, select first? 
                # Or just show welcome
                if "current_session_id" in st.session_state and st.session_state.current_session_id:
                     session_id = st.session_state.current_session_id
                else:
                     st.info("请选择或新建一个会话")
                     # Auto create if empty?
                     if st.button("开始新对话"):
                         st.session_state.current_session_id = str(uuid.uuid4())
                         st.rerun()
                     return
            else:
                 session_id = st.session_state.current_session_id
            
            # Chat Header
            st.caption(f"当前会话 ID: {session_id}")
            
            # Chat History Container
            # Try to use height for scrollable container
            try:
                chat_container = st.container(height=600)
            except TypeError:
                # Fallback for older streamlit
                chat_container = st.container()
            
            with chat_container:
                history = agent.session_manager.get_history(session_id)
                if not history:
                    st.caption("👋 你好！我是 HiveMemory 助手，有什么可以帮你的吗？")
                
                for msg in history:
                    with st.chat_message(msg.role, avatar="👤" if msg.role == "user" else "🤖"):
                        st.markdown(msg.content)
            
                # Export Button (at bottom of history)
                if history:
                    st.markdown("---")
                    col_spacer, col_btn = st.columns([5, 1])
                    with col_btn:
                        # Prepare export data
                        export_data = json.dumps([m.to_dict() for m in history], ensure_ascii=False, indent=2)
                        st.download_button(
                            label="📥 导出记录",
                            data=export_data,
                            file_name=f"chat_history_{session_id}.json",
                            mime="application/json",
                            key=f"export_{session_id}"
                        )

            # Chat Input
            if prompt := st.chat_input("输入消息...", key="chat_input"):
                # Optimistic UI update
                with chat_container:
                     with st.chat_message("user", avatar="👤"):
                        st.markdown(prompt)
                
                with st.spinner("思考中..."):
                    try:
                        response = agent.chat(session_id, prompt)
                        st.rerun()
                    except Exception as e:
                        st.error(f"对话出错: {e}")

def render_memory_view(agent):
    st.markdown("### 🧠 记忆库")
    
    search_query = st.text_input("搜索记忆...", placeholder="输入关键词")
        
    st.markdown("---")
    
    # Fetch memories
    try:
        # Construct filters
        filters = {"meta.user_id": USER_ID}
        
        # TODO: Implement search logic if query exists (using vector search or simple filter)
        # For now, just list all
        
        memories = agent.patchouli_system.storage.get_all_memories(
            filters=filters,
            limit=50
        )
        
        if not memories:
            st.info("暂无记忆条目。")
            return

        for mem in memories:
            # Determine card color or style
            with st.expander(f"📄 {mem.index.title or '无标题'} | {mem.meta.created_at.strftime('%Y-%m-%d %H:%M')}"):
                st.markdown(f"**摘要**: {mem.index.summary}")
                st.markdown(f"**标签**: {', '.join(mem.index.tags)}")
                st.text_area("内容预览", value=mem.payload.content, height=100, disabled=True, key=f"content_{mem.id}")
                
                col_act_1, col_act_2 = st.columns([1, 1])
                with col_act_2:
                    if st.button("🗑️ 删除", key=f"del_{mem.id}", type="primary"):
                        if agent.patchouli_system.storage.delete_memory(mem.id):
                            st.success("已删除")
                            st.rerun()
                        else:
                            st.error("删除失败")
                            
    except Exception as e:
        st.error(f"无法获取记忆: {e}")

def render_config_view():
    st.markdown("### ⚙️ 帕秋莉系统配置")
    st.info("🚧 帕秋莉配置功能开发中…")
    
    # Placeholder container
    with st.container():
        st.markdown("""
        <div style='background-color: #2D2D2D; padding: 20px; border-radius: 10px; color: #888;'>
            配置面板预留区域
        </div>
        """, unsafe_allow_html=True)



if __name__ == "__main__":
    main()
