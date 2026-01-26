import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from core.base import app
from database import save_chat_log, get_chat_history

# --- 页面配置 ---
st.set_page_config(page_title="TripGuard 差旅助手", page_icon="✈️", layout="centered")
st.title("✈️ TripGuard 智能差旅合规助手")

# --- Session State 初始化 ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]  # 默认生成一个短 ID

if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="您好！我是 TripGuard。输入您的 ID 可以继续之前的对话。")
    ]

# --- [核心修改] 侧边栏：登录与切换 ---
with st.sidebar:
    st.header("👤 用户登录")

    # 输入框：默认显示当前 ID
    input_id = st.text_input("Session ID (凭证)", value=st.session_state.session_id)

    # 登录按钮
    if st.button("🔄 加载/切换对话"):
        st.session_state.session_id = input_id

        # 1. 从数据库读取历史
        history = get_chat_history(input_id)

        # 2. 重置当前显示的消息列表
        st.session_state.messages = []

        if history:
            # 3. 如果有历史，转换回 LangChain 消息格式
            for role, content in history:
                if role == "user":
                    st.session_state.messages.append(HumanMessage(content=content))
                else:
                    st.session_state.messages.append(AIMessage(content=content))
            st.success(f"已恢复 {len(history)} 条记录")
        else:
            # 4. 如果没历史，显示欢迎语
            st.session_state.messages = [
                AIMessage(content="欢迎回来！这是一个新的会话。")
            ]
        st.rerun()  # 刷新页面

    st.divider()
    st.caption(f"当前 ID: {st.session_state.session_id}")

# --- 聊天界面渲染 ---
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        if msg.content:
            with st.chat_message("assistant"):
                st.markdown(msg.content)

# --- 处理用户输入 ---
if prompt := st.chat_input("请输入您的需求..."):
    # 1. 显示并记录用户输入
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))

    # [新增] 保存用户消息到数据库
    save_chat_log(st.session_state.session_id, "user", prompt)

    # 2. 调用 AI
    config = {"configurable": {"thread_id": st.session_state.session_id}}

    with st.chat_message("assistant"):
        with st.spinner("TripGuard 正在思考..."):
            try:
                inputs = {"messages": [("user", prompt)]}
                result = app.invoke(inputs, config=config)

                last_message = result["messages"][-1]
                response_content = last_message.content

                st.markdown(response_content)

                # 更新 Session State
                st.session_state.messages.append(AIMessage(content=response_content))

                # [新增] 保存 AI 回复到数据库
                save_chat_log(st.session_state.session_id, "ai", response_content)

            except Exception as e:
                st.error(f"❌ 系统出错: {str(e)}")
