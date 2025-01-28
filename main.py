import streamlit as st

from backend.core import run_llm

st.set_page_config(
    page_title="Demo Blockchain AI Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.header("My Chats")
    st.divider()
    st.subheader("Today")
    st.write("What is Blockchain and How Does It Work?")
    st.write("How is Data Stored in a Blockchain?")
    st.write("What Are Smart Contracts and How Do They Work?")
    st.write("Difference Between Public and Private Blockchains")

st.image("Fujitsu.svg")
st.divider()
st.title("Demo Blockchain AI Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []
    content = "Hello 👋. I am your personal Blockchain AI agent. Chat with me to learn more things about Blockchain!"
    st.chat_message("assistant").markdown(content)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

with st.container():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


if prompt := st.chat_input("What do you want to know?"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    try:
        with st.spinner("Thinking..."):
            print("st.session_state[chat_history]: ", st.session_state["chat_history"])
            result = run_llm(query=prompt, chat_history=st.session_state["chat_history"])

            with st.chat_message("assistant"):
                full_response = result["result"]
                st.write(full_response)
                st.session_state["chat_history"].append(("human", prompt))
                st.session_state["chat_history"].append(("ai", full_response))

        st.session_state.messages.append({"role": "assistant", "content": full_response})
    except Exception as e:  # Maneja las excepciones
        st.error(e)
