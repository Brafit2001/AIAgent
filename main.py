import base64

import streamlit as st

from backend.core import run_llm

st.set_page_config(
    page_title="Demo Blockchain AI Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Función para convertir la imagen a base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


# Ruta de la imagen
profile_pic = "./profile.jpeg"
profile_name = "Marcelino Fernandez"

# Convertir imagen a base64
base64_img = get_base64_image(profile_pic)

# HTML para mostrar la imagen en la sidebar
st.sidebar.markdown(
    f"""
    <div style="display: flex; column-gap: 20px;">
        <img src="data:image/jpeg;base64,{base64_img}" width="40" height="40" style="border-radius: 50%;">
        <div style="display: flex; flex-direction: column;">
            <h2 style="padding: 0">{profile_name}</h2>
            <p>Fujitsu</p>
        </div>
        
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.divider()
    st.subheader("Today")
    st.write("What is Blockchain and How Does It Work?")
    st.write("How is Data Stored in a Blockchain?")
    st.write("What Are Smart Contracts and How Do They Work?")
    st.write("Difference Between Public and Private Blockchains")

st.image("Fujitsu.svg")
st.divider()
st.title("Demo Blockchain AI Agent")

if "messages" not in st.session_state and "chat_history" not in st.session_state:
    st.session_state["messages"] = []
    st.session_state["chat_history"] = []

    content = "Hello 👋. I am your personal Blockchain AI agent. Chat with me to learn more things about Blockchain!"
    st.chat_message("assistant", avatar="fujitsu_icon.svg").markdown(content)

with st.container():
    for message in st.session_state.messages:
        with st.chat_message(message["role"],
                             avatar="fujitsu_icon.svg" if message["role"] == "assistant" else "profile.jpeg"):
            st.markdown(message["content"])

if prompt := st.chat_input("What do you want to know?"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="profile.jpeg").markdown(prompt)

    try:
        with st.spinner("Thinking..."):
            result = run_llm(query=prompt, chat_history=st.session_state["chat_history"])
            full_response = result["result"]
            st.chat_message("assistant", avatar="fujitsu_icon.svg").markdown(full_response)
            st.session_state["chat_history"].append(("human", prompt))
            st.session_state["chat_history"].append(("ai", full_response))

            st.session_state.messages.append({"role": "assistant", "content": full_response})
    except Exception as e:  # Maneja las excepciones
        st.error(e)
