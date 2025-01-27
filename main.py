from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import \
    HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import \
    create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os

from constants import VECTORDB_PATH
from utils.pdfFunctions import load_pdfs_from_folder

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",  # Especifica el modelo a usar
    temperature=0.7,  # Controla la aleatoriedad de la salida (0 = determinista)
    max_tokens=None,  # Sin límite en el número de tokens generados
    timeout=None,  # Sin límite de tiempo
    max_retries=2,  # Número máximo de reintentos en caso de error
    api_key=google_api_key  # La clave API
)


def generarBDVectorial(folderPath):
    raw_documents = load_pdfs_from_folder(folderPath)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")

    with st.spinner("Generando índices..."):

        if os.path.exists(VECTORDB_PATH):
            vectorstore = FAISS.load_local(VECTORDB_PATH, embeddings, allow_dangerous_deserialization=True)
            if raw_documents:
                docs = text_splitter.split_documents(documents=raw_documents)
                vectorstore.add_documents(docs)
                vectorstore.save_local(VECTORDB_PATH)
        else:
            docs = text_splitter.split_documents(documents=raw_documents)
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(VECTORDB_PATH)

    return VectorStoreRetriever(vectorstore=vectorstore)  # Devuelve el retriever


def generarConsulta(query, llm, retriever):
    # Define el prompt del sistema
    system_prompt = (
        "Use el contexto dado para responder la pregunta"
        "Si no sabe la respuesta, digamos que no lo sabe"
        "Responde con un mínimo de 5 líneas y un máximo de 10"
        "Actua como un experto capaz de explicar conceptos complejos en términos muy sencillos, no menciones que lo haces de manera sencilla"
        "No muestres que recoges la información de un contexto, actúas como un agente experto"
        "Dedicate a responder sólo la pregunta"
        "Contexto: {context}"
    )
    # Crea una plantilla de prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    # Crea una cadena para combinar documentos
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    # Crea una cadena de recuperación
    chain = create_retrieval_chain(retriever, question_answer_chain)

    return chain.invoke({"input": query})  # Devuelve la respuesta a la consulta


st.set_page_config(
    page_title="Demo Blockchain AI Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Demo Blockchain AI Agent")

with st.sidebar:
    st.button("Chat 1", type="secondary", use_container_width=True)
    st.button("Chat 2", type="secondary", use_container_width=True)
    st.button("Chat 3", type="secondary", use_container_width=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
    content = "Hello 👋. I am your personal Blockchain AI agent. Chat with me to learn more things about Blockchain!"
    st.session_state.messages.append({"role": "assistant", "content": content})

if "vectorDb" not in st.session_state:
    # Genera la base de datos vectorial
    retriever = generarBDVectorial("data")
    st.session_state.vectorDb = retriever

with st.container():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


if prompt := st.chat_input("What do you want to know?"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    try:
        # Genera la respuesta
        chat_completion = generarConsulta(prompt, llm, st.session_state.vectorDb)
        # Muestra la respuesta del asistente
        with st.chat_message("assistant"):
            full_response = chat_completion["answer"]
            st.write(full_response)

            # Muestra el contexto usado para generar la respuesta
            #with st.expander("Contexto"):
                #for contexto in chat_completion["context"]:
                    #st.write_stream(contexto)


        st.session_state.messages.append({"role": "assistant", "content": full_response})
    except Exception as e:  # Maneja las excepciones
        st.error(e)
