import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Any, Dict, List
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
from constants import INDEX_NAME

load_dotenv()


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",  # Especifica el modelo a usar
        temperature=0,  # Controla la aleatoriedad de la salida (0 = determinista)
        max_tokens=None,  # Sin límite en el número de tokens generados
        timeout=None,  # Sin límite de tiempo
        max_retries=2,  # Número máximo de reintentos en caso de error
        api_key=os.getenv("GOOGLE_API_KEY")  # La clave API
    )

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    retrieval_qa_chat_prompt = PromptTemplate(
        input_variables=["context", "input", "chat_history"],
        template=(
            "You are an expert assistant. "
            "Use the given context and the previous conversation history to answer the question.\n"
            "If you do not know the answer, say you do not know.\n"
            "Provide a response with a minimum of 5 lines and a maximum of 10.\n"
            "Act as an expert capable of explaining complex concepts in very simple terms. "
            "Do not mention that you are simplifying the explanation.\n"
            "Do not reveal that you are retrieving information from a context; act as an expert agent.\n"
            "Focus solely on answering the question.\n\n"
            "Previous Conversation History:\n{chat_history}\n\n"
            "Context:\n{context}\n\n"
            "Current Question:\n{input}\n"
            "Answer:"
        )
    )
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    return {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"]
    }

