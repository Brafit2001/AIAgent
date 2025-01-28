from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

from constants import INDEX_NAME
from utils.pdfFunctions import load_pdfs_from_folder
import os

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def ingest_docs():
    print("------ LOAD PDFS FROM FOLDER -----------")
    raw_documents = load_pdfs_from_folder("data")
    print("------ SPLITTING -----------")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    print("------ INDEXING INTO PINECONE -----------")
    PineconeVectorStore.from_documents(
        documents, embeddings, index_name=INDEX_NAME
    )


if __name__ == "__main__":
    ingest_docs()


