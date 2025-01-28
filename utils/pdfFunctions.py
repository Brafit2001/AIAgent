import glob
import os
import shutil

from langchain_community.document_loaders import PyPDFLoader

from constants import PDF_HASHES_JSON_PATH, PDF_DUPLICATED_PATH, PDF_INDEXED_PATH, PDF_DATA
from utils.jsonFunctions import open_json, write_json
from utils.cryptographicFunctions import generate_hash


def load_pdfs_from_folder(folder_path):
    """
    Carga todos los PDFs de una carpeta y devuelve una lista de documentos en formato LangChain.
    """
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            print("------- READING PDF: ", filename, "----------------")
            pdf_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)

    return documents
