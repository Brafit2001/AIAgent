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
    indexed_pdfs = open_json(PDF_HASHES_JSON_PATH)
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            file_hash = generate_hash(pdf_path)
            if file_hash not in indexed_pdfs:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                documents.extend(docs)
                indexed_pdfs[file_hash] = filename
                shutil.move(pdf_path, PDF_INDEXED_PATH)
            else:
                shutil.move(pdf_path, PDF_DUPLICATED_PATH)

    write_json(PDF_HASHES_JSON_PATH, indexed_pdfs)
    return documents
