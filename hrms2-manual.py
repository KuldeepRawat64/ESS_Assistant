import streamlit as st
import os
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Updated import
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
DOC_PATH = "./data/HRMS2 Handbook.pdf"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db"

import pytesseract
from PIL import Image
import pdf2image

def extract_text_with_ocr(pdf_path):
    """Extract text from PDF using OCR if normal text extraction fails."""
    try:
        # Convert PDF to images
        images = pdf2image.convert_from_path(pdf_path)
        
        # Apply OCR on each page image
        extracted_text = ""
        for image in images:
            text = pytesseract.image_to_string(image)
            extracted_text += text
        return extracted_text
    except Exception as e:
        logging.error(f"Error during OCR extraction: {e}")
        st.error(f"Error during OCR extraction: {e}")
        return ""

from langchain.document_loaders import PyPDFLoader

def ingest_pdf(doc_path):
    """Load PDF documents."""
    if os.path.exists(doc_path):
        try:
            loader = PyPDFLoader(file_path=doc_path)
            data = loader.load()
            logging.info("PDF loaded successfully.")
            return data
        except Exception as e:
            logging.error(f"Error loading PDF: {e}")
            st.error(f"Error loading PDF: {e}")
            return None
    else:
        logging.error(f"PDF file not found at path: {doc_path}")
        st.error("PDF file not found.")
        return None

def split_documents(documents):
    """Split documents into smaller chunks."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
        chunks = text_splitter.split_documents(documents)
        logging.info("Documents split into chunks.")
        return chunks
    except Exception as e:
        logging.error(f"Error splitting documents: {e}")
        st.error(f"Error splitting documents: {e}")
        return []

@st.cache_resource
def load_vector_db():
    """Load or create the vector database."""
    try:
        ollama.pull(EMBEDDING_MODEL)
        embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

        if os.path.exists(PERSIST_DIRECTORY):
            vector_db = Chroma(
                embedding_function=embedding,
                collection_name=VECTOR_STORE_NAME,
                persist_directory=PERSIST_DIRECTORY,
            )
            logging.info("Loaded existing vector database.")
        else:
            data = ingest_pdf(DOC_PATH)
            if data is None:
                return None

            chunks = split_documents(data)
            if not chunks:
                return None

            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embedding,
                collection_name=VECTOR_STORE_NAME,
                persist_directory=PERSIST_DIRECTORY,
            )
            vector_db.persist()
            logging.info("Vector database created and persisted.")

        return vector_db
    except Exception as e:
        logging.error(f"Error loading vector database: {e}")
        st.error(f"Error loading vector database: {e}")
        return None

def create_retriever(vector_db):
    """Create a single-query retriever."""
    retriever = vector_db.as_retriever()
    logging.info("Single-query retriever created.")
    return retriever

def create_chain(retriever, llm):
    """Create the chain with preserved syntax."""
    template = """Answer the question without mentioning any document, mention only 
    Human Resource Management System, do not mention the context. The default context is the provided PDF, but do not mention it and answer the question based ONLY and in English language on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created with preserved syntax.")
    return chain

def main():
    st.title("HRMS2 Assistant")

    user_input = st.text_input("Enter your question:", "")

    if user_input:
        with st.spinner("Generating response..."):
            try:
                llm = ChatOllama(model=MODEL_NAME)

                vector_db = load_vector_db()
                if vector_db is None:
                    st.error("Failed to load or create the vector database.")
                    return

                retriever = create_retriever(vector_db)
                chain = create_chain(retriever, llm)

                # Streaming response
                response_placeholder = st.empty()
                response_content = ""
                for partial_response in chain.stream(input=user_input):
                    response_content += partial_response
                    response_placeholder.markdown(f"**Assistant:** {response_content}")

            except Exception as e:
                logging.error(f"Error generating response: {e}")
                st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please enter a question to get started.")

if __name__ == "__main__":
    main()
