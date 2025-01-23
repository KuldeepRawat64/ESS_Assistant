import streamlit as st
import os
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename="app.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
DEFAULT_DOC_PATH = "./data/SRS_ESS_V1.0_Phase1-A.pdf"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db"


def ingest_pdf(doc_path):
    """Load PDF documents."""
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    else:
        logging.error(f"PDF file not found at path: {doc_path}")
        st.error("PDF file not found.")
        return None


def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks


@st.cache_resource
def load_vector_db(doc_path):
    """Load or create the vector database."""
    try:
        # Pull the embedding model if not already available
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
            # Load and process the PDF document
            data = ingest_pdf(doc_path)
            if data is None:
                return None

            # Split the documents into chunks
            chunks = split_documents(data)

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
        logging.error(f"Error loading vector DB: {str(e)}")
        st.error("Failed to load vector database.")
        return None


def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever


def create_chain(retriever, llm):
    """Create the chain with preserved syntax."""
    # RAG prompt
    template = """Answer the question based ONLY on the following context:
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
    st.title("Document Assistant")

    # Sidebar for file upload
    st.sidebar.title("Settings")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
    doc_path = DEFAULT_DOC_PATH

    if uploaded_file:
        temp_dir = "./temp/"
        os.makedirs(temp_dir, exist_ok=True)  # Ensure the temp directory exists

        doc_path = os.path.join(temp_dir, uploaded_file.name)
        with open(doc_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"Uploaded {uploaded_file.name}")


    # User input
    user_input = st.text_area("Enter your question:", "")

    if st.button("Submit"):
        with st.spinner("Generating response..."):
            try:
                # Initialize the language model
                llm = ChatOllama(model=MODEL_NAME)

                # Load the vector database
                vector_db = load_vector_db(doc_path)
                if vector_db is None:
                    st.error("Failed to load or create the vector database.")
                    return

                # Create the retriever
                retriever = create_retriever(vector_db, llm)

                # Create the chain
                chain = create_chain(retriever, llm)

                # Get the response
                response = chain.invoke(input=user_input)

                st.markdown("**Assistant:**")
                st.write(response)
            except Exception as e:
                logging.error(f"Error: {str(e)}")
                st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please enter a question to get started.")


if __name__ == "__main__":
    main()
