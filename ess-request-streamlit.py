import streamlit as st
import os
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import ollama
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
DOC_PATH = "./data/workflow.txt"  # Path to the text document
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db"

def ingest_text(doc_path):
    """Load text documents manually from a .txt file."""
    if os.path.exists(doc_path):
        try:
            with open(doc_path, "r", encoding="utf-8") as file:
                data = file.read()
            logging.info("Text file loaded successfully.")
            return data
        except Exception as e:
            logging.error(f"Error loading text file: {e}")
            st.error(f"Error loading text file: {e}")
            return None
    else:
        logging.error(f"Text file not found at path: {doc_path}")
        st.error("Text file not found.")
        return None

def split_documents(documents):
    """Split documents into smaller chunks."""
    try:
        # Wrap the raw text into a Document object
        doc_obj = Document(page_content=documents)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
        chunks = text_splitter.split_documents([doc_obj])  # Pass a list with the Document object
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
            data = ingest_text(DOC_PATH)  # Use text file ingestion here
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
    template = """Answer the question step by step for the process outlined below.
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough() }
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created with preserved syntax.")
    return chain

steps = [
    {"question": "Please select the advance sub-type (e.g., personal loan, salary advance).", "step": "Step 1"},
    {"question": "Please enter the amount you wish to request as an advance.", "step": "Step 2"},
    {"question": "Do you want to calculate the EMI schedule for the requested amount?", "step": "Step 3"},
    {"question": "Would you like to preview your application before submitting?", "step": "Step 4"},
    {"question": "Do you wish to submit your advance request application?", "step": "Step 5"}
]

def main():
    st.title("ESS Assistant")

    # Initialize session state for tracking steps
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0  # Start at step 0
    if "responses" not in st.session_state:
        st.session_state.responses = {}

    current_step = st.session_state.current_step
    responses = st.session_state.responses

    if current_step < len(steps):
        step = steps[current_step]
        logging.info(f"Step: {step}")
        
        # Check if current step is Step 3, 4, or 5
        if step['step'] in ['Step 3', 'Step 4', 'Step 5']:
            confirm = st.radio(f"{step['question']}", options=["No", "Yes"], index=0)

            if confirm == "Yes":
                # Save the response and move to the next step
                responses[step['step']] = "Yes"
                st.session_state.responses = responses
                st.session_state.current_step = current_step + 1  # Move to the next step

                logging.info(f"Step '{step['step']}' completed. User response: Yes. Moving to the next step.")
                st.text(f"Log: Step '{step['step']}' completed! User response: Yes. Moving to the next step.")
                
                # Show success message after submission
                if step['step'] == 'Step 5':
                    st.success("Your advance request has been successfully submitted!")
                
                # Disable further actions (stop the flow after submission)
                if step['step'] == 'Step 5':
                    st.session_state.current_step = len(steps)  # Set to the last step, preventing further steps
                
                # Refresh the UI to show the next step
                st.rerun()  # This will re-render the app and show the next question

                # Optionally: Trigger any submission or API call here (for example, save data to a database)
                # API call or final actions here

            elif confirm == "No":
                # No action needed for "No" confirmation
                logging.info(f"Step '{step['step']}' was not confirmed. User response: No.")
                st.text(f"Log: Step '{step['step']}' was not confirmed. User response: No.")
                st.warning(f"Please confirm the submission to proceed.")
        else:
            # Ask the user for their input for the current step
            user_input = st.text_input(step['question'], "")

            if user_input:
                # Display the confirmation radio options
                confirm = st.radio(f"Is this correct? Your answer: {user_input}", options=["No", "Yes"], index=0)
                logging.info(confirm)

                if confirm == "Yes":
                    # Save the response in session state
                    responses[step['step']] = user_input
                    st.session_state.responses = responses
                    st.session_state.current_step = current_step + 1  # Move to the next step only if confirmed as "Yes"
                    logging.info(f"Step '{step['step']}' completed. User response: {user_input}. Moving to the next step.")
                    st.text(f"Log: Step '{step['step']}' completed! User response: {user_input}. Moving to the next step.")
                    st.success(f"Step '{step['step']}' completed! Moving to the next step.")
                    
                    # Refresh the UI to show the next step
                    st.rerun()  # This will re-render the app and show the next question
                elif confirm == "No":
                    # No action needed for "No" confirmation, just inform the user to correct their response.
                    logging.info(f"Step '{step['step']}' was not confirmed. User response: {user_input}. Please correct your response.")
                    st.text(f"Log: Step '{step['step']}' was not confirmed. User response: {user_input}. Please correct your response.")
                    st.warning(f"Please provide your correct answer for step '{step['step']}'.")
            else:
                st.info("Please enter your response to proceed.")
    else:
        st.success("Your Festival Advance Request has been successfully submitted.")

if __name__ == "__main__":
    main()
