import os
from pathlib import Path
import logging
import sys
import uuid
import shutil
import time
from threading import Thread

import schedule
import streamlit as st
import openai

from llama_index.core import (
    SimpleDirectoryReader,
)

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex

import qdrant_client


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

sys.path.append(parent_dir)

from src.document_retrieval import DocumentRetrieval

# Minutes for scheduled cache deletion
EXIT_TIME_DELTA = 30

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Starting Streamlit app...")

# Configure other settings
openai.api_key = st.secrets.OPENAI_API_KEY


def are_credentials_set():
    """
    Check if the necessary credentials are set.
    This is a placeholder function. Replace with actual credential checks.
    """
    # For demonstration, let's assume we check for an API key in the session state
    if 'api_key' in st.session_state and st.session_state['api_key']:
        return True
    else:
        # If not set, log the missing credentials
        logger.warning("Credentials are not set.")
        return False


def save_credentials(api_key):
    st.session_state['api_key'] = api_key

    return "Credentials saved successfully."


def initialize_document_retrieval():
    """
    Initialize the DocumentRetrieval class with the API key from session state.
    This is a placeholder function. Replace with actual initialization logic.
    """
    if 'api_key' in st.session_state and st.session_state['api_key']:
        api_key = st.session_state['api_key']
        logger.info("Initializing DocumentRetrieval with provided API key.")
        
        return DocumentRetrieval(api_key)
    else:
        logger.error("API key is not set. Cannot initialize DocumentRetrieval.")
        raise ValueError("API key is not set.")


def delete_temp_dir(temp_dir: str) -> None:
    """Delete the temporary directory and its contents."""

    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logging.info(f'Temporary directory {temp_dir} deleted.')
        except:
            logging.info(f'Could not delete temporary directory {temp_dir}.')


def schedule_temp_dir_deletion(temp_dir: str, delay_minutes: int) -> None:
    """Schedule the deletion of the temporary directory after a delay."""

    schedule.every(delay_minutes).minutes.do(delete_temp_dir, temp_dir).tag(temp_dir)

    def run_scheduler() -> None:
        while schedule.get_jobs(temp_dir):
            schedule.run_pending()
            time.sleep(1)

    # Run scheduler in a separate thread to be non-blocking
    Thread(target=run_scheduler, daemon=True).start()

def save_user_files(docs, schedule_deletion=True):
    """
    Save the user's uploaded documents to a temporary directory.
    If schedule_deletion is True, schedule the deletion of the temporary directory after 30 minutes.

    :param docs: A list of uploaded documents.
    :param schedule_deletion: A boolean indicating whether to schedule deletion of the temporary directory.

    :return: The temporary directory path.
    """
    temp_folder = os.path.join(parent_dir, 'data', 'tmp', st.session_state.session_temp_subfolder)
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    else:
        # If there are already files there, delete them
        for filename in os.listdir(temp_folder):
            file_path = os.path.join(temp_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logging.error(f'Failed to delete {file_path}. Reason: {e}')

    # Save all selected files to the tmp dir with their file names
    for doc in docs:
        assert hasattr(doc, 'name'), 'doc has no attribute name.'
        assert callable(doc.getvalue), 'doc has no method getvalue.'
        temp_file = os.path.join(temp_folder, doc.name)
        with open(temp_file, 'wb') as f:
            f.write(doc.getvalue())

    if schedule_deletion:
        schedule_temp_dir_deletion(temp_folder, EXIT_TIME_DELTA)
        st.toast(
            """Your session will be active for the next 30 minutes, after this time files 
            will be deleted"""
        )

    return temp_folder
    
def main():
    st.set_page_config(
        page_title='RAG App',
    )

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'document_retrieval' not in st.session_state:
        st.session_state.document_retrieval = None
    if 'input_disabled' not in st.session_state:
        st.session_state.input_disabled = True
    if 'st_session_id' not in st.session_state:
        st.session_state.st_session_id = str(uuid.uuid4())
    if 'session_temp_subfolder' not in st.session_state:
        st.session_state.session_temp_subfolder = 'upload_' + st.session_state.st_session_id


    # Display a title and a paragraph
    st.header("Retrieval-Augmented Generation App", divider='red')

    with st.sidebar:
        st.header('Setup', divider='red')
        
        if not are_credentials_set():
            input = st.text_input('API KEY', value=st.session_state.get('api_key', ''), type='password')
            
            if st.button('Save Credentials', key='save_credentials_sidebar'):
                message = save_credentials(input)
                st.success(message)
                st.rerun()
        else:
            if st.button('Clear Credentials', key='clear_credentials'):
                save_credentials("")
                st.session_state.document_retrieval = None
                st.session_state.input_disabled = True
                st.rerun()
        
        if are_credentials_set():
            if st.session_state.document_retrieval is None:
                st.session_state.document_retrieval = initialize_document_retrieval()

        if st.session_state.document_retrieval is not None:
            st.markdown('**1. Upload your documents**')

            hide_label = """
                <style>
                    div[data-testid="stFileUploaderDropzoneInstructions"]>div>small {
                    visibility:hidden;
                    }
                    div[data-testid="stFileUploaderDropzoneInstructions"]>div>small::before {
                    content:"limit FILE_LIMITS per file • FILE_TYPES";
                    visibility:visible;
                    display:block;
                    } 
                </style>
                """

            filetypes = ['pdf', 'md','txt', 'csv','doc', 'docx', 'html', 'json', 'rtf', 'xml', 'xlsx', 'tsv']
            hide_label = hide_label.replace('FILE_LIMITS', '20 MB').replace('FILE_TYPES', ', '.join(filetypes))
            st.markdown(hide_label, unsafe_allow_html=True)
            docs = st.file_uploader("File Upload", accept_multiple_files=True, type=filetypes)

            st.markdown('**2. Process your documents and create vector store**')
            st.markdown(
                '**Note:** Depending on the size and number of your documents, this could take several minutes'
            )
            if st.button('Process'):
                    with st.spinner('Processing'):
                        try:
                            # Process the documents and create the vector store
                            if docs is not None:
                                temp_folder = save_user_files(docs, schedule_deletion=True)

                            # do something with the temp folder
                                embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
                                reader = SimpleDirectoryReader(
                                    input_dir=(temp_folder),
                                    recursive=True,
                                )

                                loaded_docs = reader.load_data()

                                client = qdrant_client.QdrantClient(location=":memory:")
                                vector_store = QdrantVectorStore(client=client, collection_name="test_store")

                                pipeline = IngestionPipeline(
                                    transformations=[
                                        SentenceSplitter(),
                                        embed_model,
                                    ],
                                    vector_store=vector_store,
                                )

                                # Ingest directly into a vector db
                                pipeline.run(documents=loaded_docs)

                                # Create your index

                                index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
                                
                                st.session_state.index = index
                                st.session_state.chat_engine = index.as_chat_engine(
                                    chat_mode="condense_question", verbose=True, streaming=True
                                )
                            
                            st.toast('File uploaded! Go ahead and ask some questions', icon='🎉')
                            st.session_state.input_disabled = False
                        except Exception as e:
                            logging.error(f'An error occurred while processing: {str(e)}')
                            st.error(f'An error occurred while processing: {str(e)}')                            

            st.markdown('**3. Ask questions about your data!**')                
            with st.expander('Chat settings', expanded=True):
                st.markdown('**Reset chat**')
                st.markdown('**Note:** Resetting the chat will clear all conversation history')
                if st.button('Reset conversation'):
                    st.session_state.messages = []
                    st.session_state.chat_engine.reset()

    # Prompt for user input and save to chat history
    if prompt := st.chat_input('Ask questions about your data', disabled=st.session_state.input_disabled):
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # # If last message is not from assistant, generate a new response
    if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            response_stream = st.session_state.chat_engine.stream_chat(prompt)
            st.write_stream(response_stream.response_gen)

            # Add response to message history
            message = {"role": "assistant", "content": response_stream.response}
            st.session_state.messages.append(message)    


if __name__ == "__main__":
    # Run the main function
    try:
        main()
        logger.info("Streamlit app started successfully.")
    except Exception as e:
        logger.error(f"Error starting Streamlit app: {e}")
        st.error(f"An error occurred: {e}")