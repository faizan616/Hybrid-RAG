import chromadb
import streamlit as st
import os
import shutil
import time
from langchain_community.vectorstores import Chroma

def initialize_vector_store(
    documents,
    embedding_model,
    persist_directory="chroma_db",
    collection_name="pdf_collection"
):
    """
    Initialize a Chroma vector store from documents.

    Args:
        documents: List of Document objects to store
        embedding_model: Embedding model to use
        persist_directory: Directory to persist the database
        collection_name: Name of the collection

    Returns:
        Chroma vector store instance and client
    """
    # Initialize Chroma client
    client = chromadb.PersistentClient(path=persist_directory)

    # Delete any existing collection to ensure clean start
    try:
        client.delete_collection(collection_name)
    except:
        pass

    # Create vector store
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        client=client,
        collection_name=collection_name
    )

    return vector_store, client

def get_vector_retriever(vector_store, search_kwargs=None):
    """
    Get a retriever from the vector store.

    Args:
        vector_store: Chroma vector store instance
        search_kwargs: Keyword arguments for the retriever (e.g., {"k": 5})

    Returns:
        Vector store retriever
    """
    if search_kwargs is None:
        search_kwargs = {"k": 5}
    return vector_store.as_retriever(search_kwargs=search_kwargs)

def clear_vector_store(persist_directory="chroma_db", collection_name="pdf_collection"):
    """
    Clear the vector store by deleting the collection and associated files.

    Args:
        persist_directory: Directory where the database is persisted
        collection_name: Name of the collection to delete

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Initialize Chroma client
        client = chromadb.PersistentClient(path=persist_directory)

        # Delete the collection
        try:
            client.delete_collection(collection_name)
        except Exception as e:
            # Collection might not exist, which is fine
            pass

        # Close the client by setting to None (resources will be cleaned up)
        client = None

        # Remove the persist directory with retries and force deletion (especially for Windows)
        max_retries = 5
        for attempt in range(max_retries):
            try:
                if os.path.exists(persist_directory):
                    # On Windows, we need to handle file locking explicitly
                    if os.name == 'nt':
                        os.system(f'rmdir /s /q "{persist_directory}"')
                    else:
                        shutil.rmtree(persist_directory, ignore_errors=True)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    return False
                time.sleep(2)

        return True
    except Exception as e:
        return False