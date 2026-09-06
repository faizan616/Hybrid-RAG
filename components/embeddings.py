import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings

@st.cache_resource
def get_embeddings_model(model_name="nomic-embed-text:latest"):
    """
    Get the embeddings model.

    Args:
        model_name: Name of the Ollama embedding model

    Returns:
        OllamaEmbeddings instance
    """
    return OllamaEmbeddings(model=model_name)