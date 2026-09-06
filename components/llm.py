import streamlit as st
from langchain_ollama import OllamaLLM

@st.cache_resource
def get_llm(model_name="qwen2.5:latest", temperature=0.7):
    """
    Get the LLM model.

    Args:
        model_name: Name of the Ollama model
        temperature: Temperature for generation

    Returns:
        OllamaLLM instance
    """
    return OllamaLLM(model=model_name, temperature=temperature)