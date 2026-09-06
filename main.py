import os
import uuid
import shutil

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import chromadb

# Import our separated components
from components.chunking import chunk_documents
from components.embeddings import get_embeddings_model
from components.vector_store import initialize_vector_store, get_vector_retriever, clear_vector_store
from components.retriever import create_hybrid_retriever, create_bm25_retriever
from components.llm import get_llm

# Create directory for uploaded files if it doesn't exist
UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "client" not in st.session_state:
    st.session_state.client = None
if "bm25_retriever" not in st.session_state:
    st.session_state.bm25_retriever = None
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "clear_flag" not in st.session_state:
    st.session_state.clear_flag = False

# Page configuration
st.set_page_config(page_title="Hybrid RAG PDF Q&A", page_icon="📚")
st.title("📚 Hybrid RAG PDF Q&A Chatbot")

# Sidebar for PDF upload and settings
with st.sidebar:
    st.header("Configuration")

    # File uploader - we'll control this with a key that changes when we clear
    uploader_key = "file_uploader_" + str(st.session_state.get("uploader_counter", 0))
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=uploader_key
    )

    temperature = st.slider(
        "Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1
    )
    hybrid_search_ratio = st.slider(
        "Hybrid Search Ratio (Vector vs Keyword)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
    )

    # Add clear documents button
    if st.button("Clear Documents"):
        try:
            # Reset session state first to release resources
            st.session_state.vector_store = None
            st.session_state.bm25_retriever = None
            st.session_state.documents_processed = False
            st.session_state.messages = []
            st.session_state.clear_flag = True
            if st.session_state.client:
                try:
                    st.session_state.client = None
                except:
                    pass

            # Clear vector store using the component function
            if not clear_vector_store():
                st.warning("Warning: Some vector store files may not have been completely removed.")

            # Remove all uploaded PDFs
            if os.path.exists(UPLOAD_DIR):
                shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
                os.makedirs(UPLOAD_DIR, exist_ok=True)

            # Increment the uploader key to reset the file uploader
            st.session_state.uploader_counter = st.session_state.get("uploader_counter", 0) + 1
            st.session_state.clear_flag = False

            st.success("All documents have been completely cleared. You can upload new documents.")
            st.rerun()
        except Exception as e:
            st.error(f"Error clearing documents: {str(e)}")

# Initialize components using separated functions
llm = get_llm(temperature=temperature)
embeddings = get_embeddings_model()

def process_documents(uploaded_files):
    if not uploaded_files or st.session_state.clear_flag:
        return None

    # Save uploaded files and load them
    docs = []

    for uploaded_file in uploaded_files:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")

        # Save the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Load and split the PDF
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()  # Load pages first

        # Chunk the documents
        chunked_docs = chunk_documents(pages, chunk_size=1000, chunk_overlap=200)

        # Add source metadata
        for doc in chunked_docs:
            doc.metadata["source"] = uploaded_file.name

        docs.extend(chunked_docs)

    return docs

# Only process documents if we have files and haven't just cleared
if uploaded_files and not st.session_state.documents_processed and not st.session_state.clear_flag:
    with st.spinner("Processing documents..."):
        docs = process_documents(uploaded_files)

        if docs:
            try:
                # Initialize vector store
                vector_store, client = initialize_vector_store(
                    documents=docs,
                    embedding_model=embeddings,
                    persist_directory="chroma_db",
                    collection_name="pdf_collection"
                )
                st.session_state.vector_store = vector_store
                st.session_state.client = client

                # Create BM25 retriever
                bm25_retriever = create_bm25_retriever(docs, k=5)
                st.session_state.bm25_retriever = bm25_retriever

                st.session_state.documents_processed = True
                st.rerun()
            except Exception as e:
                st.error(f"Failed to initialize ChromaDB: {str(e)}")
                st.stop()

# Define RAG chain
def get_rag_chain(llm):
    template = """Answer the question based on the following context:

    Context:
    {docs}

    Question: {question}

    Answer in detail, citing sources when available. If you don't know the answer, say you don't know:"""

    prompt = ChatPromptTemplate.from_template(template)

    return (
        {"docs": RunnablePassthrough(),
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def format_docs(docs):
    if not docs:
        return "No relevant documents found"
    return "\n\n".join(f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}" for doc in docs)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        if st.session_state.vector_store is None:
            st.error("Please upload and process documents first.")
            st.stop()

        # Create hybrid retriever with fallback
        try:
            vector_retriever = get_vector_retriever(
                st.session_state.vector_store,
                search_kwargs={"k": 5}
            )
            ensemble_retriever = create_hybrid_retriever(
                st.session_state.bm25_retriever,
                vector_retriever,
                hybrid_search_ratio=hybrid_search_ratio
            )

            # Retrieve relevant documents with error handling
            with st.spinner("Searching documents..."):
                retrieved_docs = ensemble_retriever.invoke(prompt)

                print(retrieved_docs)

                if not retrieved_docs:
                    # Fallback to simple vector search if hybrid fails
                    retrieved_docs = vector_retriever.invoke(prompt)

            # Prepare the RAG chain
            rag_chain = get_rag_chain(llm)

            # Format documents for context
            formatted_docs = format_docs(retrieved_docs)

            # Stream the response
            response = st.write_stream(
                rag_chain.stream({
                    "question": prompt,
                    "docs": formatted_docs
                })
            )

            # Add retrieval details to the message
            retrieval_details = {
                "retrieved_docs": [{
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get("source", "unknown")
                } for doc in retrieved_docs] if retrieved_docs else []
            }

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            response = "I encountered an error while processing your request. Please try again."
            retrieval_details = {"retrieved_docs": []}

        # Store the complete response with retrieval details
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "retrieval_details": retrieval_details
        })