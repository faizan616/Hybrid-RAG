from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into chunks.

    Args:
        documents: List of Document objects to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of chunked Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)