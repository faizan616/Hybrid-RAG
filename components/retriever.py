from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

def create_hybrid_retriever(bm25_retriever, vector_retriever, hybrid_search_ratio=0.5):
    """
    Create a hybrid retriever combining BM25 and vector retrievers.

    Args:
        bm25_retriever: BM25 retriever instance
        vector_retriever: Vector store retriever instance
        hybrid_search_ratio: Weight for vector retriever (0.0 to 1.0),
                             where (1 - hybrid_search_ratio) is weight for BM25

    Returns:
        EnsembleRetriever instance
    """
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[1 - hybrid_search_ratio, hybrid_search_ratio],
    )
    return ensemble_retriever

def create_bm25_retriever(documents, k=5):
    """
    Create a BM25 retriever from documents.

    Args:
        documents: List of Document objects
        k: Number of documents to retrieve

    Returns:
        BM25Retriever instance
    """
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k
    return retriever