"""
Text Chunking Module

This module handles text splitting and chunking for RAG applications.
Uses LangChain's text splitters for intelligent chunking.
"""

# Handle different LangChain versions
try:
    # LangChain 1.0+ uses separate package
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        # LangChain 0.1.0+ (but < 1.0)
        from langchain.text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        try:
            # Older LangChain versions
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            raise ImportError(
                "Could not import RecursiveCharacterTextSplitter. "
                "Please install: pip install langchain-text-splitters"
            )

from typing import List
import tiktoken


class TextChunker:
    """Handles text chunking for vector storage."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None
    ):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk (in characters)
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to use for splitting (default: None)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Default separators for intelligent splitting
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len
        )
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        chunks = self.text_splitter.split_text(text)
        return chunks
    
    def chunk_documents(self, documents: List[str]) -> List[dict]:
        """
        Split documents into chunks with metadata.
        
        Args:
            documents: List of document texts
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            chunks = self.chunk_text(doc)
            
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk,
                    "document_id": doc_idx,
                    "chunk_id": chunk_idx,
                    "chunk_size": len(chunk)
                })
        
        return all_chunks
    
    def estimate_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """
        Estimate the number of tokens in text using tiktoken.
        
        Args:
            text: Input text
            model: Model name for token encoding
            
        Returns:
            Estimated token count
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
            tokens = encoding.encode(text)
            return len(tokens)
        except Exception:
            # Fallback: rough estimate (1 token ≈ 4 characters)
            return len(text) // 4

