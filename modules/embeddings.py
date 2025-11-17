"""
Embeddings Generation Module

This module handles embedding generation for text chunks.
Supports OpenAI embeddings and fallback to local embeddings.
"""

from typing import List, Optional
import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class EmbeddingsGenerator:
    """Handles embedding generation for text chunks."""
    
    def __init__(self, use_openai: bool = True):
        """
        Initialize the embeddings generator.
        
        Args:
            use_openai: Whether to use OpenAI embeddings (default: True)
        """
        self.use_openai = use_openai
        self.openai_client = None
        self.local_model = None
        
        # Initialize OpenAI if available and requested
        if use_openai and OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    self.openai_client = OpenAI(api_key=api_key)
                    st.success("✓ Using OpenAI embeddings")
                except Exception as e:
                    st.warning(f"Failed to initialize OpenAI: {str(e)}")
                    self.use_openai = False
            else:
                st.warning("OpenAI API key not found. Falling back to local embeddings.")
                self.use_openai = False
        
        # Initialize local model as fallback
        if not self.use_openai:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    # Use a lightweight, fast model
                    self.local_model = SentenceTransformer('all-MiniLM-L6-v2')
                    st.info("✓ Using local embeddings (sentence-transformers)")
                except Exception as e:
                    st.error(f"Failed to load local embedding model: {str(e)}")
            else:
                st.error("No embedding model available. Please install sentence-transformers.")
    
    def generate_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors, or None if generation fails
        """
        if not texts:
            return None
        
        try:
            if self.use_openai and self.openai_client:
                return self._generate_openai_embeddings(texts)
            elif self.local_model:
                return self._generate_local_embeddings(texts)
            else:
                st.error("No embedding model available.")
                return None
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            return None
    
    def _generate_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            st.error(f"OpenAI embedding error: {str(e)}")
            # Fallback to local if OpenAI fails
            if self.local_model:
                st.info("Falling back to local embeddings...")
                return self._generate_local_embeddings(texts)
            raise
    
    def _generate_local_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local sentence-transformers model."""
        if not self.local_model:
            raise ValueError("Local embedding model not initialized")
        
        embeddings = self.local_model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings.
        
        Returns:
            Embedding dimension
        """
        if self.use_openai and self.openai_client:
            # OpenAI text-embedding-3-small has 1536 dimensions
            return 1536
        elif self.local_model:
            # all-MiniLM-L6-v2 has 384 dimensions
            return 384
        else:
            return 0

