"""
Vector Store Module

This module handles ChromaDB vector store operations for persistent storage.
Manages document embeddings, retrieval, and similarity search.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Tuple
import os
import streamlit as st


class VectorStore:
    """Manages ChromaDB vector store operations."""
    
    def __init__(self, persist_directory: str = "./vectorstore", collection_name: str = "smartbookqa"):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
        except Exception as e:
            st.error(f"Error initializing vector store: {str(e)}")
            raise
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict] = None,
        ids: List[str] = None
    ) -> bool:
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text chunks
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries (optional)
            ids: List of document IDs (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not texts or not embeddings:
                st.error("No texts or embeddings provided.")
                return False
            
            if len(texts) != len(embeddings):
                st.error("Mismatch between number of texts and embeddings.")
                return False
            
            # Generate IDs if not provided
            if ids is None:
                ids = [f"doc_{i}" for i in range(len(texts))]
            
            # Create default metadata if not provided
            if metadatas is None:
                metadatas = [{"text": text[:100]} for text in texts]
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            return True
            
        except Exception as e:
            st.error(f"Error adding documents to vector store: {str(e)}")
            return False
    
    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        filter_metadata: Dict = None
    ) -> List[Dict]:
        """
        Search for similar documents in the vector store.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of dictionaries containing documents, distances, and metadata
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and len(results['documents'][0]) > 0:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        "document": results['documents'][0][i],
                        "distance": results['distances'][0][i] if results['distances'] else None,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "id": results['ids'][0][i] if results['ids'] else None
                    })
            
            return formatted_results
            
        except Exception as e:
            st.error(f"Error searching vector store: {str(e)}")
            return []
    
    def get_collection_count(self) -> int:
        """
        Get the number of documents in the collection.
        
        Returns:
            Number of documents
        """
        try:
            return self.collection.count()
        except Exception:
            return 0
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            return True
        except Exception as e:
            st.error(f"Error clearing collection: {str(e)}")
            return False
    
    def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete specific documents from the collection.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=ids)
            return True
        except Exception as e:
            st.error(f"Error deleting documents: {str(e)}")
            return False

