"""
RAG Pipeline Module

This module orchestrates the complete RAG (Retrieval-Augmented Generation) pipeline.
Combines vector store retrieval with LLM answer generation.
"""

from typing import List, Dict, Optional, Tuple
from modules.embeddings import EmbeddingsGenerator
from modules.vectorstore import VectorStore
from modules.llm_handler import LLMHandler
import streamlit as st


class RAGPipeline:
    """Orchestrates the complete RAG pipeline."""
    
    def __init__(
        self,
        vectorstore: VectorStore,
        embeddings_generator: EmbeddingsGenerator,
        llm_handler: LLMHandler,
        top_k: int = 5
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            vectorstore: VectorStore instance
            embeddings_generator: EmbeddingsGenerator instance
            llm_handler: LLMHandler instance
            top_k: Number of top documents to retrieve
        """
        self.vectorstore = vectorstore
        self.embeddings_generator = embeddings_generator
        self.llm_handler = llm_handler
        self.top_k = top_k
    
    def query(
        self,
        question: str,
        max_tokens: int = 500
    ) -> Dict[str, any]:
        """
        Process a query through the RAG pipeline.
        
        Args:
            question: User's question
            max_tokens: Maximum tokens in the answer
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        if not question or not question.strip():
            return {
                "answer": "Please provide a valid question.",
                "sources": [],
                "metadata": {}
            }
        
        try:
            # Step 1: Generate embedding for the question
            with st.spinner("Generating query embedding..."):
                query_embeddings = self.embeddings_generator.generate_embeddings([question])
            
            if not query_embeddings or len(query_embeddings) == 0:
                return {
                    "answer": "Error generating query embedding.",
                    "sources": [],
                    "metadata": {}
                }
            
            query_embedding = query_embeddings[0]
            
            # Step 2: Retrieve relevant documents from vector store
            with st.spinner(f"Retrieving top {self.top_k} relevant passages..."):
                search_results = self.vectorstore.search(
                    query_embedding=query_embedding,
                    n_results=self.top_k
                )
            
            if not search_results:
                return {
                    "answer": "No relevant documents found in the knowledge base. Please upload and process PDFs first.",
                    "sources": [],
                    "metadata": {}
                }
            
            # Step 3: Prepare context from retrieved documents
            context_parts = []
            sources = []
            
            for i, result in enumerate(search_results, 1):
                doc_text = result.get("document", "")
                distance = result.get("distance", 0)
                metadata = result.get("metadata", {})
                
                # Calculate similarity score (1 - distance for cosine similarity)
                similarity = 1 - distance if distance is not None else 0
                
                context_parts.append(f"[Source {i}] {doc_text}")
                sources.append({
                    "text": doc_text[:200] + "..." if len(doc_text) > 200 else doc_text,
                    "similarity": round(similarity, 3),
                    "metadata": metadata
                })
            
            context = "\n\n".join(context_parts)
            
            # Step 4: Generate answer using LLM
            with st.spinner("Generating answer..."):
                answer = self.llm_handler.generate_answer(
                    question=question,
                    context=context,
                    max_tokens=max_tokens
                )
            
            if not answer:
                return {
                    "answer": "Error generating answer. Please try again.",
                    "sources": sources,
                    "metadata": {}
                }
            
            # Step 5: Return results
            return {
                "answer": answer,
                "sources": sources,
                "metadata": {
                    "num_sources": len(sources),
                    "top_k": self.top_k
                }
            }
            
        except Exception as e:
            st.error(f"Error in RAG pipeline: {str(e)}")
            return {
                "answer": f"An error occurred: {str(e)}",
                "sources": [],
                "metadata": {}
            }
    
    def add_documents_to_knowledge_base(
        self,
        texts: List[str],
        metadatas: List[Dict] = None
    ) -> bool:
        """
        Add documents to the knowledge base.
        
        Args:
            texts: List of text chunks to add
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embeddings
            with st.spinner("Generating embeddings for documents..."):
                embeddings = self.embeddings_generator.generate_embeddings(texts)
            
            if not embeddings:
                st.error("Failed to generate embeddings.")
                return False
            
            # Generate IDs
            existing_count = self.vectorstore.get_collection_count()
            ids = [f"doc_{existing_count + i}" for i in range(len(texts))]
            
            # Add to vector store
            success = self.vectorstore.add_documents(
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            return success
            
        except Exception as e:
            st.error(f"Error adding documents to knowledge base: {str(e)}")
            return False

