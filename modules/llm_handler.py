"""
LLM Handler Module

This module handles LLM interactions for answer generation.
Supports OpenAI API and fallback to lightweight local LLMs.
"""

import os
from typing import Optional, List, Dict
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import local LLM libraries
try:
    # Try newer langchain_community first
    try:
        from langchain_community.llms import Ollama
    except ImportError:
        # Fallback to older langchain.llms
        from langchain.llms import Ollama
    
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class LLMHandler:
    """Handles LLM interactions for answer generation."""
    
    def __init__(self, use_openai: bool = True):
        """
        Initialize the LLM handler.
        
        Args:
            use_openai: Whether to use OpenAI API (default: True)
        """
        self.use_openai = use_openai
        self.openai_client = None
        self.local_llm = None
        self.llm_type = None
        
        # Initialize OpenAI if available and requested
        if use_openai and OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    self.openai_client = OpenAI(api_key=api_key)
                    self.llm_type = "openai"
                    st.success("✓ Using OpenAI GPT for answer generation")
                except Exception as e:
                    st.warning(f"Failed to initialize OpenAI: {str(e)}")
                    self.use_openai = False
            else:
                st.warning("OpenAI API key not found. Trying local LLM fallback...")
                self.use_openai = False
        
        # Initialize local LLM as fallback
        if not self.use_openai:
            # Try Ollama first (lightweight and fast)
            if OLLAMA_AVAILABLE:
                try:
                    self.local_llm = Ollama(
                        model="llama2",
                        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
                    )
                    self.llm_type = "ollama"
                    st.info("✓ Using Ollama (local LLM) for answer generation")
                except Exception as e:
                    st.warning(f"Ollama not available: {str(e)}")
            
            # Fallback to transformers with a small model
            if not self.local_llm and TRANSFORMERS_AVAILABLE:
                try:
                    # Use a small, fast model
                    model_name = "gpt2"  # Very lightweight
                    self.local_llm = pipeline(
                        "text-generation",
                        model=model_name,
                        tokenizer=model_name,
                        max_length=512,
                        device=-1  # CPU
                    )
                    self.llm_type = "transformers"
                    st.info("✓ Using local transformers model (GPT-2) for answer generation")
                except Exception as e:
                    st.warning(f"Transformers model not available: {str(e)}")
            
            if not self.local_llm:
                st.error("No LLM available. Please install Ollama or set up OpenAI API key.")
    
    def generate_answer(
        self,
        question: str,
        context: str,
        max_tokens: int = 500
    ) -> Optional[str]:
        """
        Generate an answer to a question based on context.
        
        Args:
            question: User's question
            context: Retrieved context from vector store
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated answer, or None if generation fails
        """
        if not question or not context:
            return None
        
        try:
            if self.use_openai and self.openai_client:
                return self._generate_openai_answer(question, context, max_tokens)
            elif self.local_llm:
                return self._generate_local_answer(question, context, max_tokens)
            else:
                st.error("No LLM available for answer generation.")
                return None
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            return None
    
    def _generate_openai_answer(
        self,
        question: str,
        context: str,
        max_tokens: int
    ) -> str:
        """Generate answer using OpenAI API."""
        prompt = f"""Based on the following context, answer the question. 
If the answer is not in the context, say "I don't have enough information to answer this question based on the provided context."

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"OpenAI API error: {str(e)}")
            raise
    
    def _generate_local_answer(
        self,
        question: str,
        context: str,
        max_tokens: int
    ) -> str:
        """Generate answer using local LLM."""
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        if self.llm_type == "ollama":
            try:
                response = self.local_llm(prompt)
                return response.strip()
            except Exception as e:
                st.error(f"Ollama error: {str(e)}")
                return "Error generating answer with local LLM."
        
        elif self.llm_type == "transformers":
            try:
                # For transformers pipeline
                response = self.local_llm(
                    prompt,
                    max_length=len(prompt.split()) + max_tokens,
                    num_return_sequences=1,
                    pad_token_id=self.local_llm.tokenizer.eos_token_id
                )
                generated_text = response[0]['generated_text']
                # Extract only the answer part
                answer = generated_text.split("Answer:")[-1].strip()
                return answer
            except Exception as e:
                st.error(f"Transformers error: {str(e)}")
                return "Error generating answer with local LLM."
        
        else:
            return "No local LLM available."
    
    def is_available(self) -> bool:
        """Check if any LLM is available."""
        return (self.use_openai and self.openai_client is not None) or self.local_llm is not None

