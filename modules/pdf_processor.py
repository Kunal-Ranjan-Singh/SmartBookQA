"""
PDF Processing Module

This module handles PDF file extraction and text processing.
Supports PyPDF2 for extracting text from PDF documents.
"""

import PyPDF2
import io
from typing import List, Optional
import streamlit as st


class PDFProcessor:
    """Handles PDF text extraction and processing."""
    
    def __init__(self):
        """Initialize the PDF processor."""
        pass
    
    def extract_text_from_pdf(self, pdf_file) -> Optional[str]:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_file: File-like object or bytes containing PDF data
            
        Returns:
            Extracted text as string, or None if extraction fails
        """
        try:
            # Handle Streamlit UploadedFile
            if hasattr(pdf_file, 'read'):
                pdf_file.seek(0)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
            else:
                # Handle bytes
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
            
            text_content = []
            
            # Extract text from each page
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(f"--- Page {page_num + 1} ---\n{page_text}")
                except Exception as e:
                    st.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                    continue
            
            if not text_content:
                st.error("No text could be extracted from the PDF.")
                return None
            
            full_text = "\n\n".join(text_content)
            return full_text
            
        except PyPDF2.errors.PdfReadError as e:
            st.error(f"Error reading PDF file: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Unexpected error processing PDF: {str(e)}")
            return None
    
    def validate_pdf(self, pdf_file) -> bool:
        """
        Validate if the uploaded file is a valid PDF.
        
        Args:
            pdf_file: File-like object or bytes containing PDF data
            
        Returns:
            True if valid PDF, False otherwise
        """
        try:
            if hasattr(pdf_file, 'read'):
                pdf_file.seek(0)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
            else:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
            
            # Check if PDF has at least one page
            return len(pdf_reader.pages) > 0
            
        except Exception:
            return False

