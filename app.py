"""
SmartBookQA - Main Streamlit Application

A complete RAG (Retrieval-Augmented Generation) application for PDF question-answering.
"""

import streamlit as st
import os
from pathlib import Path
from modules.pdf_processor import PDFProcessor
from modules.text_chunker import TextChunker
from modules.embeddings import EmbeddingsGenerator
from modules.vectorstore import VectorStore
from modules.llm_handler import LLMHandler
from modules.rag_pipeline import RAGPipeline

# Page configuration
st.set_page_config(
    page_title="SmartBookQA",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .source-box {
        padding: 0.75rem;
        border-left: 4px solid #1f77b4;
        background-color: #f9f9f9;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.vectorstore = None
    st.session_state.embeddings_generator = None
    st.session_state.llm_handler = None
    st.session_state.rag_pipeline = None
    st.session_state.documents_loaded = False
    st.session_state.uploaded_files = []


def initialize_components():
    """Initialize all RAG components."""
    if st.session_state.initialized:
        return
    
    try:
        with st.spinner("Initializing SmartBookQA components..."):
            # Initialize embeddings generator
            use_openai = os.getenv("OPENAI_API_KEY") is not None
            st.session_state.embeddings_generator = EmbeddingsGenerator(use_openai=use_openai)
            
            # Initialize vector store
            st.session_state.vectorstore = VectorStore(
                persist_directory="./vectorstore",
                collection_name="smartbookqa"
            )
            
            # Initialize LLM handler
            st.session_state.llm_handler = LLMHandler(use_openai=use_openai)
            
            # Initialize RAG pipeline
            st.session_state.rag_pipeline = RAGPipeline(
                vectorstore=st.session_state.vectorstore,
                embeddings_generator=st.session_state.embeddings_generator,
                llm_handler=st.session_state.llm_handler,
                top_k=5
            )
            
            st.session_state.initialized = True
            st.success("✓ Components initialized successfully!")
            
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        st.stop()


def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">📚 SmartBookQA</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent PDF Question-Answering System</p>', unsafe_allow_html=True)
    
    # Initialize components
    initialize_components()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        ["📄 Upload & Process PDFs", "❓ Ask Questions", "📊 Knowledge Base Status"]
    )
    
    # Route to appropriate page
    if page == "📄 Upload & Process PDFs":
        upload_and_process_page()
    elif page == "❓ Ask Questions":
        ask_questions_page()
    elif page == "📊 Knowledge Base Status":
        status_page()


def upload_and_process_page():
    """Page for uploading and processing PDFs."""
    st.header("📄 Upload & Process PDFs")
    
    st.markdown("""
    <div class="info-box">
        <strong>Instructions:</strong><br>
        1. Upload one or more PDF files<br>
        2. Click "Process PDFs" to extract text and create embeddings<br>
        3. Documents will be stored in the vector database for question-answering
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"📎 {len(uploaded_files)} file(s) selected")
        
        # Process button
        if st.button("🔄 Process PDFs", type="primary", use_container_width=True):
            process_pdfs(uploaded_files)
    
    # Show processing history
    if st.session_state.uploaded_files:
        st.subheader("📋 Processing History")
        for file_info in st.session_state.uploaded_files:
            st.success(f"✓ {file_info['name']} - {file_info['chunks']} chunks processed")


def process_pdfs(uploaded_files):
    """Process uploaded PDF files."""
    pdf_processor = PDFProcessor()
    text_chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
    
    all_chunks = []
    processed_files = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        progress_bar.progress((idx + 1) / len(uploaded_files))
        
        # Extract text from PDF
        text = pdf_processor.extract_text_from_pdf(uploaded_file)
        
        if text:
            # Chunk the text
            chunks = text_chunker.chunk_text(text)
            
            if chunks:
                all_chunks.extend(chunks)
                processed_files.append({
                    "name": uploaded_file.name,
                    "chunks": len(chunks),
                    "text_length": len(text)
                })
                st.success(f"✓ {uploaded_file.name}: {len(chunks)} chunks created")
            else:
                st.warning(f"⚠ {uploaded_file.name}: No chunks created")
        else:
            st.error(f"✗ {uploaded_file.name}: Failed to extract text")
    
    # Add chunks to knowledge base
    if all_chunks:
        status_text.text("Adding documents to knowledge base...")
        
        # Create metadata for each chunk
        metadatas = [{"source": "uploaded_pdf", "chunk_index": i} for i in range(len(all_chunks))]
        
        success = st.session_state.rag_pipeline.add_documents_to_knowledge_base(
            texts=all_chunks,
            metadatas=metadatas
        )
        
        if success:
            st.session_state.uploaded_files.extend(processed_files)
            st.session_state.documents_loaded = True
            st.success(f"✅ Successfully processed {len(processed_files)} file(s) with {len(all_chunks)} total chunks!")
        else:
            st.error("Failed to add documents to knowledge base.")
    
    progress_bar.empty()
    status_text.empty()


def ask_questions_page():
    """Page for asking questions."""
    st.header("❓ Ask Questions")
    
    # Check if documents are loaded
    doc_count = st.session_state.vectorstore.get_collection_count()
    
    if doc_count == 0:
        st.warning("⚠️ No documents in knowledge base. Please upload and process PDFs first.")
        st.info("Go to 'Upload & Process PDFs' page to add documents.")
        return
    
    st.info(f"📚 Knowledge base contains {doc_count} document chunks. You can now ask questions!")
    
    # Question input
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is the main topic discussed in the document?",
        key="question_input"
    )
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        top_k = st.slider("Number of source passages to retrieve", 3, 10, 5)
        max_tokens = st.slider("Maximum answer length (tokens)", 200, 1000, 500)
        
        # Update RAG pipeline top_k
        st.session_state.rag_pipeline.top_k = top_k
    
    # Ask button
    if st.button("🔍 Ask Question", type="primary", use_container_width=True):
        if question:
            answer_question(question, max_tokens)
        else:
            st.warning("Please enter a question.")


def answer_question(question: str, max_tokens: int):
    """Process and answer a question."""
    # Display question
    st.markdown("### ❓ Your Question")
    st.info(question)
    
    # Get answer from RAG pipeline
    result = st.session_state.rag_pipeline.query(question, max_tokens=max_tokens)
    
    # Display answer
    st.markdown("### 💡 Answer")
    st.markdown(f"<div class='info-box'>{result['answer']}</div>", unsafe_allow_html=True)
    
    # Display sources
    if result['sources']:
        st.markdown("### 📑 Source Passages")
        st.info(f"Retrieved {len(result['sources'])} relevant passage(s)")
        
        for i, source in enumerate(result['sources'], 1):
            similarity = source.get('similarity', 0)
            text = source.get('text', '')
            
            st.markdown(f"""
            <div class="source-box">
                <strong>Source {i}</strong> (Similarity: {similarity:.1%})<br>
                {text}
            </div>
            """, unsafe_allow_html=True)


def status_page():
    """Page showing knowledge base status."""
    st.header("📊 Knowledge Base Status")
    
    # Get collection count
    doc_count = st.session_state.vectorstore.get_collection_count()
    
    # Status metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Documents", doc_count)
    
    with col2:
        embedding_dim = st.session_state.embeddings_generator.get_embedding_dimension()
        st.metric("Embedding Dimension", embedding_dim)
    
    with col3:
        llm_status = "Available" if st.session_state.llm_handler.is_available() else "Not Available"
        st.metric("LLM Status", llm_status)
    
    # Vector store info
    st.subheader("Vector Store Information")
    st.info(f"**Persist Directory:** `./vectorstore`")
    st.info(f"**Collection Name:** `smartbookqa`")
    
    # Clear database option
    st.subheader("Database Management")
    
    if st.button("🗑️ Clear All Documents", type="secondary"):
        if st.session_state.vectorstore.clear_collection():
            st.session_state.documents_loaded = False
            st.session_state.uploaded_files = []
            st.success("✅ Knowledge base cleared successfully!")
            st.rerun()
        else:
            st.error("Failed to clear knowledge base.")


if __name__ == "__main__":
    main()

