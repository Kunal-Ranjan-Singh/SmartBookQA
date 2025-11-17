# 📚 SmartBookQA

A complete end-to-end RAG (Retrieval-Augmented Generation) application for intelligent PDF question-answering. Built with Python, Streamlit, ChromaDB, LangChain, and OpenAI (with local fallback support).

## ✨ Features

- **PDF Upload & Processing**: Upload multiple PDF files and extract text automatically
- **Intelligent Chunking**: Smart text splitting with configurable chunk size and overlap
- **Vector Storage**: Persistent ChromaDB vector store for efficient similarity search
- **Embedding Generation**: OpenAI embeddings with automatic fallback to local sentence-transformers
- **RAG Pipeline**: Complete retrieval-augmented generation with context-aware answers
- **LLM Support**: OpenAI GPT with fallback to local LLMs (Ollama/Transformers)
- **Source Attribution**: Shows which passages were used to generate each answer
- **Clean UI**: Modern, intuitive Streamlit interface

## 🏗️ Project Structure

```
SmartBookQA/
├── app.py                      # Main Streamlit application
├── modules/
│   ├── pdf_processor.py        # PDF text extraction
│   ├── text_chunker.py         # Text chunking logic
│   ├── embeddings.py           # Embedding generation
│   ├── vectorstore.py          # ChromaDB vector store management
│   ├── llm_handler.py          # LLM interaction (OpenAI + local fallback)
│   └── rag_pipeline.py         # Complete RAG pipeline orchestration
├── data/                       # Uploaded PDFs (created automatically)
├── vectorstore/                # ChromaDB persistence (created automatically)
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
└── README.md                   # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download the Project

```bash
cd SmartBookQA
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables (Optional)

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env
```

Edit `.env` and add your OpenAI API key (optional):

```
OPENAI_API_KEY=your_openai_api_key_here
```

**Note**: If you don't provide an OpenAI API key, the app will automatically use local embeddings (sentence-transformers) and attempt to use local LLMs.

## 🎯 Usage

### Starting the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Using SmartBookQA

1. **Upload PDFs**
   - Navigate to "📄 Upload & Process PDFs"
   - Click "Choose PDF files" and select one or more PDFs
   - Click "🔄 Process PDFs" to extract text and create embeddings
   - Wait for processing to complete

2. **Ask Questions**
   - Navigate to "❓ Ask Questions"
   - Enter your question in the text input
   - Adjust advanced options if needed (number of sources, answer length)
   - Click "🔍 Ask Question"
   - View the answer and source passages used

3. **Check Status**
   - Navigate to "📊 Knowledge Base Status"
   - View statistics about your knowledge base
   - Clear documents if needed

## 🔧 Configuration

### Embedding Models

- **OpenAI** (default if API key provided): `text-embedding-3-small` (1536 dimensions)
- **Local Fallback**: `all-MiniLM-L6-v2` (384 dimensions)

### LLM Models

- **OpenAI** (default if API key provided): `gpt-3.5-turbo`
- **Local Fallback Options**:
  - Ollama (if installed): `llama2`
  - Transformers: `gpt2` (lightweight)

### Chunking Parameters

Default settings in `modules/text_chunker.py`:
- Chunk size: 1000 characters
- Chunk overlap: 200 characters

You can modify these in the `TextChunker` initialization.

## 🛠️ Development

### Module Overview

- **`pdf_processor.py`**: Handles PDF file reading and text extraction using PyPDF2
- **`text_chunker.py`**: Splits text into chunks using LangChain's RecursiveCharacterTextSplitter
- **`embeddings.py`**: Generates embeddings using OpenAI or sentence-transformers
- **`vectorstore.py`**: Manages ChromaDB operations (add, search, delete documents)
- **`llm_handler.py`**: Handles LLM interactions with OpenAI and local fallbacks
- **`rag_pipeline.py`**: Orchestrates the complete RAG workflow

### Adding Custom Features

1. **Custom Chunking**: Modify `TextChunker` class in `modules/text_chunker.py`
2. **Different Embeddings**: Update `EmbeddingsGenerator` in `modules/embeddings.py`
3. **Alternative Vector Store**: Replace `VectorStore` class in `modules/vectorstore.py`
4. **Custom LLM**: Extend `LLMHandler` in `modules/llm_handler.py`

## 📦 Dependencies

### Core Dependencies

- `streamlit`: Web framework
- `PyPDF2`: PDF processing
- `chromadb`: Vector database
- `langchain`: Text splitting and LLM utilities
- `openai`: OpenAI API client
- `tiktoken`: Token counting
- `python-dotenv`: Environment variable management

### Optional Dependencies

- `sentence-transformers`: Local embeddings (automatic fallback)
- `transformers`: Local LLM support
- `torch`: PyTorch for transformers
- `ollama`: Ollama LLM integration

## 🔍 Troubleshooting

### Issue: "No embedding model available"

**Solution**: Install sentence-transformers:
```bash
pip install sentence-transformers
```

### Issue: "No LLM available"

**Solution**: Either:
1. Set `OPENAI_API_KEY` in `.env` file, or
2. Install Ollama and ensure it's running, or
3. Install transformers (will use GPT-2 as fallback)

### Issue: PDF extraction fails

**Solution**: 
- Ensure PDF is not password-protected
- Check if PDF contains extractable text (not just images)
- Try a different PDF file

### Issue: ChromaDB errors

**Solution**: 
- Delete the `vectorstore/` directory and restart
- Ensure write permissions in the project directory

## 🚢 Deployment

### Local Deployment

Simply run:
```bash
streamlit run app.py
```

### Cloud Deployment (Streamlit Cloud)

1. Push your code to GitHub
2. Connect your repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add environment variables in Streamlit Cloud settings:
   - `OPENAI_API_KEY` (if using OpenAI)
4. Deploy!

### Docker Deployment (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t smartbookqa .
docker run -p 8501:8501 smartbookqa
```

## 📝 License

This project is open source and available for educational and commercial use.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues, questions, or suggestions, please open an issue on the project repository.

---

**Built with ❤️ using Python, Streamlit, ChromaDB, and LangChain**

