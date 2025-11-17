# 🚀 Quick Start Guide

## Installation (3 Steps)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Set OpenAI API Key
Create a `.env` file:
```
OPENAI_API_KEY=your_key_here
```

**Note**: Without OpenAI API key, the app uses local embeddings and LLMs automatically.

### 3. Run the App
```bash
streamlit run app.py
```

That's it! The app will open at `http://localhost:8501`

## First Time Usage

1. **Upload PDFs**: Go to "📄 Upload & Process PDFs" → Upload files → Click "Process PDFs"
2. **Ask Questions**: Go to "❓ Ask Questions" → Type your question → Click "Ask Question"
3. **View Status**: Go to "📊 Knowledge Base Status" to see document count

## Troubleshooting

- **No embeddings?** → Install: `pip install sentence-transformers`
- **No LLM?** → Either set `OPENAI_API_KEY` or install Ollama
- **PDF errors?** → Ensure PDF has extractable text (not just images)

For detailed documentation, see [README.md](README.md)

