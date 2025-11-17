# 🚀 Deploy SmartBookQA to Streamlit Cloud

This guide will walk you through deploying your SmartBookQA application to Streamlit Cloud.

## Prerequisites

1. A GitHub account
2. Your code pushed to a GitHub repository
3. A Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

## Step-by-Step Deployment

### Step 1: Prepare Your Repository

1. **Initialize Git** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit: SmartBookQA RAG application"
   ```

2. **Create a GitHub Repository**:
   - Go to [GitHub](https://github.com) and create a new repository
   - Name it something like `SmartBookQA` or `smartbookqa`
   - **Do NOT** initialize with README, .gitignore, or license (we already have these)

3. **Push Your Code**:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Create New App**:
   - Click "New app" button
   - Select your GitHub repository
   - Choose the branch (usually `main` or `master`)
   - Set the main file path: `app.py`
   - Click "Deploy"

### Step 3: Configure Environment Variables

1. **Open App Settings**:
   - Click the "⋮" (three dots) menu next to your app
   - Select "Settings"

2. **Add Environment Variables**:
   - Click "Secrets" tab
   - Add your OpenAI API key (if using OpenAI):
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Save and Restart**:
   - Click "Save"
   - The app will automatically restart with new settings

### Step 4: Verify Deployment

1. **Check App Status**:
   - Your app URL will be: `https://YOUR_APP_NAME.streamlit.app`
   - Wait for the "Running" status (green indicator)

2. **Test the Application**:
   - Upload a test PDF
   - Process it
   - Ask a question to verify everything works

## Important Notes for Streamlit Cloud

### File Storage Limitations

⚠️ **Important**: Streamlit Cloud has limitations:
- **No persistent storage**: Files uploaded during a session are temporary
- **Vector store**: ChromaDB data will be reset when the app restarts
- **Uploaded PDFs**: Will be lost when the session ends

### Solutions for Production

1. **Use External Storage**:
   - Store PDFs in cloud storage (S3, Google Cloud Storage, etc.)
   - Use a managed vector database (Pinecone, Weaviate, etc.)

2. **Database Persistence**:
   - Connect to an external database for vector storage
   - Use Streamlit's secrets to store database credentials

### Recommended Production Setup

For a production deployment, consider:

1. **Vector Database**: Use Pinecone, Weaviate, or Qdrant (cloud-hosted)
2. **File Storage**: Use AWS S3, Google Cloud Storage, or Azure Blob Storage
3. **LLM**: Use OpenAI API or other cloud LLM services

## Alternative: Deploy to Other Platforms

### Heroku

1. Create a `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Add `runtime.txt`:
   ```
   python-3.11.0
   ```

3. Deploy via Heroku CLI or GitHub integration

### Docker Deployment

1. Create `Dockerfile`:
   ```dockerfile
   FROM python:3.11-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   EXPOSE 8501

   HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

   ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. Build and run:
   ```bash
   docker build -t smartbookqa .
   docker run -p 8501:8501 smartbookqa
   ```

### Railway

1. Connect your GitHub repository
2. Railway auto-detects Python apps
3. Set environment variables in Railway dashboard
4. Deploy automatically

## Troubleshooting

### Common Issues

1. **Import Errors**:
   - Ensure all dependencies are in `requirements.txt`
   - Check that package versions are compatible

2. **Memory Issues**:
   - Streamlit Cloud free tier has memory limits
   - Consider using lighter models or optimizing code

3. **Timeout Errors**:
   - Large PDFs may take time to process
   - Add progress indicators and optimize chunking

4. **Environment Variables Not Working**:
   - Check Streamlit Cloud secrets are set correctly
   - Restart the app after adding secrets

### Getting Help

- Check Streamlit Cloud logs in the app dashboard
- Review [Streamlit Cloud documentation](https://docs.streamlit.io/streamlit-community-cloud)
- Check GitHub issues for common problems

## Quick Deployment Checklist

- [ ] Code pushed to GitHub
- [ ] Repository is public (or you have Streamlit Cloud Pro)
- [ ] `requirements.txt` is up to date
- [ ] `.env.example` is in repository (not `.env` with secrets)
- [ ] Environment variables set in Streamlit Cloud
- [ ] App deployed and running
- [ ] Tested with sample PDF

---

**Your app will be live at**: `https://YOUR_APP_NAME.streamlit.app`

Enjoy your deployed SmartBookQA application! 🎉

