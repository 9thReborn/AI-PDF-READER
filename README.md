# AI PDF Reader 📄🤖

Upload PDF documents and ask questions about them using AI-powered Q&A. Built with Streamlit, Groq API, and local embeddings.

## Features

- 📄 **PDF Upload**: Support for PDF document uploads
- 🤖 **AI Q&A**: Ask questions about your documents using Groq's LLM
- 🔍 **Local Embeddings**: Fast, private vector search using FAISS
- 🎯 **Pre-built Questions**: Quick question templates for common queries
- 🔒 **Secure**: API keys stored as environment variables

## Tech Stack

- **Frontend**: Streamlit
- **AI**: Groq API (Llama models)
- **Embeddings**: Sentence Transformers + FAISS
- **PDF Processing**: PyPDF + LangChain

## Local Development

### Prerequisites
- Python 3.8+
- Groq API key (get free at [console.groq.com](https://console.groq.com))

### Setup
```bash
# Clone the repository
git clone https://github.com/9thReborn/AI-PDF-READER.git
cd AI-PDF-READER

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Run the app
streamlit run AisocRawAndStupid.py
```

## 🚀 Deployment on Streamlit Cloud

### Step 1: Fork/Clone Repository
Ensure your code is on GitHub.

### Step 2: Go to Streamlit Cloud
1. Visit [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account

### Step 3: Deploy
1. Click **"New app"**
2. Select your repository: `9thReborn/AI-PDF-READER`
3. Set main file path: `AisocRawAndStupid.py`
4. Click **"Deploy"**

### Step 4: Add Secrets
1. In your Streamlit Cloud dashboard, go to your app
2. Click **"⋮"** → **"Settings"**
3. Go to **"Secrets"** section
4. Add your secret:
   ```
   GROQ_API_KEY = "your_actual_api_key_here"
   ```

### Step 5: Access Your App
Your app will be live at: `https://your-app-name.streamlit.app`

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Your Groq API key | Yes |

## Usage

1. Upload a PDF document
2. Wait for processing (embeddings creation)
3. Ask questions about the document
4. Use pre-built question templates or ask custom questions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally
5. Submit a pull request

## License

MIT License - feel free to use this project for your own purposes.
