import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import tempfile
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Vercel-specific configuration
if os.getenv('VERCEL_ENV'):
    # Disable some features for Vercel
    st.set_page_config(page_title="PDF Q&A", page_icon="📄", layout="wide")
    # Reduce model size for faster loading
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Keep lightweight model
else:
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

def initialize_groq(api_key):
    return Groq(api_key=api_key)

def get_groq_response(client, context, question, model_name="llama-3.1-8b-instant"):
    prompt = f"""
        Based on the following context, answer the question concisely and accurately. Only use information present in the context.
        Context: {context}
        Question: {question}
        Answer:
        """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role":"user", "content":prompt}
            ],
            temperature=0.0,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error:{str(e)}"

class LocalVectorStore:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.chunks = []
        self.embeddings = None
        self.index = None

    def add_documents(self, documents):
        self.chunks = [doc.page_content for doc in documents]

        embeddings = self.embedding_model.encode(self.chunks)
        self.embeddings = np.array(embeddings).astype('float32')

        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)

    def similarity_search(self, query, k=4):
        if self.index is None:
            return []

        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')

        distances, indices = self.index.search(query_embedding, k)
        results = []

        for i in indices[0]:
            if i < len(self.chunks):
                results.append(self.chunks[i])

        return results

def load_and_split_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        split_docs = text_splitter.split_documents(documents)

        os.remove(temp_file_path)
        return split_docs
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise e

def process_document(uploaded_file, groq_client, embedding_model):
    st.write("Processing document...")

    with st.spinner("Reading PDF..."):
        chunks = load_and_split_pdf(uploaded_file)

    if not chunks:
        st.error("Failed to read or split the PDF document")
        return

    with st.spinner("Creating embeddings..."):
        vector_store = LocalVectorStore(embedding_model)
        vector_store.add_documents(chunks)

    st.success("Document processed successfully!", icon="✅")
    st.balloons()

    st.session_state.vector_store = vector_store
    st.session_state.groq_client = groq_client
    st.session_state.ready = True

def main():
    st.title("📄 PDF Q&A Assistant")
    st.caption("Upload a PDF and ask questions about it")

    # Get API key from environment variable
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        st.error("⚠️ Groq API key not found!")
        st.info("Please set the GROQ_API_KEY environment variable")
        return

    # Initialize clients
    groq_client = initialize_groq(groq_api_key)
    embedding_model = load_embedding_model()

    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None and not st.session_state.get("ready", False):
        process_document(uploaded_file, groq_client, embedding_model)

    if st.session_state.get("ready", False):
        st.header("Ask your questions")

        # Predefined questions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("What is this document about?"):
                st.session_state.question = "What is this document about?"
            if st.button("Who are the main authors or people mentioned?"):
                st.session_state.question = "Who are the main authors or people mentioned?"
        with col2:
            if st.button("What are the key findings or conclusions?"):
                st.session_state.question = "What are the key findings or conclusions?"
            if st.button("Can you summarise the main points?"):
                st.session_state.question = "Can you summarise the main points?"

        # Custom question input
        question = st.text_input("Or ask your own question:", value=st.session_state.get("question", ""))

        if question:
            try:
                with st.spinner("Searching for relevant context..."):
                    relevant_chunks = st.session_state.vector_store.similarity_search(question, k=4)
                    if not relevant_chunks:
                        st.warning("No relevant context found for your question.")
                        return

                    context = "\n\n".join(relevant_chunks)

                with st.spinner("Getting answer..."):
                    answer = get_groq_response(
                        st.session_state.groq_client,
                        context,
                        question,
                        st.session_state.get("model_name", "llama-3.1-8b-instant")
                    )

                st.write("**Answer:**")
                st.write(answer)

                st.success("Answer generated successfully.")

                with st.expander("View source chunks"):
                    for i, chunk in enumerate(relevant_chunks):
                        display_chunk = chunk[:400] + "..." if len(chunk) > 400 else chunk
                        st.write(f"**Chunk {i+1}:**")
                        st.write(display_chunk)
                        st.write("---")

            except Exception as e:
                if "rate limit" in str(e).lower():
                    st.error("Rate limit reached. Please wait a moment and try again.")
                    st.info("Free tier limits are generous but not unlimited.")
                else:
                    st.error(f"Error: {str(e)}")
                    st.info("Try simplifying your question or check your API key.")

if __name__ == "__main__":
    main()