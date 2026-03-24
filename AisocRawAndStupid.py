import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.text_splitters import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import tempfile
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def initialize_groq(api_key):
    return Groq(api_key = api_key)

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
    
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

class LocalVectorStore:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.chunks = []
        self.embeddings=None
        self.index=None
        

    def add_documents(self,documents):
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
    with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as temp_file:
        # temp_file.write(uploaded_file.getvalue())
        temp_file.write(uploaded_file.read()) 
        temp_file_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,length_function=len,separators=["\n\n", "\n", ". ", " ", ""])
        split_docs = text_splitter.split_documents(documents)

        os.remove(temp_file_path) # Clean up the temporary file
        return split_docs
    except Exception as e:
        os.remove(temp_file_path)

def process_document(uploaded_file, groq_client, embedding_model):
    st.write("Processing document...")

    # Each time stremlit is loading up the chunk, it shows a spinner or message 
    with st.spinner("Reading PDF..."):
        chunks = load_and_split_pdf(uploaded_file)

    if not chunks:
        st.error("failed to read or split the PDF document")
        return

    # st.success(f"Loaded {len(chunks)} chunks from the document")

    with st.spinner("Loading..."):
        vector_store = LocalVectorStore(embedding_model)
        vector_store.add_documents(chunks)

    # st.success("Vector store created successfully")
    st.success("Document processed successfully!", icon="✅")
    st.balloons()

    st.session_state.vector_store = vector_store
    st.session_state.groq_client = groq_client
    st.session_state.ready = True

    if st.session_state.get("ready", False):
        st.header("Ask your questions")
        st.write("**Try asking:**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("What is this document about?"):
                st.session_state.question = "what is this document about?"
            if st.button("Who are the main authors or people mentioned?"):
                st.session_state.question = "Who are the main authors or people mentioned?"
        with col2:
            if st.button("What are the key findings or conclusions?"):
                st.session_state.question = "What are the key findings or conclusions?"
            if st.button("Can you summarise the main points?"):
                st.session_state.question = "Can you summarise the main points?"

        question = st.text_input("Ask a question about the document:", value=st.session_state.get("question",""))

        if question:
            try:
                with st.spinner("Searching for relevant context..."):
                    relevant_chunks = st.session_state.vector_store.similarity_search(question, k=4)
                    if not relevant_chunks:
                        st.warning("No relevant context found for your question.")
                        return

                    context = "\n\n".join(relevant_chunks)

                    answer = get_groq_response(
                        st.session_state.groq_client, context, question, st.session_state.get("model_name","llama-3.1-8b-instant")
                    )
                st.write("**Answer:**")
                st.write(answer)

                st.success("Answer generated successfully.")
                with st.expander("View source chunks"):
                    for i, chunk in enumerate(relevant_chunks):
                        #Truncate long chunks for readability
                        display_chunk = chunk[:400] + "..." if len(chunk) > 400 else chunk
                        st.write(display_chunk)
                        st.write("...")
            except Exception as e:
                # Handle different types of error gracefully
                if "rate limit" in str(e).lower():
                    st.error("Rate limit reached. Please wait a moment and try again")
                    st.info("Free tier limit are generous but not unlimited")
                else:
                    st.error("Error: {str(e)}")
                    st.info("Try simplifying your question or check API key")


with st.sidebar:
    st.title("📄 PDF Q&A")
    st.caption("Powered by Groq + local embeddings")

    st.divider()

    # Get API key from environment variable
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        st.error("⚠️ Groq API key not found!")
        st.info("Please set the GROQ_API_KEY environment variable or add it to your .env file")
        st.stop()

    model_options = {
        "llama-3.1-8b-instant": "Llama 3.1 8B  •  fast",
        "llama-3.1-70b-versatile": "Llama 3.1 70B  •  powerful",
        "gemma2-9b-it": "Gemma 2 9B  •  balanced",
    }

    selected_model = st.selectbox(
        "Model",
        options=list(model_options.keys()),
        format_func=lambda k: model_options[k],
        index=0
    )

    st.caption("Free tier → generous but rate-limited")

    st.markdown("---")
    st.markdown("**Tips**")
    st.markdown("• Clear & specific questions work best")
    st.markdown("• Large PDFs may take longer to process")

def main():
    st.set_page_config(
        page_title="Pdf Doc Reader",
        page_icon=":robot:",
        layout="wide"
    )
    st.title("📄 PDF Document Reader")
    st.markdown("Upload a PDF → ask natural language questions about its content")

    if not groq_api_key:
        st.sidebar.warning("Please enter your API key to proceed.")
        return

    groq_client = initialize_groq(groq_api_key)
    embedding_model = load_embedding_model()

    st.session_state.selected_model = selected_model

    uploaded_file = st.file_uploader("Upload a pdf document", type=["pdf"], help="Only PDF files are supported",label_visibility="collapsed")

    if uploaded_file is not None:
        process_document(uploaded_file, groq_client, embedding_model)

if __name__ == "__main__":
    main()

    