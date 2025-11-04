import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
import tempfile
import os

# ---------------------------
# Streamlit App Configuration
# ---------------------------
st.set_page_config(page_title="📚 Chat with Your Documents", layout="wide")
st.title("📚 Chat with Your Documents (Chroma + Groq)")

with st.sidebar:
    st.header("⚙️ Settings")
    groq_api_key = st.text_input("Enter your Groq API Key", type="password")
    model_name = st.selectbox("Select Model", [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "deepseek-r1-distill-llama-70b"
    ])
    st.markdown("---")
    uploaded_files = st.file_uploader("📁 Upload Documents", type=["pdf", "txt", "docx"], accept_multiple_files=True)
    clear_chat = st.button("🗑️ Clear Chat History")

# ---------------------------
# Initialize session state
# ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if clear_chat:
    st.session_state.chat_history = []
    st.experimental_rerun()

# ---------------------------
# Helper functions
# ---------------------------
def load_document(file):
    """Load a single document"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp_file:
        tmp_file.write(file.read())
        path = tmp_file.name

    if file.name.endswith(".pdf"):
        loader = PyPDFLoader(path)
    elif file.name.endswith(".txt"):
        loader = TextLoader(path)
    elif file.name.endswith(".docx"):
        loader = Docx2txtLoader(path)
    else:
        raise ValueError("Unsupported file type.")
    return loader.load()

def process_documents(files):
    """Combine, split and embed documents"""
    docs = []
    for file in files:
        docs.extend(load_document(file))

    st.info(f"✅ Loaded {len(files)} document(s). Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    st.info("🔍 Creating embeddings (may take a minute)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="chroma_db")
    st.success("✅ Embeddings created successfully!")

    return vectordb

# ---------------------------
# Build and Run Chat Chain
# ---------------------------
if uploaded_files and groq_api_key:
    vectordb = process_documents(uploaded_files)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

    st.success("✅ You can now start chatting with your documents!")

    user_query = st.chat_input("💬 Ask something about your documents...")

    if user_query:
        with st.spinner("🤔 Thinking..."):
            result = qa_chain.invoke({"question": user_query, "chat_history": st.session_state.chat_history})
            response = result["answer"]

            st.session_state.chat_history.append(("You", user_query))
            st.session_state.chat_history.append(("AI", response))

    # Display chat history
    for role, text in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"**🧑‍💻 {role}:** {text}")
        else:
            st.markdown(f"**🤖 {role}:** {text}")

else:
    st.info("👈 Upload documents and enter your Groq API key to begin.")
