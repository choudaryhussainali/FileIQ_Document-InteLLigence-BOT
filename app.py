import streamlit as st
import os
import tempfile
from typing import List

# LangChain imports - SIMPLIFIED VERSION THAT WORKS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Page configuration
st.set_page_config(
    page_title="Chat with Your Documents 📚",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

import streamlit as st

# --- Sidebar Styling ---
import streamlit as st

# ---- ULTRA MODERN SIDEBAR STYLING ----
st.markdown("""
<style>
/* Sidebar Background */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f172a 0%, #1e293b 45%, #020617 100%);
    color: #f1f5f9 !important;
    padding: 1.4rem 1rem;
    border-right: 1px solid rgba(255,255,255,0.05);
    box-shadow: 4px 0 20px rgba(0,0,0,0.3);
}

/* Title */
.sidebar-title {
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    letter-spacing: 1px;
}

/* Glowing Divider */
.divider {
    height: 1px;
    margin: 1rem 0;
    background: linear-gradient(90deg, rgba(56,189,248,0) 0%, rgba(56,189,248,0.7) 50%, rgba(56,189,248,0) 100%);
}

/* Section Box */
.glass-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
}
.glass-box:hover {
    background: rgba(255,255,255,0.07);
}

/* Dropdown & Inputs */
.stSelectbox, .stTextInput {
    border-radius: 8px !important;
}

/* Buttons */
.stButton button {
    border-radius: 10px;
    background: linear-gradient(90deg, #2563eb, #9333ea);
    color: #fff;
    border: none;
    box-shadow: 0 0 12px rgba(147,51,234,0.4);
    transition: all 0.3s ease-in-out;
}
.stButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 0 18px rgba(147,51,234,0.7);
    background: linear-gradient(90deg, #1d4ed8, #7e22ce);
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-weight: 700 !important;
}

/* Caption */
.stCaption {
    color: #94a3b8 !important;
    text-align: center;
    font-size: 0.8rem !important;
    margin-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ---- SIDEBAR CONTENT ----
with st.sidebar:
    st.markdown('<div class="sidebar-title">🤖 Document Intelligence Bot</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # --- How to Use ---
    with st.expander("📘 How to Use", expanded=True):
        st.markdown("""
        **Quick Start Guide**
        1️⃣ Select an AI Model  
        2️⃣ Enter your API Key  
        3️⃣ Upload PDF / DOCX / TXT  
        4️⃣ Click **Process Documents**  
        5️⃣ Ask your questions! 💬  

        **Supported Formats**
        - 📄 PDF  
        - 📝 Word  
        - 📃 Text
        """)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # --- Model Selection ---
    st.markdown('<div class="glass-box">', unsafe_allow_html=True)
    st.markdown("### 🧠 Model Selection")
    model_option = st.selectbox(
        "Choose AI Model:",
        [
            "⚡ Llama-3.2-3B (Groq) — Fast",
            "🆓 Gemini 1.5 Flash — Free",
            "💪 Mixtral-8x7B (Groq) — Powerful",
            "⚖️ Llama-3.1-8B (Groq) — Balanced"
        ],
        index=0
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # --- API Key ---
    api_key = None
    st.markdown('<div class="glass-box">', unsafe_allow_html=True)
    if "Groq" in model_option:
        api_key = st.text_input("🔑 API Key", type="password", help="Get your key at console.groq.com")
        if not api_key:
            st.info("👆 Enter your API key to continue")
    elif "Gemini" in model_option:
        api_key = st.text_input("🔑 Gemini API Key", type="password", help="Get your key at ai.google.dev")
        if not api_key:
            st.info("👆 Enter your Gemini API key to continue")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Stats ---
    if st.session_state.get("vectorstore"):
        st.markdown('<div class="glass-box">', unsafe_allow_html=True)
        st.markdown("### 📊 Document Stats")
        st.metric("Documents Loaded", len(st.session_state.get("processed_files", [])))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Actions ---
    st.markdown('<div class="glass-box">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("📄 Reset Docs", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.retriever = None
            st.session_state.llm = None
            st.session_state.processed_files = []
            st.session_state.messages = []
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Export Chat ---
    if st.session_state.get("messages"):
        st.markdown('<div class="glass-box">', unsafe_allow_html=True)
        chat_text = "\n\n".join([
            f"{'USER' if msg['role'] == 'user' else 'ASSISTANT'}: {msg['content']}"
            for msg in st.session_state.messages
        ])
        st.download_button(
            "💾 Export Chat History",
            chat_text,
            file_name="chat_history.txt",
            mime="text/plain",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.caption("Designed with ❤️ by CH Hussain Ali")



# Main content
st.title("📚 Chat with Your Documents")
st.markdown("Upload your documents and ask questions using AI")

# Function to load documents
def load_documents(uploaded_files):
    """Load documents from uploaded files"""
    documents = []
    temp_dir = tempfile.mkdtemp()
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif uploaded_file.name.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            elif uploaded_file.name.endswith('.txt'):
                loader = TextLoader(file_path)
            else:
                continue
            
            docs = loader.load()
            for doc in docs:
                doc.metadata['source'] = uploaded_file.name
            documents.extend(docs)
        
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {str(e)}")
    
    return documents

# Function to initialize LLM
def initialize_llm(model_option, api_key):
    """Initialize the selected LLM"""
    try:
        if "Llama-3.2-3B" in model_option:
            return ChatGroq(model="llama-3.1-8b-instant", temperature=0.7, groq_api_key=api_key)
        elif "Gemini" in model_option:
            return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.7)
        elif "Mixtral-8x7B" in model_option:
            return ChatGroq(model="mixtral-8x7b-32768", temperature=0.7, groq_api_key=api_key)
        elif "Llama-3.1-8B" in model_option:
            return ChatGroq(model="llama-3.1-8b-instant", temperature=0.7, groq_api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None

# Function to format documents
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# File upload section
st.subheader("📤 Upload Your Documents")

uploaded_files = st.file_uploader(
    "Choose files (PDF, DOCX, TXT)",
    type=['pdf', 'docx', 'txt'],
    accept_multiple_files=True,
    help="Upload one or more documents"
)

# Process documents button
if uploaded_files and api_key:
    if st.button("🚀 Process Documents", type="primary", use_container_width=True):
        with st.spinner("Processing your documents..."):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Load documents
                status_text.text("📂 Loading documents...")
                progress_bar.progress(20)
                documents = load_documents(uploaded_files)
                
                if not documents:
                    st.error("No documents loaded. Check your files.")
                    st.stop()
                
                # Split documents
                status_text.text("✂️ Splitting text...")
                progress_bar.progress(40)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = text_splitter.split_documents(documents)
                
                # Create embeddings
                status_text.text("🧮 Creating embeddings...")
                progress_bar.progress(60)
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
                
                # Create vector store
                status_text.text("💾 Building database...")
                progress_bar.progress(80)
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings
                )
                
                # Initialize LLM
                status_text.text("🤖 Initializing AI...")
                llm = initialize_llm(model_option, api_key)
                
                if not llm:
                    st.error("Failed to initialize AI model.")
                    st.stop()
                
                # Save to session
                st.session_state.vectorstore = vectorstore
                st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                st.session_state.llm = llm
                st.session_state.processed_files = [f.name for f in uploaded_files]
                
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                
                st.success(f"✅ Processed {len(uploaded_files)} document(s) into {len(chunks)} chunks!")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

elif uploaded_files and not api_key:
    st.warning("⚠️ Please enter your API key in the sidebar.")

# Chat interface
if st.session_state.retriever and st.session_state.llm:
    st.markdown("---")
    st.subheader("💬 Chat with Your Documents")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    for idx, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {idx}: {source['filename']}**")
                        st.text(source["content"][:300] + "...")
                        st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get relevant documents
                    retrieved_docs = st.session_state.retriever.invoke(prompt)
                    
                    # Create prompt
                    template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer concisely based on the context provided."""
                    
                    prompt_template = ChatPromptTemplate.from_template(template)
                    
                    # Create chain using LCEL
                    chain = (
                        {"context": lambda x: format_docs(retrieved_docs), "question": RunnablePassthrough()}
                        | prompt_template
                        | st.session_state.llm
                        | StrOutputParser()
                    )
                    
                    # Get answer
                    answer = chain.invoke(prompt)
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Prepare sources
                    source_info = []
                    for doc in retrieved_docs:
                        source_info.append({
                            "filename": doc.metadata.get("source", "Unknown"),
                            "content": doc.page_content
                        })
                    
                    # Show sources
                    if source_info:
                        with st.expander("📚 View Sources"):
                            for idx, source in enumerate(source_info, 1):
                                st.markdown(f"**Source {idx}: {source['filename']}**")
                                st.text(source["content"][:300] + "...")
                                st.markdown("---")
                    
                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": source_info
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)

else:
    st.info("👆 Upload documents and click 'Process Documents' to start!")
    
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - What is the main topic?
        - Summarize the key points
        - What dates are mentioned?
        - List all names in the documents
        """)





