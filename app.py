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

# Sidebar
with st.sidebar:
    st.title("🤖 Document Intelligence Bot")
    
    st.markdown("---")
    
    # Instructions
    with st.expander("📖 How to Use", expanded=True):
        st.markdown("""
        **Steps:**
        1. Select your AI model
        2. Enter API key
        3. Upload documents (PDF, DOCX, TXT)
        4. Click Process Documents
        5. Ask questions!
        
        **Supported Formats:**
        - 📄 PDF (.pdf)
        - 📝 Word (.docx)
        - 📃 Text (.txt)
        """)
    
    st.markdown("---")
    
    # Model selection
    st.subheader("🧠 AI Model Selection")
    
    model_option = st.selectbox(
        "Choose Model:",
        [
            "Llama-3.2-3B (Groq) - Fast ⚡",
            "Gemini 1.5 Flash - Free 🆓",
            "Mixtral-8x7B (Groq) - Powerful 💪",
            "Llama-3.1-8B (Groq) - Balanced ⚖️"
        ],
        index=0
    )
    
    # API Key input
    api_key = None
    if "Groq" in model_option:
        api_key = st.text_input(
            "Groq API Key:",
            type="password",
            help="Get free key at console.groq.com"
        )
        if not api_key:
            st.info("👆 Enter your Groq API key to continue")
    elif "Gemini" in model_option:
        api_key = st.text_input(
            "Gemini API Key:",
            type="password",
            help="Get free key at ai.google.dev"
        )
        if not api_key:
            st.info("👆 Enter your Gemini API key to continue")
    
    st.markdown("---")
    
    # Statistics
    if st.session_state.vectorstore:
        st.subheader("📊 Statistics")
        st.metric("Documents Loaded", len(st.session_state.processed_files))
    
    st.markdown("---")
    
    # Action buttons
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
    
    # Export chat
    if st.session_state.messages:
        st.markdown("---")
        chat_text = "\n\n".join([
            f"{'USER' if msg['role'] == 'user' else 'ASSISTANT'}: {msg['content']}" 
            for msg in st.session_state.messages
        ])
        st.download_button(
            "💾 Export Chat",
            chat_text,
            file_name="chat_history.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    st.markdown("---")
    st.caption("Built with Streamlit & LangChain")

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


