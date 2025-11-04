import streamlit as st
import os
import tempfile
from pathlib import Path
from typing import List
import time

# LangChain imports - FULLY CORRECTED for 2024+
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Page configuration
st.set_page_config(
    page_title="Chat with Your Documents 📚",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .upload-text {
        font-size: 14px;
        color: #666;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
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
        2. Enter API key (if required)
        3. Upload documents (PDF, DOCX, TXT)
        4. Wait for processing
        5. Ask questions!
        
        **Supported Formats:**
        - 📄 PDF (.pdf)
        - 📝 Word (.docx)
        - 📃 Text (.txt)
        
        **Features:**
        - Multi-document chat
        - Source citations
        - Conversation history
        - Multiple AI models
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
    
    # API Key input based on model
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
        try:
            chunk_count = st.session_state.vectorstore._collection.count()
            st.metric("Text Chunks", chunk_count)
        except:
            pass
    
    st.markdown("---")
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("📄 Reset Docs", use_container_width=True):
            st.session_state.qa_chain = None
            st.session_state.vectorstore = None
            st.session_state.processed_files = []
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
    
    # Export chat
    if st.session_state.messages:
        st.markdown("---")
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
    
    st.markdown("---")
    st.caption("Built with Streamlit & LangChain")

# Main content
st.title("📚 Chat with Your Documents")
st.markdown("Upload your documents and ask questions using AI - **100% Free & Open Source**")

# Function to load documents
def load_documents(uploaded_files):
    """Load documents from uploaded files"""
    documents = []
    temp_dir = tempfile.mkdtemp()
    
    for uploaded_file in uploaded_files:
        # Save file temporarily
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load based on file type
        try:
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif uploaded_file.name.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            elif uploaded_file.name.endswith('.txt'):
                loader = TextLoader(file_path)
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}")
                continue
            
            docs = loader.load()
            # Add filename to metadata
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
            return ChatGroq(
                model="llama-3.2-3b-preview",
                temperature=0.7,
                groq_api_key=api_key
            )
        elif "Gemini" in model_option:
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=api_key,
                temperature=0.7
            )
        elif "Mixtral-8x7B" in model_option:
            return ChatGroq(
                model="mixtral-8x7b-32768",
                temperature=0.7,
                groq_api_key=api_key
            )
        elif "Llama-3.1-8B" in model_option:
            return ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.7,
                groq_api_key=api_key
            )
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None

# Function to create conversational chain (Modern approach)
def create_conversational_chain(llm, retriever):
    """Create a modern conversational retrieval chain"""
    
    # Contextualize question prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # Answer question prompt
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Create question-answer chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Create retrieval chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# File upload section
st.subheader("📤 Upload Your Documents")

uploaded_files = st.file_uploader(
    "Choose files (PDF, DOCX, TXT)",
    type=['pdf', 'docx', 'txt'],
    accept_multiple_files=True,
    help="Upload one or more documents to analyze"
)

# Process documents button
if uploaded_files and api_key:
    if st.button("🚀 Process Documents", type="primary", use_container_width=True):
        with st.spinner("Processing your documents..."):
            try:
                # Load documents
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("📂 Loading documents...")
                progress_bar.progress(20)
                documents = load_documents(uploaded_files)
                
                if not documents:
                    st.error("No documents were successfully loaded. Please check your files.")
                    st.stop()
                
                # Split documents
                status_text.text("✂️ Splitting text into chunks...")
                progress_bar.progress(40)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", " ", ""]
                )
                chunks = text_splitter.split_documents(documents)
                
                # Create embeddings
                status_text.text("🧮 Creating embeddings...")
                progress_bar.progress(60)
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                
                # Create vector store
                status_text.text("💾 Building vector database...")
                progress_bar.progress(80)
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory="./chroma_db"
                )
                
                # Initialize LLM
                status_text.text("🤖 Initializing AI model...")
                llm = initialize_llm(model_option, api_key)
                
                if not llm:
                    st.error("Failed to initialize the AI model. Please check your API key.")
                    st.stop()
                
                # Create conversational chain
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                qa_chain = create_conversational_chain(llm, retriever)
                
                # Save to session state
                st.session_state.qa_chain = qa_chain
                st.session_state.vectorstore = vectorstore
                st.session_state.processed_files = [f.name for f in uploaded_files]
                
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                
                st.success(f"✅ Successfully processed {len(uploaded_files)} document(s) into {len(chunks)} chunks!")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                st.exception(e)

elif uploaded_files and not api_key:
    st.warning("⚠️ Please enter your API key in the sidebar to process documents.")

# Chat interface
if st.session_state.qa_chain:
    st.markdown("---")
    st.subheader("💬 Chat with Your Documents")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Source Documents"):
                    for idx, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {idx}: {source['filename']}**")
                        st.text(source["content"][:400] + "..." if len(source["content"]) > 400 else source["content"])
                        st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Invoke the chain with chat history
                    response = st.session_state.qa_chain.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.chat_history
                    })
                    
                    answer = response["answer"]
                    source_docs = response.get("context", [])
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Update chat history
                    st.session_state.chat_history.extend([
                        HumanMessage(content=prompt),
                        AIMessage(content=answer)
                    ])
                    
                    # Prepare source information
                    source_info = []
                    for doc in source_docs:
                        source_info.append({
                            "filename": doc.metadata.get("source", "Unknown"),
                            "content": doc.page_content
                        })
                    
                    # Show sources
                    if source_info:
                        with st.expander("📚 View Source Documents"):
                            for idx, source in enumerate(source_info, 1):
                                st.markdown(f"**Source {idx}: {source['filename']}**")
                                st.text(source["content"][:400] + "..." if len(source["content"]) > 400 else source["content"])
                                st.markdown("---")
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": source_info
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })

else:
    # Welcome message
    st.info("👆 Upload documents above and click 'Process Documents' to get started!")
    
    # Example questions
    with st.expander("💡 Example Questions You Can Ask"):
        st.markdown("""
        - What is the main topic of these documents?
        - Summarize the key points from the uploaded files
        - What are the important dates mentioned?
        - List all the names mentioned in the documents
        - Explain [specific concept] from the documents
        - Compare and contrast different sections
        - What conclusions can be drawn from this information?
        """)
