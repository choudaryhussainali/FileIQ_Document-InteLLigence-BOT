import streamlit as st

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_groq import ChatGroq
    from langchain.chains import ConversationalRetrievalChain
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
except ImportError as e:
    st.error(f"""Error loading required packages: {str(e)}
    Please install the correct package versions using:
    pip install -r requirements.txt""")
    st.stop()

import tempfile
import os
import time
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

os.environ["PYTHONWARNINGS"] = "ignore"

# Page configuration
st.set_page_config(
    page_title="Chat with Your Documents 📚",
    page_icon="📚",
    layout="wide",
)

# Sidebar with app information
with st.sidebar:
    st.title("Document Intelligence-Bot 🤖")
    st.markdown("""
    Upload documents and chat with them using Llama model via Groq!
    
    **Supported file formats:**
    - PDF (.pdf)
    - Text (.txt)
    - Word (.docx)
    
    **How it works:**
    1. Enter your Groq API key
    2. Upload one or more documents
    3. Wait for processing to complete
    4. Start asking questions about your documents!
    """)
    
    # API key input
    groq_api_key = st.text_input("Enter Groq API Key:", type="password")
    
    # Model selection
    model_name = st.selectbox(
        "Select Llama Model:",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "deepseek-r1-distill-llama-70b"],
        index=0
    )
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.conversation = None
        st.session_state.document_processed = False
        st.experimental_rerun()

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

def process_documents(uploaded_files):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)
        
        documents = []
        for file_path in file_paths:
            try:
                if file_path.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                elif file_path.endswith(".txt"):
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
                elif file_path.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                    documents.extend(loader.load())
            except Exception as e:
                st.error(f"Error loading {os.path.basename(file_path)}: {str(e)}")
        
        if not documents:
            st.error("No documents were successfully processed. Please check the file formats.")
            return None
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store

def init_conversation_chain(vector_store, groq_api_key, model_name):
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name=model_name
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )
    
    return conversation_chain

uploaded_files = st.file_uploader(
    "Upload your documents here",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)

if uploaded_files and not st.session_state.document_processed:
    if not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar.")
    else:
        with st.spinner("Processing documents... This may take a minute."):
            vector_store = process_documents(uploaded_files)
            
            if vector_store:
                st.session_state.conversation = init_conversation_chain(
                    vector_store,
                    groq_api_key,
                    model_name
                )
                st.session_state.document_processed = True
                
                st.success(f"✅ {len(uploaded_files)} documents processed successfully! You can now start chatting.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.document_processed:
    user_question = st.chat_input("Ask something about your documents...")
    
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        with st.chat_message("user"):
            st.write(user_question)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            chat_history = [(m["content"], "") for m in st.session_state.messages[:-1] if m["role"] == "user"]
            
            try:
                with st.spinner("Thinking..."):
                    response = st.session_state.conversation(
                        {"question": user_question, "chat_history": chat_history}
                    )
                    
                    answer = response["answer"]
                    
                    for chunk in answer.split():
                        full_response += chunk + " "
                        time.sleep(0.02)
                        message_placeholder.write(full_response + "▌")
                    
                    message_placeholder.write(full_response)
                    
                    if response.get("source_documents"):
                        with st.expander("View Source Documents"):
                            for i, doc in enumerate(response["source_documents"]):
                                st.markdown(f"**Source {i+1}**")
                                st.markdown(f"*Content:* {doc.page_content[:300]}...")
                                st.markdown(f"*Source:* {doc.metadata.get('source', 'Unknown')}")
                                st.markdown("---")
            
            except Exception as e:
                full_response = f"Error: {str(e)}"
                message_placeholder.write(full_response)
                st.error("An error occurred while generating the response. Please check your API key and try again.")
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    if not uploaded_files:
        st.info("👆 Please upload one or more documents to get started!")