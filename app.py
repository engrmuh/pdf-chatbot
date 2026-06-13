import subprocess
import sys

# 1. AUTOMATIC SETUP: Ensure all required packages are installed with correct names
required_packages = {
    "streamlit": "streamlit",
    "langchain_community": "langchain-community",
    "langchain_text_splitters": "langchain-text-splitters",
    "langchain_ollama": "langchain-ollama",
    "langchain_core": "langchain-core",
    "langchain_classic": "langchain-classic",
    "chromadb": "chromadb",
    "pypdf": "pypdf"
}

for module_name, package_name in required_packages.items():
    try:
        __import__(module_name)
    except ImportError:
        print(f"Installing missing dependency: {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

# 2. NOW IMPORT THE PACKAGES
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import os
import tempfile

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# 3. STREAMLIT PAGE CONFIGURATION
st.set_page_config(page_title="Offline PDF Chatbot", page_icon="📄", layout="centered")
st.title("📄 Chat with your PDF (100% Local)")
st.write("Upload a PDF document and ask questions locally without your data leaving your computer.")

# 4. SIDEBAR CONFIGURATION
with st.sidebar:
    st.header("Local Settings")
    st.info("Ollama is handling everything locally!", icon="🤖")
    
    # Set gemma2:2b as the default option
    llm_model = st.selectbox(
        "Select Local LLM:",
        ["gemma2:2b"],
        index=0
    )
# 5. FILE UPLOADER WIDGET
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded bytes to a temporary file path for the PDF loader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Process the file
    with st.spinner("Processing PDF and generating local embeddings... This may take a moment."):
        try:
            # Load the PDF
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            
            # Split the document into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
            splits = text_splitter.split_documents(docs)
            
            # Create local embeddings using Ollama (nomic-embed-text is highly efficient)
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            st.success("PDF processed successfully! Ask away.", icon="✅")
        except Exception as e:
            st.error(f"Error processing PDF. Ensure Ollama is running and models are downloaded. Error: {e}")
            st.stop()
        finally:
            # Clean up the temporary file from disk securely
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    # 6. CHAT INTERFACE
    # Initialize chat history in Streamlit session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if user_question := st.chat_input("What would you like to know about this document?"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        # 7. BUILD LOCAL RAG CHAIN & GENERATE RESPONSE
        llm = ChatOllama(model=llm_model, temperature=0)

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Keep your answer brief and concise.\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # Assemble the LangChain workflow
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Generate response locally
        with st.chat_message("assistant"):
            with st.spinner("Thinking locally..."):
                try:
                    response = rag_chain.invoke({"input": user_question})
                    answer = response["answer"]
                    st.markdown(answer)
                    
                    sources = response.get("context", [])
                    if sources:
                        with st.expander("📄 Sources"):
                            for doc in sources:
                                page = doc.metadata.get("page", 0)
                                snippet = doc.page_content[:200]
                                st.markdown(f"**Page {page + 1}**")
                                st.caption(snippet)
                                st.divider()

                    # Add assistant response to history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Failed to generate response. Is Ollama running? Error: {e}")