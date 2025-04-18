import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

import os
import tempfile

st.set_page_config(page_title="LocalDoc Q&A", layout="wide")
st.title("📄💬 Chat with your PDF (LLM + Ollama)")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # Save the uploaded file to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.split_documents(pages)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create vectorstore
    vectordb = FAISS.from_documents(docs, embeddings)

    # Initialize Ollama LLM
    llm = Ollama(model="llama3")

    # Setup RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever()
    )

    # Question input
    st.subheader("Ask a question from your PDF:")
    user_question = st.text_input("💬 Your question:")

    if user_question:
        with st.spinner("Thinking..."):
            result = qa_chain.run(user_question)
        st.success(result)
