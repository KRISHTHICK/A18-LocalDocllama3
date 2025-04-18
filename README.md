# A18-LocalDocllama3
GenAI

Nice! Here’s a **simple and fun project idea** using **LLMs with Ollama**, **no API**, and **Streamlit**:

---

### 🔹 Project Title: **"LocalDoc Q&A: Chat with Your PDFs using LLM (Ollama + Streamlit)"**

---

### 💡 What it does:
You upload a **PDF**, and the app uses a **local LLM (via Ollama)** to answer your questions based on the content of that PDF — **no internet, no API key needed**.

---

### ⚙️ Tech Stack:
- **Ollama** (e.g., llama3, mistral, or any supported model)
- **Streamlit** (for UI)
- **LangChain** or `llama-index` (to handle PDF + retrieval)
- **PyMuPDF or pdfplumber** (for reading PDFs)

---

### 🧠 Core Idea:
- Load a local model using Ollama (e.g., `llama3`)
- Extract text from uploaded PDF
- Use basic chunking + embedding + local RAG to let the LLM answer user questions about the doc

---

### 📦 Folder Structure:
```
localdoc_qa/
│
├── app.py                 # Streamlit app
├── ollama_runner.py       # Interact with LLM via subprocess or langchain
├── document_loader.py     # PDF loader
└── requirements.txt
```

---

### 🚀 Streamlit App Flow:
1. Upload PDF
2. Show extracted content briefly (optional)
3. User types a question
4. App passes relevant context + question to Ollama model
5. Display LLM’s response

---

### ✨ Why this is good:
- No API or internet dependency
- Simple LLM + RAG pipeline
- Easy to extend later (multi-PDF, CSV, chatbot, etc.)

---

### 🔄 Variants you can build:
- Chatbot for **your notes or books**
- Q&A over **job descriptions/resumes**
- Summarizer for uploaded documents
- FAQ generator from PDF content

---

Awesome! Here's the **complete working project** for a **PDF Q&A Chatbot using Ollama and Streamlit** — no external APIs required!

---

## ✅ Step-by-Step Setup

### 🔧 1. **Install Dependencies**

Create a folder, say `localdoc_qa`, and inside that, create a `requirements.txt` file:

```txt
streamlit
langchain
pypdf
faiss-cpu
tqdm
sentence-transformers
```

Then install everything:

```bash
pip install -r requirements.txt
```

Also, make sure you have **Ollama installed**:  
👉 [https://ollama.com/download](https://ollama.com/download)

And pull a model (example using LLaMA 3):

```bash
ollama pull llama3
```

---

### 📄 2. **File: `app.py` (Main Streamlit App)**

```python
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
```

---

### ✅ 3. **Run the App**

In the project folder, run:

```bash
streamlit run app.py
```

---

## 🧠 Example Use Cases
- Upload a research paper → ask it to explain sections
- Upload a resume → ask for candidate’s strengths
- Upload a company brochure → ask for product details

---

## 💡 Bonus Tips
- To make it multi-PDF, you can expand the loader.
- You can swap FAISS with ChromaDB for persistence.
- Change LLM (`llama3`) to another local model like `mistral`, `gemma`, etc.

---

Perfect! Let's **extend the app** in two ways:

---

## 🌟 Extension 1: **Chat Interface** (Memory-style Q&A)

You'll use `ChatMessageHistory` to show previous Q&A, so it feels like an ongoing conversation.

### ✏️ Updated `app.py` with Chat UI:

```python
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

import os
import tempfile

st.set_page_config(page_title="📚🧠 Chat with your PDFs", layout="wide")
st.title("📄💬 PDF Chatbot using Local LLM (Ollama)")

uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_docs = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            loader = PyPDFLoader(tmp.name)
            all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(docs, embeddings)

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = Ollama(model="llama3")

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=False
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.subheader("💬 Chat with your document(s):")
    user_query = st.chat_input("Ask a question")

    if user_query:
        with st.spinner("Thinking..."):
            response = qa_chain.run(user_query)
            st.session_state.chat_history.append((user_query, response))

    # Display chat history
    for user_msg, bot_msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            st.markdown(bot_msg)
```

---

## 🔄 Features in this Update:
✅ Supports **multiple PDFs**  
✅ Remembers chat history  
✅ Uses **Streamlit's `chat_input` and chat messages**  
✅ Leverages **ConversationalRetrievalChain** for context-aware responses

---

## 🚀 To Run:
Same as before:

```bash
streamlit run app.py
```

