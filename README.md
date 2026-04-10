# 🧠 RAG-Based Document Question Answering System

## 📌 Project Overview

This project is a **Retrieval-Augmented Generation (RAG) system** that allows users to ask questions based on their own documents.

It works by:

* Loading documents (TXT / PDF)
* Splitting them into chunks
* Converting them into vector embeddings
* Storing them in a vector database
* Retrieving relevant chunks based on a query
* Generating answers using an LLM

👉 The system ensures that answers are generated **from your data**, not from general internet knowledge.

---

## 🚀 Features

* 📂 Supports `.txt` and `.pdf` documents
* 🧹 Automatic text cleaning
* ✂️ Smart chunking using Recursive Text Splitter
* 🔍 Semantic search using embeddings
* 🧠 Local LLM-based answer generation
* 💾 Persistent vector storage
* 🔁 Two vector database options:

  * ChromaDB
  * FAISS

---

## 🏗️ Project Architecture

```
Documents → Cleaning → Chunking → Embeddings → Vector DB
                                              ↓
                                         Retriever
                                              ↓
                                            LLM
                                              ↓
                                           Answer
```

---

## 📁 Project Structure

```
RAG/
│── docs/                  # Input documents
│── db/                    # Chroma database storage
│── faiss_index/           # FAISS index storage
│── ingestion_pipeline.py  # Main pipeline (RAG system)
│── .env                   # API keys (optional)
│── README.md              # Project documentation
```

---

## ⚙️ Installation

### 1. Create Virtual Environment

```bash
python -m venv myVirtualEnv
myVirtualEnv\Scripts\activate
```

---

### 2. Install Dependencies

```bash
pip install langchain
pip install langchain-community
pip install langchain-text-splitters
pip install transformers torch
pip install sentence-transformers
pip install faiss-cpu
pip install python-dotenv
```

---

## 📂 Add Your Documents

Place your files inside the `docs/` folder:

```
docs/
├── Tesla.txt
├── Microsoft.txt
├── Nvidia.txt
```

---

## ▶️ Run the Project

```bash
python ingestion_pipeline.py
```

---

## 💬 Example Usage

```
Please Ask your Question: What does Tesla do?

Answer:
Tesla is an American company that manufactures electric vehicles...

Sources:
- docs/Tesla.txt
```

---

# 🗄️ Vector Database Options

This project provides **two implementations**:

---

## 🔵 1. ChromaDB Version

### ✅ When to use:

* You want **persistent database with metadata**
* You plan to scale your project
* You want **production-level vector DB**

### ❗ Requirements:

* Requires **C++ Build Tools (Windows)**
* May fail without proper setup

### 🔧 Used in code:

```python
from langchain_chroma import Chroma
```

---

## 🟢 2. FAISS Version (Recommended)

### ✅ When to use:

* You want **quick setup**
* You are learning RAG
* You want **no dependency issues**
* You want local and fast performance

### ❗ Advantages:

* No C++ installation required
* Faster local execution
* Easy to use

### 🔧 Used in code:

```python
from langchain_community.vectorstores import FAISS
```

---

## ⚖️ Chroma vs FAISS

| Feature          | Chroma     | FAISS            |
| ---------------- | ---------- | ---------------- |
| Setup Difficulty | ❌ Hard     | ✅ Easy           |
| Performance      | ✅ Good     | ✅ Very Fast      |
| Persistence      | ✅ Yes      | ✅ Yes            |
| Dependencies     | ❌ Heavy    | ✅ Lightweight    |
| Best For         | Production | Learning & Local |

---

## 🧠 Embedding Model

```python
all-MiniLM-L6-v2
```

* Lightweight and fast
* Works locally
* No API key required

---

## 🤖 LLM Used

```python
GPT-2 (HuggingFace)
```

* Runs locally
* No API cost
* Used for generating answers

---

## ⚠️ Limitations

* GPT-2 provides **basic answers (not very intelligent)**
* No chat memory
* Limited reasoning capability

---

## 🚀 Future Improvements

* 🔥 Replace GPT-2 with:

  * OpenAI GPT
  * Groq LLM
  * FLAN-T5 / LLaMA
* 💬 Add chat history (memory)
* 🌐 Build UI using Streamlit
* ☁️ Deploy to cloud
* 📊 Add evaluation metrics

---

## 🧪 Sample Questions

Try asking:

* What does Tesla do?
* When was Microsoft founded?
* Compare Nvidia and Google
* What products does Apple make?

---

## 🎯 Key Learnings

* RAG pipeline design
* Embeddings and vector search
* Chunking strategies
* LLM integration
* Handling real-world dependency issues

---

## 👨‍💻 Author

Developed by **Devbrat Singh**

---

## ⭐ Final Note

This project demonstrates a **complete end-to-end RAG system**, similar to what is used in:

* AI Chatbots
* Document Search Systems
* Knowledge Assistants

---

🚀 *You have successfully built a real-world AI system!*
