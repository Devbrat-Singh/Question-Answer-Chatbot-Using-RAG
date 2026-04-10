# package does:

# langchain → core framework
# langchain-community → integrations (loaders, tools, etc.)
# langchain-text-splitters → for chunking documents (important for RAG)
# langchain-openai → OpenAI integration
# langchain-chroma → vector DB (for embeddings storage)

import os
from dotenv import load_dotenv

load_dotenv() 

# Method to load document


from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader

def load_documents(docs_path="docs"):
    # It will load all text files from the docs directory
    print(f"Loading document from {docs_path}...")

    # Check docs directory exists or not
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files")
    
    # Load all .txt files from the docs directory
    loader=DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents=loader.load()

    if len(documents)==0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your documents")
     
    #  To view content
    for i,doc in enumerate(documents[:2]): # Show first 2 documents
        print(f"\nDocument {i+1}: ")
        print(f" Sources: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f" Content preview:{doc.page_content[:100]}...")
        print(f" metadata:{doc.metadata}")

    return documents    


# If file contains special characters so we have to clean it----> use this when 1st method shows encoded error
from langchain_core.documents import Document

def load_documents2(docs_path="docs"):
    print(f"Loading document from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"{docs_path} does not exist")

    documents = []

    for file in os.listdir(docs_path):
        if file.endswith(".txt"):
            file_path = os.path.join(docs_path, file)

            try:
                # 🔥 Read file safely
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

                # 🔥 Clean text (IMPORTANT)
                text = text.replace("\n", " ")
                text = text.replace("\t", " ")
                text = text.encode("ascii", "ignore").decode()  # remove weird symbols

                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": file_path}
                    )
                )

            except Exception as e:
                print(f"Skipping {file}: {e}")

    if len(documents) == 0:
        raise FileNotFoundError("No valid documents found")

    # Preview
    for i, doc in enumerate(documents[:len(documents)]):
        print(f"\nDocument {i+1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Length: {len(doc.page_content)}")
        print(f"Preview: {doc.page_content[:100]}...")

    return documents

# Method to chunk document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents,chunk_size=800,chunk_overlap=50):
    """" Split the documents into smaller chunks with overlap """
    print("Splitting documents into chunks...")

    text_spliter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks=text_spliter.split_documents(documents)

    # Preview
    if chunks:
        for i,chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1}----")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content[:200].replace("\n"," "))
            print("-"*50)

        if len(chunks)>5:
            print(f"\n.....and {len(chunks)-5} more chunks")

    return chunks            


# Method for Vector Embedding and storing

# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# from langchain_chroma import Chroma  

from langchain_community.vectorstores import FAISS

def create_vector_store(chunks, persist_directory="db/chroma_db"): # directory tell location to store vector embedding
    """ Create and persist ChromaDB vector store"""
    print("Creating and Storing in ChromaDB...")

    # embedding_model=OpenAIEmbeddings(model="text-embedding-3-small")
    embedding_model = HuggingFaceEmbeddings( model_name="all-MiniLM-L6-v2")

    # Create ChromaDB vector store
    print("--- Creating vector store ---")
  
    # Take chunks and convert it to vector embedding and store it to
    # vector_store=Chroma.from_documents(
    #     documents=chunks,
    #     embedding=embedding_model,
    #     persist_directory=persist_directory,
    #     collection_metadata={"hnsw:space":"cosine"}
    # )

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )
    vector_store.save_local("faiss_index")

    print("--- Finished Creating vector store ---")

    print(f"Vector store created and saved to {persist_directory}")

    return vector_store


# Step 1: Load data

documents=load_documents2(docs_path="docs")

# Step 2: Split Documents(Chunking)

chunks=split_documents(documents=documents,chunk_size=800,chunk_overlap=50)

# Step 3: Generate Embeddings & Store in Vector Database

# if not os.path.exists("db/chroma_db"):
# vector_store=create_vector_store(chunks=chunks)
# else:
#     print("Skip rebuilding DB as it already exists")

if not os.path.exists("faiss_index"):
    vector_store = create_vector_store(chunks=chunks)
else:
    print("Using existing FAISS index")



# Step5: Create Retriever

# Part 1: Load Vector DB + Retriever

def load_vector_store(persist_directory="db/chroma_db"):
    print("\nLoading vector store...")

    # Same embedding model used during ingestion
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Load existing Chroma DB from disk
    # vector_store = Chroma(
    #     persist_directory=persist_directory,
    #     embedding_function=embedding_model
    # )

    vector_store = FAISS.load_local(
        "faiss_index",
        embedding_model,
        allow_dangerous_deserialization=True
    )

    # Convert vector DB into retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    return retriever

# Part 2: Create QA Chain (LLM)

from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

def create_qa_chain(retriever):
    print("Loading LLM...")

    # Load local HuggingFace Model (used to generate answer)
    pipe=pipeline(
        "text-generation", # task type
        model="gpt2", #LLM model
        # max_length=200 # max output length
        max_new_tokens=200,
        truncation=True
    )

    # wrap HuggingFace model into LangChain format
    llm=HuggingFacePipeline(pipeline=pipe)

    from langchain.chains import RetrievalQA

    # Create RAG QA chain
    qa_chain=RetrievalQA.from_chain_type(
        llm=llm,                    # LLM generates answer
        retriever=retriever,        # fetches relevant chunks
        return_source_documents=True# also return sources
    )

    return qa_chain

# Part 3: Ask Questions

def ask_questions(qa_chain):
    print("\nAsk questions about your documents (type 'exit' to quit)\n")
    while True:
        query=input("Please Ask your Question:")

        if query.lower()=="exit":
            print("Existing...")
            break

        result=qa_chain.invoke({"query":query})

        print("\nAnswer: ")
        print(result["result"])

        print("\nSources: ")
        for doc in result["source_documents"]:
            print("-",doc.metadata["source"])

        print("\n"+ "="*60) 


# Retrieval

retriever=load_vector_store()
qa_chain=create_qa_chain(retriever=retriever)

ask_questions(qa_chain=qa_chain)
