import os
import json
import tempfile
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()
VECTOR_STORE_PATH = "vector_store"
HISTORY_FILE = os.path.join(VECTOR_STORE_PATH, "conversation_history.json")

# Ensure folder exists
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

# Initialize sentence transformer embeddings (free)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def process_files(files, chunk_size=1000, chunk_overlap=100):
    docs = []
    for file in files:
        file_ext = os.path.splitext(file.name)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        
        if file_ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif file_ext == ".csv":
            loader = CSVLoader(tmp_path)
        elif file_ext == ".txt":
            loader = TextLoader(tmp_path)
        else:
            continue
        docs.extend(loader.load())

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    # Store vectors
    Chroma.from_documents(chunks, embeddings, persist_directory=VECTOR_STORE_PATH)

def ask_question(query, k=3):
    vectordb = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)

    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa({"query": query})

    answer = result["result"]
    sources = result.get("source_documents", [])

    # Save history
    log_result(query, answer, sources)

    return answer, [doc.metadata for doc in sources]

def log_result(query, answer, sources):
    entry = {
        "query": query,
        "answer": answer,
        "sources": [doc.metadata for doc in sources]
    }

    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    else:
        history = []

    history.append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def load_conversation_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []