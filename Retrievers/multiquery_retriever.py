import logging

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# 1. ENABLE LOGGING
# ============================================================

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


# ============================================================
# 2. LOAD DOCUMENT
# ============================================================

loader = TextLoader(
    "vector_stores/document.txt",
    encoding="utf-8"
)

documents = loader.load()


# ============================================================
# 3. SPLIT DOCUMENT INTO CHUNKS
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# ============================================================
# 4. CREATE EMBEDDING MODEL
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B"
)


# ============================================================
# 5. CREATE VECTOR STORE
# ============================================================

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="multiquery_test",
    persist_directory="./multiquery_chroma_db"
)


# ============================================================
# 6. CREATE NORMAL VECTOR STORE RETRIEVER
# ============================================================

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 2
    }
)


# ============================================================
# 7. CREATE LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key="",
    temperature=0.7
)


# ============================================================
# 8. CREATE MULTI-QUERY RETRIEVER
# ============================================================

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)


# ============================================================
# 9. USER QUERY
# ============================================================

query = "How does machine learning learn from data?"


# ============================================================
# 10. RETRIEVE DOCUMENTS
# ============================================================

docs = multi_query_retriever.invoke(query)


# ============================================================
# 11. PRINT RESULTS
# ============================================================

print("\n======================================")
print("MULTI-QUERY RETRIEVER RESULTS")
print("======================================")

print("\nOriginal Query:")
print(query)


print("\nRetrieved Documents:")

for i, doc in enumerate(docs):

    print(f"\n--- Result {i + 1} ---")

    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)