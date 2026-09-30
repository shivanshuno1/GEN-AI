from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# 1. Load TXT file
loader = TextLoader("vector_stores/document.txt", encoding="utf-8")

documents = loader.load()

print("Loaded document:")
print(documents)


# 2. Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=64,
    chunk_overlap=20
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i + 1}:")
    print(chunk.page_content)


# 3. Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B"
)


# 4. Store chunks + embeddings in Chroma
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="my_documents",
    persist_directory="./chroma_db"
)

print("\nDocuments successfully stored in Chroma!")


query = "What is the branch of artificial intelligence"

# 5. Perform similarity search
results = vector_store.similarity_search(query, k=3)

print("\nSimilarity search results:")
for i, result in enumerate(results):
    print(f"\nResult {i + 1}:")
    print(result.page_content)