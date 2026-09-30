from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
    top_k_results=3,
    doc_content_chars_max=4000
)

docs = retriever.invoke(
    "Artificial Intelligence"
)

print("\nRetrieved documents:")
for i, doc in enumerate(docs):  
    print(f"\nDocument {i + 1}:")
    print(doc.page_content)