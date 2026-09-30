

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")


documents = ["Shivanshu Rawat is a software engineer.", "Kritika Panwar is a Doctor.",
             "Python is the easiest programming language to learn.", 
             "Shivanshu Rawat loves Kritika Panwar."
             "Kritika Panwar loves Shivanshu Rawat is not sure.",
             "Kritika Panwar and shivanshu rawat both are from uttarkhand."
             ]


query = "Who is Shivanshu Rawat?"

query2 = "Do both Shivanshu Rawat and Kritika Panwar love each other?"

# Get embeddings for documents and query
document_embeddings = model.encode(documents)
query_embedding = model.encode(query)
query2_embedding = model.encode(query2)

# Compute cosine similarity between queries and documents
similarities = util.cos_sim(query_embedding, document_embeddings)
similarities2 = util.cos_sim(query2_embedding, document_embeddings)

# Get the index of the most similar document for each query
most_similar_index = similarities.argmax()
most_similar_index2 = similarities2.argmax()

print(f"Most similar document for query 1: {documents[most_similar_index]}")
print(f"Most similar document for query 2: {documents[most_similar_index2]}")
