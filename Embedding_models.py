'''from sentence_transformers import SentenceTransformer

# Load the model (downloads once)
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

# Your sentences
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium."
]

# Get embeddings (lists of numbers)
embeddings = model.encode(sentences)

# Print the first embedding's first 5 values
print(embeddings)'''


from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

sentences1 = ["Shivanshu Rawat", "Kritika Panwar"]
embeddings = model.encode(sentences1)

sentences2 = ["Vaibhav Sharma", "Kritika Panwar"]
embeddings2 = model.encode(sentences2)

similarity1 = util.cos_sim(embeddings[0], embeddings[1])
similarity2 = util.cos_sim(embeddings2[0], embeddings2[1])
print(f"Similarity 1 Between {sentences1[0]} and {sentences1[1]}: {similarity1.item():.4f}")
print(f"Similarity 2 Between {sentences2[0]} and {sentences2[1]}: {similarity2.item():.4f}")