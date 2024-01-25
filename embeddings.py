import os

from dotenv import load_dotenv
from openai import OpenAI

# Loads the .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
organization = os.getenv("OPENAI_ORG_ID")


def find_similar_words(word, n=5):
    client = OpenAI(organization=organization, api_key=api_key)

    # Create an embedding for 'word'
    embedding_response = client.embeddings.create(input=word, model="text-embedding-ada-002")
    word_embedding = embedding_response.result["alternatives"][0]["embedding"]

    # Create embeddings for multiple words (in this example, common English words are used)
    possible_words = ["weather", "either", "however", "although", "about", "because", "where", "while"]
    embeddings_response = client.embeddings.create(input=possible_words, model="text-embedding-ada-002")

    # Calculate the similarity for each word
    similarities = []
    for i, embedding_data in enumerate(embeddings_response.result["alternatives"]):
        cos_similarity = cosine_similarity(word_embedding, embedding_data["embedding"])
        similarities.append((possible_words[i], cos_similarity))

    # Sort by similarity and return top 'n' words
    similar_words = sorted(similarities, key=lambda x: x[1], reverse=True)[:n]
    return [word for word, _ in similar_words]


def cosine_similarity(vec1, vec2):
    # Function to calculate cosine similarity
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a ** 2 for a in vec1) ** 0.5
    magnitude2 = sum(b ** 2 for b in vec2) ** 0.5
    return dot_product / (magnitude1 * magnitude2)


# Find words similar to 'whether'
similar_words = find_similar_words("whether")
print(similar_words)
