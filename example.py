import os

from dotenv import load_dotenv
from openai import OpenAI

# Loads the .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
organization = os.getenv("OPENAI_ORG_ID")

client = OpenAI(organization=organization, api_key=api_key)

response = client.embeddings.create(
    input="Your text string goes here",
    model="text-embedding-ada-002",
    
)

print(response.data[0].embedding)
