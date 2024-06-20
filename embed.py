import json
from upstash_vector import Index, Vector
from openai import OpenAI
import dotenv
import os
dotenv.load_dotenv()

###
# this script embeds the markdown content of the docs.json file into the upstash vector index
# i used firecrawl to  get the mardkdown and metadata of the modal docs and then stored in a json file
# we use upstash vector db to store the embeddings of the markdown content
# we use openai to get the embeddings of the markdown content
###



OPENAI_API_KEY =  os.getenv("OPENAI_API_KEY")
UPSTASH_VECTOR_REST_URL = os.getenv("UPSTASH_VECTOR_REST_URL")
UPSTASH_VECTOR_REST_TOKEN = os.getenv("UPSTASH_VECTOR_REST_TOKEN")
JSON_FILE_PATH = 'docs.json'  


index = Index(url=UPSTASH_VECTOR_REST_URL, token=UPSTASH_VECTOR_REST_TOKEN)

client = OpenAI(api_key=OPENAI_API_KEY)

# Function to get embedding for a given content
def get_embedding(content):
    response = client.embeddings.create(
        input=content,
        model="text-embedding-3-small"
    )
    print(response.data[0].embedding)
    embedding = response.data[0].embedding
    print(f"Embedding for {content}: {embedding}")
    return embedding

# Load JSON data
with open(JSON_FILE_PATH, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Process each chunk in the JSON file
for chunk in data:
    markdown = chunk.get('markdown', '')
    metadata = chunk.get('metadata', {})
    source_url = metadata.get('sourceURL', '')
    og_description = metadata.get('ogDescription', '')
    if not markdown or not source_url:
        print(f"Skipping chunk due to missing markdown or sourceURL.")
        continue

    embedding = get_embedding(markdown)
    
    # Metadata to be stored with the vector
    metadata = {
        "sourceURL": source_url,
        "ogDescription": og_description,
        "markdown": markdown  # Including markdown content in metadata
    }
    
    # Create a vector object
    vector = Vector(id=source_url, vector=embedding, metadata=metadata)
    
    # Upsert the vector into the Upstash index
    index.upsert(vectors=[vector])
    print(f"Upserted vector for {source_url}")

print("All markdown chunks have been embedded and upserted.")
