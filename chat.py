import json
import os
from upstash_vector import Index
from openai import OpenAI
from groq import Groq
import dotenv

###
# This script uses the Groq API to answer questions about the Modal docs.
# It takes a question as input and uses the openai embeddings and our vectore index to get the most relevant context.
# It then uses the Groq API to generate a response based on the user question and the context.
###

# Load environment variables
dotenv.load_dotenv()

# Constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UPSTASH_VECTOR_REST_URL = os.getenv("UPSTASH_VECTOR_REST_URL")
UPSTASH_VECTOR_REST_TOKEN = os.getenv("UPSTASH_VECTOR_REST_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize clients
index = Index(url=UPSTASH_VECTOR_REST_URL, token=UPSTASH_VECTOR_REST_TOKEN)
client = OpenAI(api_key=OPENAI_API_KEY)
groq = Groq(api_key=GROQ_API_KEY)

# System message for the assistant
SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are a helpful docs assistant working for Modal.com. Modal's platform empowers data/AI/ML teams to develop faster "
        "at lower cost by providing serverless compute. You answer the question given only using the context. If you do not know the answer, you can say 'I do not know' and the user will be notified.Only awnser questions about Modal.com and its services. Do not answer questions about other topics."
    ),
}

def get_embedding(content):
    """Get the embedding for the given content."""
    response = client.embeddings.create(
        input=content,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def ask_question(question):
    """Process the question and provide a response."""
    # Get the embedding for the question
    question_embedding = get_embedding(question)
   

    # Query the index for relevant context

    res = index.query(vector=question_embedding, top_k=2, include_metadata=True)
    context = []

    for r in res:
        if r.score > 0.7: # adjust this value to your liking
            print(f"Score: {r.score}")  # Print the score
            print (r.metadata['markdown'])
            
            context.append(r.metadata['markdown'])

    context = "\n".join(context)

    # Formulate the final prompt with the question and context
    final_prompt = f"Question: {question}\n\nContext: {context}"
    messages = [SYSTEM_MESSAGE, {"role": "user", "content": final_prompt}]
    
    # Get the response from the assistant
    chat_completion = groq.chat.completions.create(model="llama3-70b-8192", messages=messages)
    response_text = chat_completion.choices[0].message.content

    print("Response:", response_text)

# Main loop
if __name__ == "__main__":
    while True:
        user_question = input("Enter a question: ")
        if user_question.lower() == "exit":
            break
        ask_question(user_question)
