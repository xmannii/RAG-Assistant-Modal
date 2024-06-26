# Modal Docs Chatbot

This project aims to create a chatbot that can assist users in navigating and understanding the documentation of Modal.com. Modal is a platform that empowers data/AI/ML teams to develop faster at a lower cost by providing serverless compute. With the vast amount of documentation available, it can be challenging for users to find the information they need quickly. This chatbot addresses that issue by providing a conversational interface to access the relevant information from the docs.

## Importance of a Chatbot for Modal

1. **Improved User Experience**: A chatbot provides a more intuitive and user-friendly way for users to interact with the Modal documentation. Instead of manually searching through the docs, users can simply ask questions and receive direct answers.

2. **Faster Information Retrieval**: By leveraging natural language processing and vector search, the chatbot can quickly identify the most relevant sections of the documentation based on the user's question. This saves time and effort in finding the required information.

3. **Increased Engagement**: A chatbot encourages users to explore and learn more about Modal's platform. The interactive nature of the chatbot makes it more engaging and enjoyable for users to discover the features and capabilities of Modal.

4. **Reduced Support Burden**: By providing instant answers to common questions, the chatbot can reduce the workload on Modal's support team. This allows the support team to focus on more complex and unique inquiries.

## Project Overview

The project involves the following steps:

1. **Web Crawling**: We used the Firecrawl platform to crawl the documentation of Modal.com and retrieve the content of each document.

2. **Tokenization**: We created a `tokenizer.py` script to tokenize the content of each document and ensure it fits within the embedding model's limit.

3. **Embedding**: We embedded the tokenized content of each document along with its metadata into an Upstash Vector Database. This allows for efficient similarity search based on the document embeddings.

4. **Chatbot Development**: We developed a chatbot in Python that utilizes the OpenAI embeddings and the Upstash Vector Database. The chatbot takes a user's question as input, retrieves the most relevant context from the vector database, and generates a response using the Groq API.

5. **Context Filtering**: To ensure the quality of the retrieved context, we added a score threshold of 0.7. Only the context with a similarity score above this threshold is considered for generating the response.

By following these steps, we created a chatbot that can effectively assist users in finding the information they need from the Modal documentation. The chatbot provides a convenient and efficient way to navigate the docs, enhancing the overall user experience and making it easier for users to leverage the power of Modal's platform.
