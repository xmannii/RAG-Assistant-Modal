import os
import tiktoken


###
# This script tokenizes the content of the MDX files of docs chunk in the chunks directory.
# It prints a summary of the token counts, including the highest, lowest, and total counts.
# all the modal docs chunks are below the 8191 token limit so we can safely use the text-embedding-3-small model
###


# Constants
TOKEN_LIMIT = 8191
ENCODING_NAME = "text-embedding-3-small"  

# Function to load and encode the content of an MDX file
def tokenize_and_check(file_path, encoding):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    tokens = encoding.encode(content)
    num_tokens = len(tokens)
    if num_tokens > TOKEN_LIMIT:
        print(f"Warning: '{file_path}' has {num_tokens} tokens, which exceeds the limit of {TOKEN_LIMIT} tokens.")
    return num_tokens

# Load the encoding
encoding = tiktoken.encoding_for_model(ENCODING_NAME)

# Directory containing the MDX files
results_dir = 'chunks'

# Variables to track token counts
token_counts = []

# Check each file in the results directory
for filename in os.listdir(results_dir):
    if filename.endswith('.mdx'):
        file_path = os.path.join(results_dir, filename)
        num_tokens = tokenize_and_check(file_path, encoding)
        print(f"'{file_path}' has {num_tokens} tokens.")
        token_counts.append(num_tokens)

# Summary of token counts
if token_counts:
    highest_count = max(token_counts)
    lowest_count = min(token_counts)
    total_count = sum(token_counts)
    print("\nSummary of token counts:")
    print(f"Highest token count: {highest_count}")
    print(f"Lowest token count: {lowest_count}")
    print(f"Total token count: {total_count}")
else:
    print("No MDX files found in the directory.")

print("Tokenization check completed.")
