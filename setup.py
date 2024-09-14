import pandas as pd
import google.generativeai as genai
import os
import json

# Configure Gemini API
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=api_key)

# Load JSON data
with open('data.json', 'r') as file:
    data = json.load(file)

# Extract sections from the JSON structure
sections = data.get('sections', [])

# Prepare a list to store the extracted title and text from each section
rows = []

# Iterate through sections to extract the title and relevant text for embedding
for section in sections:
    title = section.get('title', 'Untitled Section')
    description = section.get('description', '')
    
    # Combine the title and description text for embedding
    text = f"{description} {json.dumps(section.get('features', ''))}"
    
    rows.append({'Title': title, 'Text': text})

# Create a DataFrame from the extracted data
df = pd.DataFrame(rows)

# Define the function to get embeddings from the content
def embed_fn(title, text):
    return genai.embed_content(model='models/text-embedding-004', content=text)["embedding"]

# Apply the embedding function to each row in the dataframe
df['Embeddings'] = df.apply(lambda row: embed_fn(row['Title'], row['Text']), axis=1)

# Save the dataframe to a feather file
df.to_feather('data_with_embeddings.feather')

print("Embeddings have been successfully added and saved to 'data_with_embeddings.feather'.")
