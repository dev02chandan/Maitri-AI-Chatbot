import pandas as pd
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key='AIzaSyCJyQo4J3-Xa9vpqjzMt6bmtJzxIGEOOjY')

# Load JSON data
data = pd.read_json('data.json')

df = pd.DataFrame(data)
df.columns = ['Title', 'Text']

# Get the embeddings of each text and add to an embeddings column in the dataframe
def embed_fn(title, text):
    return genai.embed_content(model='models/text-embedding-004', content=text)["embedding"]

df['Embeddings'] = df.apply(lambda row: embed_fn(row['Title'], row['Text']), axis=1)

# Save the dataframe to a feather file
df.to_feather('data_with_embeddings.feather')
