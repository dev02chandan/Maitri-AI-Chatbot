import streamlit as st
import google.generativeai as genai
import os
import faiss
import numpy as np
import pandas as pd
import textwrap

# App title and configuration
st.set_page_config(page_title="Maitri AI Chatbot")

# Gemini API Credentials
with st.sidebar:
    st.title('Maitri AI Chatbot')
    if 'GEMINI_API_KEY' in st.secrets:
        st.success('API key already provided!', icon='✅')
        gemini_api = st.secrets['GEMINI_API_KEY']
    else:
        gemini_api = st.text_input('Enter Gemini API token:', type='password')
        if not gemini_api:
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    os.environ['GEMINI_API_KEY'] = gemini_api

    # st.subheader('Models and parameters')
    # temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    # top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    # max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8)

# Configure Gemini API
genai.configure(api_key=gemini_api)

# Load JSON data
data = pd.read_json('data.json')

df = pd.DataFrame(data)
df.columns = ['Title', 'Text']

# Get the embeddings of each text and add to an embeddings column in the dataframe
def embed_fn(title, text):
    return genai.embed_content(model='models/text-embedding-004', content=text)["embedding"]

df['Embeddings'] = df.apply(lambda row: embed_fn(row['Title'], row['Text']), axis=1)

# Function to find the best passages
def find_best_passages(query, dataframe, top_n=3):
    query_embedding = genai.embed_content(model='models/text-embedding-004', content=query)["embedding"]
    dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding)
    top_indices = np.argsort(dot_products)[-top_n:][::-1]
    return dataframe.iloc[top_indices]['Text'].tolist()

# Function to make prompt
def make_prompt(query, relevant_passages):
    escaped_passages = [passage.replace("'", "").replace('"', "").replace("\n", " ") for passage in relevant_passages]
    joined_passages = "\n\n".join(f"PASSAGE {i+1}: {passage}" for i, passage in enumerate(escaped_passages))
    prompt = textwrap.dedent(f"""
    Persona: You are Maitri AI Chatbot, representing MaitriAI, a leading software company specializing in web application development, website design, logo design, software development, and cutting-edge AI applications. You are knowledgeable, formal, and detailed in your responses.

    Task: Answer questions about Maitri AI, its services, and related information. Provide detailed and kind responses in a conversational manner.

    Format: Respond in a formal and elaborate manner, providing as much relevant information as possible. If you do not know the answer, respond by saying you do not know.

    Context: {joined_passages}

    QUESTION: '{query}'
    
    ANSWER:
    """)
    return prompt

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating Gemini response
def generate_gemini_response(query):
    relevant_passages = find_best_passages(query, df)
    prompt = make_prompt(query, relevant_passages)
    response = genai.GenerativeModel('models/gemini-1.5-flash-latest').generate_content(prompt)
    return response.text

# User-provided prompt
if prompt := st.chat_input(disabled=not gemini_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_gemini_response(prompt)
            placeholder = st.empty()
            placeholder.markdown(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
