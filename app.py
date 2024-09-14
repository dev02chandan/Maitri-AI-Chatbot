import streamlit as st
import google.generativeai as genai
import os
import faiss
import numpy as np
import pandas as pd
import textwrap

# App title and configuration
st.set_page_config(page_title="OnlyProfitYou AI Chatbot")

# Configure Gemini API
if 'GEMINI_API_KEY' in st.secrets:
    gemini_api = st.secrets['GEMINI_API_KEY']
else:
    gemini_api = st.text_input('Enter Gemini API token:', type='password', key='api_input')
    if gemini_api and gemini_api.startswith('r8_') and len(gemini_api) == 40:
        st.secrets['GEMINI_API_KEY'] = gemini_api
        st.experimental_rerun()

if 'GEMINI_API_KEY' in st.secrets:
    genai.configure(api_key=st.secrets['GEMINI_API_KEY'])
    hide_sidebar = True
else:
    st.sidebar.write("Please enter your API key")
    hide_sidebar = False

# Hide sidebar if API key is set
if hide_sidebar:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Add logo
logo_path = "logo.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=200)

# Load the dataframe with precomputed embeddings
df = pd.read_feather('data_with_embeddings.feather')

# Function to find the best passages
def find_best_passages(query, dataframe, top_n=3):
    query_embedding = genai.embed_content(model='models/text-embedding-004', content=query)["embedding"]
    dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding)
    top_indices = np.argsort(dot_products)[-top_n:][::-1]
    
    # Combine 'Title' and 'Text' for each of the top matches to provide better context
    return dataframe.iloc[top_indices].apply(lambda row: f"{row['Title']}: {row['Text']}", axis=1).tolist()

# Function to make prompt
def make_prompt(query, relevant_passages):
    escaped_passages = [passage.replace("'", "").replace('"', "").replace("\n", " ") for passage in relevant_passages]
    joined_passages = "\n\n".join(f"PASSAGE {i+1}: {passage}" for i, passage in enumerate(escaped_passages))
    prompt = textwrap.dedent(f"""
    Persona: You are OnlyProfitYou (OPU) AI Assistant, representing OPU, a transformative platform dedicated to democratizing financial success through stock market trading and education. You are knowledgeable, formal, and detailed in your responses.

    Task: Answer questions about OnlyProfitYou (OPU), its services, stock market trading strategies, educational programs, career development, and franchise opportunities. Provide detailed and helpful responses in a conversational manner. If the context from the provided passages is relevant to the query, use it to give a comprehensive answer. If the context is not relevant, acknowledge that you do not know the answer. At the end of your response, direct the user to the website: https://www.onlyprofityou.com for more details.

    Format: Respond in a formal and informative manner, providing relevant information. If you do not know the answer, respond politely by saying you do not know.

    Context: {joined_passages}

    QUESTION: '{query}'

    ANSWER:
    """)
    return prompt

# Initialize chat history with Gemini's API if not already set
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to OnlyProfitYou AI! How can I assist you with information about OnlyProfitYou today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to OnlyProfitYou AI! How can I assist you with information about OnlyProfitYou today?"}]
    st.session_state.chat_history = []  # Clear the Gemini chat history
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating Gemini response with chat history
def generate_gemini_response(query):
    # Retrieve relevant passages
    relevant_passages = find_best_passages(query, df)
    
    # Create a prompt with the relevant passages
    prompt = make_prompt(query, relevant_passages)
    
    # If chat is already ongoing, append new message to the history
    if "chat" not in st.session_state:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        st.session_state.chat = model.start_chat(history=st.session_state.chat_history)
    
    # Send the user's message to Gemini with the chat history
    chat = st.session_state.chat
    response = chat.send_message(prompt)  # Update with the query
    
    # Update chat history with user and assistant messages
    st.session_state.chat_history.append({"role": "user", "parts": prompt})
    st.session_state.chat_history.append({"role": "assistant", "parts": response.text})
    
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
