import streamlit as st
import google.generativeai as genai
import os

# Setup Gemini API
api_key_gemini = os.getenv('GEMINI_API_KEY')
if api_key_gemini is None:
    st.error("Gemini API key is not set in environment variables. Please set it before proceeding.")
    st.stop()

genai.configure(api_key=api_key_gemini)
model = genai.GenerativeModel('gemini-pro')

# Add logo
logo_path = "images/logo.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=200)

# Change app title
st.title("Maitri AI Chatbot")

# System prompt
system_prompt = (
    "Persona: You are Maitri AI Chatbot, representing MaitriAI, a leading software company specializing in web application development, "
    "website design, logo design, software development, and cutting-edge AI applications. You are knowledgeable, formal, and detailed in your responses.\n\n"
    "Task: Answer questions about Maitri AI, its services, and related information. Provide detailed and kind responses in a conversational manner.\n\n"
    "Context: MaitriAI is dedicated to innovation and exceptional results, transforming digital presence and helping businesses thrive in the digital era.\n\n"
    "Format: Respond in a formal and elaborate manner, providing as much relevant information as possible. If you do not know the answer, respond by saying you do not know."
)

# User input
user_input = st.text_area("Enter your prompt:", height=100)

# Generate button
if st.button("Generate Response"):
    if user_input:
        with st.spinner("Generating response..."):
            # Generate content with system prompt
            full_input = f"{system_prompt}\n\n{user_input}"
            response = model.generate_content(full_input)
            
            # Display the response
            st.subheader("Maitri AI's Response:")
            st.write(response.text)
    else:
        st.warning("Please enter a prompt.")


