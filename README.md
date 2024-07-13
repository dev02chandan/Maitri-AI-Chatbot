
# Maitri AI Chatbot

Maitri AI Chatbot is a Streamlit-based application designed to provide detailed and kind responses to queries about Maitri AI. The chatbot uses the Google Gemini API for generating and embedding content.

## Features

- Provides information about Maitri AI, its services, and related topics.
- Precomputes embeddings for faster responses.
- Retrieves and presents the top 3 relevant passages for comprehensive answers.
- User-friendly interface with a conversational tone.

## Prerequisites

- Python 3.7 or higher
- Streamlit
- Google Generative AI library
- FAISS (for similarity search)
- pandas

## Setup Instructions

### Step 1: Prepare the Data

1. Create a JSON file named `data.json` in the following format:

```json
[
    {
        "title": "About Maitri AI",
        "content": "Maitri AI is an artificial intelligence company focused on developing advanced AI technologies to enhance human experiences and empower businesses across various industries. Their mission is to harness the power of AI to drive innovation, solve complex problems, and create a positive impact on society. Maitri AI believes in the potential of AI to transform the way we live, work, and interact. They specialize in developing intelligent systems that can understand and interpret human language, recognize patterns in data, make intelligent decisions, and continuously learn and adapt. [1]"
    },
    {
        "title": "Maitri AI Mission",
        "content": "Maitri AI's mission is to harness AI's potential to transform lives, push boundaries, build ethical solutions, drive social impact, empower individuals, and shape a better future. [2]"
    },
    ...
]
```

### Step 2: Precompute Embeddings

1. Ensure the required libraries are installed:

```bash
pip install pandas google-generativeai pyarrow
```

2. Run the setup script to compute embeddings and save the DataFrame:

```bash
python3 setup.py
```

### Step 3: Run the Streamlit Application

1. Ensure the required libraries are installed:

```bash
pip install streamlit faiss-cpu
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

## Files

- `data.json`: The JSON file containing data about Maitri AI.
- `setup.py`: Script to compute embeddings and save the data.
- `app.py`: The main Streamlit application.

## Usage

1. Add your data to `data.json` in the specified format.
2. Run `setup.py` to precompute embeddings and save the data.
3. Run `app.py` to start the Streamlit application and interact with the Maitri AI Chatbot.

## License

This project is licensed under the MIT License.
