
# Maitri AI RAG-Based-Chatbot

Maitri AI Chatbot is a Streamlit-based application designed to provide detailed and kind responses to queries about Maitri AI. The chatbot uses the Google Gemini API for generating and embedding content. Check the app [here](https://maitriai-chatbot.streamlit.app/)

![image](https://github.com/user-attachments/assets/93fefa93-93e2-48da-b68b-4cec6f767f18)


## Features

- RAG Implementation with Gemini API.
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

### Step 2: Create a Virtual Environment

1. Create a virtual environment:

```bash
python3 -m venv env
```

2. Activate the virtual environment:

- On Windows:

```bash
.\env\Scripts activate
```

- On macOS and Linux:

```bash
source env/bin/activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```


### Step 3: Precompute Embeddings

Run the setup script to compute embeddings and save the DataFrame:

```bash
python3 setup.py
```

### Step 4: Run the Streamlit Application

Run the Streamlit app:

```bash
streamlit run app.py
```

## Files

- `data.json`: The JSON file containing data about Maitri AI. (You can add your data)
- `setup.py`: Script to compute embeddings and save the data. (For RAG)
- `app.py`: The main Streamlit application.

## Usage

1. Add your data to `data.json` in the specified format.
2. Run `setup.py` to precompute embeddings and save the data.
3. Run `app.py` to start the Streamlit application and interact with the Maitri AI Chatbot.

## License

This project is licensed under the MIT License.
