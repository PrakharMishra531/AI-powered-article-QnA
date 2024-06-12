# AI Powered Article QnA
### Streamlit-Based URL Embedding and Q&A System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

A modern web application built with Streamlit that allows users to paste URLs, extract embeddings from the content, and use a Retrieval-Augmented Generation (RAG) approach to answer user questions based on the articles' content.

## Features

- **URL Input:** Paste up to 3 URLs to extract content.
- **Embedding and RAG:** Generate embeddings for the articles and use RAG for enhanced question answering.
- **Hugging Face Integration:** Leverage models from Hugging Face for embeddings and language modeling.
- **Local Model Alternative:** Option to use local models like Ollama Mistral.
- **User-Friendly Interface:** Simple and intuitive Streamlit-based UI.

## Demo

![Demo](demo.gif)


## Tech Stack

- ![Streamlit](https://img.shields.io/badge/Streamlit-black?style=flat&logo=streamlit) **Streamlit**
- ![LangChain](https://img.shields.io/badge/LangChain-blue?style=flat&logo=chain) **LangChain**
- ![HuggingFace](https://img.shields.io/badge/Hugging%20Face-yellow?style=flat&logo=huggingface) **Hugging Face**
- ![Ollama](https://img.shields.io/badge/Ollama-green?style=flat) **Ollama**
- ![Mistral](https://img.shields.io/badge/Mistral-purple?style=flat) **Mistral**
- ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) **Python**
- ![FAISS](https://img.shields.io/badge/FAISS-FF6F00?style=flat) **FAISS**

## Installation

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```

2. **Create and Activate Virtual Environment:**
    ```bash
    python -m venv env
    source env/bin/activate   # On Windows use `env\Scripts\activate`
    ```

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set Up Environment Variables:**
    Create a `.env` file and add your Hugging Face API token and other necessary configurations.
    ```bash
    HUGGINGFACE_API_KEY=your_api_key_here
    ```

## Usage

1. **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

2. **Interact with the Application:**
    - Open the web browser at the provided local URL.
    - Paste up to 3 URLs.
    - Ask questions and get answers based on the content of the URLs.

## Dependencies

- `streamlit`
- `transformers`
- `faiss-cpu`
- `sentence-transformers`
- `requests`
- `beautifulsoup4`
- `numpy`
- `pandas`


