import asyncio
import sys
import types

# Fix torch + asyncio bugs in Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import torch
torch.classes = types.SimpleNamespace()
sys.modules['torch.classes'] = torch.classes





import streamlit as st
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.llms.groq import Groq
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # Add this import
import dotenv


# Set your Groq API key
GROQ_API_KEY = dotenv.get_key('../../lama-index-agens/.env', "GROQ_API_KEY")

with st.sidebar:
    st.title("Chatbot sidebar")
    st.markdown('''
    ### About
    This is a chatbot app.
    - [Streamlit](https://streamlit.io/)
    - [LlamaIndex](https://docs.llamaindex.ai/en/stable/)
    - [Using HuggingFace](https://huggingface.co/)
    ''')

def main():
    st.header("ChatbotAy")
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    documents = reader.load_data()

    # Configure local embedding model
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Configure LLM
    llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)


    # Set both LLM and embedding model in Settings
    Settings.embed_model = embed_model

    index = VectorStoreIndex.from_documents(documents)
    query = st.text_input("Enter your question:")
    if query:
        chat_engine = index.as_chat_engine(llm=llm, verbose=True)
        response = chat_engine.chat(query)
        st.write(response.response)

if __name__ == "__main__":
    main()