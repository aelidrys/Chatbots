from llama_index.core import load_index_from_storage
from llama_index.core.storage import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import CondenseQuestionChatEngine
import dotenv


# Set your Groq API key
GROQ_API_KEY = dotenv.get_key('.env', "GROQ_API_KEY")



# Load embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load your index
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context, embed_model=embed_model)

# Use Groq as LLM
llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)

# Create the query engine
query_engine = index.as_query_engine(llm=llm)

# Create the chat engine (this is a real agent-style chatbot)
chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine,
    llm=llm,
    system_prompt="You are a helpful assistant. Answer briefly and clearly.",
    verbose=False
)

# Run the chatbot
while True:
    query = input("User: ")
    if query == 'quit':
        break
    response = chat_engine.chat(query)
    print("Chatbot: ", response)
