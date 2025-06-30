import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding



# Load data
reader = SimpleDirectoryReader("./data", recursive=True, exclude_hidden=True)
docs = reader.load_data()

# Split the Docs to nodes
node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=0)
nodes = node_parser.get_nodes_from_documents(docs, show_progress=True)
print(f'Number of nodes: {len(nodes)}')


# Index
# Use Hugging Face embedding model (you can choose a different one)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create index with custom embedding model
index = VectorStoreIndex(nodes, embed_model=embed_model)
print(f"Index created: {index}")


# # Store Index
index.storage_context.persist("./storage")

