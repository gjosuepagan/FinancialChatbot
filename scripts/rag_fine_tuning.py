import json
import glob
import os
import tensorflow as tf
from keras_nlp.models import GemmaCausalLM
from keras.optimizers import Adam
from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone, ServerlessSpec
import numpy as np

#Connecting Pinecone
pc = Pinecone(
        api_key="pcsk_6t5DTe_G5S8dt9DpQQqdXTYeVWrY61j1u4rQa7rXPqLBrmq9YTNpVw9b84cYLuK2j8uX2G"
    )

index_name = 'pdf-vectorised'
if index_name not in pc.list_indexes().names():
    raise ValueError("Index of embeddings not available. Please run embedding script first")

pinecone_index = pc.Index(index_name)

# Load Gemma model
kggl_moneybot_model = GemmaCausalLM.from_preset("gemma2_2b_en")

# Load the embedding model (needed to embed the query)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define a retriever function using the Pinecone index
def retriever(query, k=4):
    # Convert the query into an embedding vector
    query_embedding = embedding_model.encode([query])
    # Search Pinecone for the top k documents
    results = pinecone_index.query(vector=query_embedding[0], top_k=k, include_values=False)
    # Retrieve document contents for each result
    retrieved_docs = [match['metadata']['text'] for match in results['matches']]
    return retrieved_docs

def fine_tune_rag(retriever, model, train_data, optimizer, k=4):
    """
    Fine-tune the model using RAG.
    :param retriever: Pinecone retriever for document retrieval.
    :param model: Pre-trained language model (GemmaCausalLM in this case).
    :param train_data: Training dataset containing (query, answer) pairs.
    :param optimizer: Optimizer for fine-tuning the model.
    :param k: Number of documents to retrieve.
    """
    # Loop through training data (query, answer pairs)
    for query, answer in train_data:
        # Step 1: Retrieve relevant documents based on the query
        retrieved_docs = retriever(query, k=k)

        # Step 2: Combine the retrieved documents with the query as input
        context = "\n".join(retrieved_docs) + "\n\n" + query
        
        # Step 3: Feed the context (query + retrieved docs) to the model
        inputs = model.preprocess_input(context)
        
        # Step 4: Fine-tune the model using the expected output (answer)
        outputs = model.preprocess_output(answer)

        # Forward pass to compute loss
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss = model.compute_loss(predictions, outputs)

        # Step 5: Backpropagation and optimization
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        # Optionally print loss for monitoring
        print(f"Loss: {loss.numpy()}")

# Load all JSON datasets for fine-tuning
dataset_path = os.path.expanduser("~/data/fine_tuning_datasets/*.json")
file_paths = glob.glob(dataset_path)

# Aggregate data from all JSON files
all_train_data = []

for file_path in file_paths:
    with open(file_path, 'r') as file:
        dataset = json.load(file)
        # Assuming each JSON file is a dictionary with format { "query": "question", "answer": "answer text" }
        train_data = [(item['query'], item['answer']) for item in dataset]
        all_train_data.extend(train_data)  # Collect data from all files

# Optimizer for training
optimizer = Adam(learning_rate=5e-5)

# Fine-tune the model using RAG
fine_tune_rag(retriever, kggl_moneybot_model, all_train_data, optimizer, k=4)

# Save the fine-tuned model
kggl_moneybot_model.save("kggl_moneybot_model")
print("Fine-tuned model saved as 'kggl_moneybot_model'")