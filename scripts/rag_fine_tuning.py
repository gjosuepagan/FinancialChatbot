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

print("Starting script...")  # Confirm script start

# Connect to Pinecone
pc = Pinecone(
    api_key="pcsk_6t5DTe_G5S8dt9DpQQqdXTYeVWrY61j1u4rQa7rXPqLBrmq9YTNpVw9b84cYLuK2j8uX2G"
)
index_name = 'pdf-vectorised'
if index_name not in pc.list_indexes().names():
    raise ValueError("Index of embeddings not available. Please run embedding script first")
pinecone_index = pc.Index(index_name)

print("Connected to Pinecone and retrieved index.")

# Load Gemma model
kggl_moneybot_model = GemmaCausalLM.from_preset("gemma2_2b_en")
print("Loaded Gemma model.")

# Load the embedding model
embedding_model = SentenceTransformer('all-distilroberta-v1')
print("Loaded embedding model.")

# Define retriever function
def retriever(query, k=4):
    query_embedding = embedding_model.encode([query])
    results = pinecone_index.query(vector=query_embedding[0], top_k=k, include_metadata=True)
    retrieved_docs = [match['metadata']['text'] for match in results['matches']]
    print(f"Retrieved {len(retrieved_docs)} docs for query.")  # Confirm retrieval
    return retrieved_docs

# Define fine-tuning function with debug prints
def fine_tune_rag(retriever, model, train_data, optimizer, k=4):
    print("Starting fine-tuning...")  # Debug start
    for i, (query, answer) in enumerate(train_data):
        retrieved_docs = retriever(query, k=k)
        context = "\n".join(retrieved_docs) + "\n\n" + query
        inputs = model.preprocess_input(context)
        outputs = model.preprocess_output(answer)

        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss = model.compute_loss(predictions, outputs)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        # Print loss every 10 steps
        if i % 10 == 0:
            print(f"Step {i}, Loss: {loss.numpy()}")
    print("Fine-tuning complete.")  # Debug end

# Load JSON datasets with debug prints
dataset_path = os.path.expanduser("~/data/fine_tuning_datasets/*.json")
file_paths = glob.glob(dataset_path)
print(f"Found {len(file_paths)} dataset files.")  # Debug dataset loading

all_train_data = []
for file_path in file_paths:
    with open(file_path, 'r') as file:
        dataset = json.load(file)
        train_data = [(item['query'], item['answer']) for item in dataset]
        all_train_data.extend(train_data)
print(f"Loaded {len(all_train_data)} training data pairs.")  # Confirm training data

# Initialize optimizer
optimizer = Adam(learning_rate=5e-5)

# Run fine-tuning
fine_tune_rag(retriever, kggl_moneybot_model, all_train_data, optimizer, k=4)

# Save the fine-tuned model
kggl_moneybot_model.save("kggl_moneybot_model")
print("Fine-tuned model saved as 'kggl_moneybot_model'")
