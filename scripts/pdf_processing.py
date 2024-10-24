import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from pinecone import PineconeClient
import pinecone

#Connecting Pinecone
pinecone.init(api_key='pcsk_6t5DTe_G5S8dt9DpQQqdXTYeVWrY61j1u4rQa7rXPqLBrmq9YTNpVw9b84cYLuK2j8uX2G')
pinecone_index = pinecone.Index("pdf_vectorised")

# Embedding text chunk with SentenceTransformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Working directory
cwd = os.getcwd()

# Raw PDFs folder
raw_pdf_folder = os.path.join(cwd, 'data', 'raw_pdf')
processed_pdf_folder = os.path.join(cwd, 'data', 'processed_pdf_rag')

# List of pdf in folder
raw_pdfs = os.listdir(raw_pdf_folder)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

for pdf in raw_pdfs:
    # Full path to the PDF file
    pdf_path = os.path.join(raw_pdf_folder, pdf)
    
    # Load and extract text from the PDF
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        # Splitting into chunks
        text_chunks = text_splitter.split_text(text)

        # Save processed text for RAG just in case
        txt_filename = os.path.join(processed_pdf_folder, pdf[:-4] + '_chunks.txt')
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(text_chunks))
        print(f"Processed and saved: {txt_filename}")

        # Chunk embedding
        embeddings = embedding_model.encode(text_chunks_show_progress_bar=True)

        # Uploading to Pinecone
        for i,embedding in enumerate(embeddings):
            chunk_id = f"{pdf[:-4]}_chunk_{i}"
            pinecone_index.upsert(vectors=[(chunk_id, embedding)])

        print(f'Embeddings for {pdf} in Pinecone')
        
    except Exception as e:
        print(f"Error processing {pdf}: {e}")
