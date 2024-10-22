from PyPDF2 import PdfReader
import os

# Load and extract text from a PDF
reader = PdfReader('data/raw_pdfs/financial_document.pdf')
text = ""
for page in reader.pages:
    text += page.extract_text()

# Save processed text for RAG
with open('data/processed_pdfs/financial_text.txt', 'w') as f:
    f.write(text)
