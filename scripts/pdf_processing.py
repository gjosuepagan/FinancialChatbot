from PyPDF2 import PdfReader
import os

# Working directory
cwd = os.getcwd()

# Raw PDFs folder
pdf_folder = os.path.join(cwd, 'raw_pdfs')

# List of pdf in folder
raw_pdfs = os.listdir(pdf_folder)

for pdf in raw_pdfs:
    # Full path to the PDF file
    pdf_path = os.path.join(pdf_folder, pdf)
    
    # Load and extract text from the PDF
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        # Save processed text for RAG (Retrieval-Augmented Generation)
        txt_filename = os.path.join(pdf_folder, pdf[:-4] + '.txt')
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Processed and saved: {txt_filename}")
    except Exception as e:
        print(f"Error processing {pdf}: {e}")
