import os
import fitz  # pymupdf
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def ingest_pdf(pdf_path, user_id=None):
    print(f"Extracting text from {pdf_path}...")
    text = extract_text(pdf_path)
    
    print("Chunking text...")
    chunks = chunk_text(text)
    print(f"Total chunks: {len(chunks)}")
    
    print("Embedding and storing chunks...")
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()
        supabase.table("documents").insert({
            "user_id": user_id,
            "content": chunk,
            "embedding": embedding,
            "metadata": {"source": pdf_path, "chunk_index": i}
        }).execute()
        print(f"Stored chunk {i+1}/{len(chunks)}")
    
    print("Done.")

if __name__ == "__main__":
    ingest_pdf("test.pdf")