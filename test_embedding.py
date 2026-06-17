import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv()

# Init
model = SentenceTransformer('all-MiniLM-L6-v2')
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Test text
text = "FastAPI is a modern web framework for building APIs with Python."

# Generate embedding
embedding = model.encode(text).tolist()
print(f"Embedding generated. Dimensions: {len(embedding)}")

# Insert into Supabase
result = supabase.table("documents").insert({
    "content": text,
    "embedding": embedding,
    "metadata": {"source": "test"}
}).execute()

print("Inserted into Supabase:", result.data)