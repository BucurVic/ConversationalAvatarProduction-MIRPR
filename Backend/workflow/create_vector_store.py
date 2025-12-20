import os
os.environ["OMP_NUM_THREADS"] = "1"
import time
import faiss
import torch
import pickle
from sentence_transformers import SentenceTransformer
from loader import load_data
from chunk_splitter import chunk_function, EMBEDDING_MODEL_NAME

# DB_FAISS_PATH = 'vectorstore/'
DB_FAISS_PATH = 'vectorstoretmp/'

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def create_vector_store():

    print("Pasul 1: Încărcarea datelor din PDF...")
    knowledge_base = load_data()
    
    # Verificăm dacă avem date
    if not knowledge_base:
        print("EROARE: Nu s-au încărcat documente. Verifică calea fișierului PDF.")
        return

    print("Pasul 2: Împărțirea documentelor în 'chunks'...")
    docs_processed = chunk_function(256, knowledge_base)
    
    print(f"S-au procesat {len(docs_processed)} 'chunks' de documente.")

    texts = [doc.page_content for doc in docs_processed]

    print(f"Pasul 3: Inițializarea modelului de embedding: {EMBEDDING_MODEL_NAME}...")
    
    device = get_device()
    print(f"--> Modelul va rula pe: {device.upper()}")
    
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device=device
    )

    def embed(texts):
        # Generăm embedding-urile
        embeddings = embedding_model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True 
        ).astype("float32")
        return embeddings

    print("Pasul 4: Crearea bazei de date vectoriale FAISS...")
    start_time = time.time()
    
    # Generăm embedding-urile
    embeddings = embed(texts)
    
    end_time = time.time()
    print(f"Timpul necesar pentru a crea embedding-urile: {end_time - start_time:.2f} secunde.")

    print(f"Pasul 5: Salvarea bazei de date în directorul '{DB_FAISS_PATH}'...")
    
    dimension = embeddings.shape[1]
    
    # Folosim Inner Product (IP) pentru Cosine Similarity
    index = faiss.IndexFlatIP(dimension) 
    
    index.add(embeddings)

    os.makedirs(DB_FAISS_PATH, exist_ok=True)

    faiss.write_index(index, f"{DB_FAISS_PATH}/index.faiss")

    with open(f"{DB_FAISS_PATH}/index.pkl", "wb") as f:
        pickle.dump(texts, f)
    
    print("Baza de date vectorială a fost creată și salvată cu succes.")

if __name__ == "__main__":
    create_vector_store()