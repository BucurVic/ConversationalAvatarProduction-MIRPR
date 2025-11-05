import time
from loader import load_data
from chunk_splitter import chunk_function, EMBEDDING_MODEL_NAME
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Definim calea unde va fi salvată baza de date vectorială
DB_FAISS_PATH = 'vectorstore/'

def create_vector_store():
    """
    Funcție principală pentru a crea și salva baza de date vectorială.
    """
    print("Pasul 1: Încărcarea datelor din PDF...")
    knowledge_base = load_data()
    
    print("Pasul 2: Împărțirea documentelor în 'chunks'...")
    # Folosim chunk_size=256, așa cum ai definit în chunk_splitter.py
    docs_processed = chunk_function(256, knowledge_base)
    
    print(f"S-au procesat {len(docs_processed)} 'chunks' de documente.")

    print(f"Pasul 3: Inițializarea modelului de embedding: {EMBEDDING_MODEL_NAME}...")
    # Inițializează modelul de embedding-uri
    # 'model_kwargs' ne asigură că rulează pe CPU.
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )

    print("Pasul 4: Crearea bazei de date vectoriale FAISS...")
    start_time = time.time()
    
    # Aceasta este comanda care construiește indexul.
    # Trece prin toate 'docs_processed', calculează embedding-ul pentru fiecare
    # și le adaugă în indexul FAISS.
    db = FAISS.from_documents(docs_processed, embeddings)
    
    end_time = time.time()
    print(f"Timpul necesar pentru a crea embedding-urile și indexul: {end_time - start_time:.2f} secunde.")

    print(f"Pasul 5: Salvarea bazei de date în directorul '{DB_FAISS_PATH}'...")
    # Salvăm indexul pe disc pentru a-l putea refolosi ulterior
    db.save_local(DB_FAISS_PATH)
    
    print("Baza de date vectorială a fost creată și salvată cu succes.")

if __name__ == "__main__":
    create_vector_store()