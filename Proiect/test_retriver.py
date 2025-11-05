from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from unidecode import unidecode # <--- ADAUGĂ ASTA

DB_FAISS_PATH = 'vectorstore/'
EMBEDDING_MODEL_NAME = "thenlper/gte-small"

def test_retrieval():
    """
    Încarcă baza de date FAISS și efectuează o căutare semantică de test.
    """
    print(f"Încărcarea modelului de embedding: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    
    print(f"Încărcarea bazei de date vectoriale din '{DB_FAISS_PATH}'...")
    db = FAISS.load_local(
        DB_FAISS_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True 
    )
    
    print("Baza de date a fost încărcată cu succes.")
    
    # 3. Definește o întrebare de test (cu diacritice, e ok)
    query_original = "Ce este un spațiu vectorial?"
    
    # 4. Normalizează întrebarea!
    query_normalized = unidecode(query_original)
    
    print(f"\nÎntrebare originală: '{query_original}'")
    print(f"Întrebare normalizată (trimisă la căutare): '{query_normalized}'")
    
    # 5. Efectuează căutarea semantică
    print(f"\nSe caută cele mai relevante K=3 documente...")
    
    # Folosim query-ul normalizat pentru căutare
    results = db.similarity_search(query_normalized, k=3)
    
    print("\n--- REZULTATE GĂSITE (CONTEXT) ---")
    for i, doc in enumerate(results):
        print(f"\n--- Rezultatul #{i+1} (Sursa: {doc.metadata.get('source', 'CAPITOL N/A')}) ---")
        # Acum și textul din 'doc' ar trebui să fie curat, fără diacritice stricate
        print(doc.page_content)
        print("---------------------------------" + "-" * len(str(i+1)))

if __name__ == "__main__":
    test_retrieval()