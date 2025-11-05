import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from unidecode import unidecode # <-- Adăugat pentru normalizare

# --- Configurare ---
DB_FAISS_PATH = 'vectorstore/'
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
LM_STUDIO_URL = "http://localhost:1234/v1"

# !!! IMPORTANT: Numele modelului tău v0.3 !!!
# Am pus numele pe care l-ai menționat. Verifică dacă e exact așa.
LLM_MODEL = "mistral-7b-instruct-v0.3.Q4_K_M.gguf"

def load_db():
    """Încarcă baza de date vectorială FAISS."""
    print(f"Încărcarea modelului de embedding: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    
    print(f"Încărcarea bazei de date vectoriale din '{DB_FAISS_PATH}'...")
    # S-a adăugat 'allow_dangerous_deserialization=True'
    db = FAISS.load_local(
        DB_FAISS_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True 
    )
    print("Baza de date a fost încărcată cu succes.")
    return db

def create_prompt(context_docs, query):
    """Creează promptul final pentru LLM."""
    
    # Extragem conținutul paginilor (care este deja curățat, fără diacritice)
    context = "\n\n---\n\n".join([doc.page_content for doc in context_docs])
    
    # Șablonul de prompt
    prompt_template = f"""
Ești un asistent AI specializat în cursul de geometrie. 
Răspunde la următoarea întrebare bazându-te **exclusiv** pe contextul oferit mai jos. 
Textul contextului este în limba română, dar fără diacritice.
Răspunsul tău trebuie să fie în limba română (poți folosi diacritice în răspunsul tău).
Dacă răspunsul nu se află în context, spune "Informatia nu a fost gasita in materialele de curs."

CONTEXT:
---
{context}
---

ÎNTREBARE:
{query}

RĂSPUNS:
"""
    return prompt_template

def get_llm_response(prompt):
    """Trimite promptul către LM Studio și preia răspunsul."""
    print("\n[INFO] Conectare la LM Studio...")
    
    client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio") # api_key nu este validat

    try:
        start_time = time.time()
        completion = client.chat.completions.create(
            model=LLM_MODEL, 
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3, # Setăm o temperatură joasă pentru răspunsuri fidele
        )
        end_time = time.time()
        
        print(f"[INFO] Răspuns generat în {end_time - start_time:.2f} secunde.")
        return completion.choices[0].message.content
        
    except Exception as e:
        print(f"\n[EROARE] Nu s-a putut conecta la LM Studio.")
        print("Verifică următoarele:")
        print(f"1. Serverul LM Studio este pornit la {LM_STUDIO_URL}?")
        print(f"2. Numele modelului '{LLM_MODEL}' este corect și modelul este încărcat complet?")
        print(f"Detalii eroare: {e}")
        return None

# --- Funcția principală ---
if __name__ == "__main__":
    # 1. Încarcă baza de date
    db = load_db()
    
    # 2. Definește întrebarea (cu diacritice, e OK)
    query_original = "Ce este un spațiu vectorial?"
    
    # 3. Normalizează întrebarea pentru căutare (MODIFICARE)
    query_normalized = unidecode(query_original)
    
    print(f"\nÎntrebare originală: '{query_original}'")
    print(f"Întrebare normalizată (pt. căutare): '{query_normalized}'")

    # 4. Retrieval (Regăsire) - folosim întrebarea normalizată
    context_docs = db.similarity_search(query_normalized, k=4)
    
    # 5. Prompting (Creare Prompt) - folosim întrebarea originală
    prompt = create_prompt(context_docs, query_original)
    
    # 6. Generation (Generare)
    response = get_llm_response(prompt)
    
    if response:
        print("\n--- RĂSPUNSUL AVATARULUI ---")
        print(response)