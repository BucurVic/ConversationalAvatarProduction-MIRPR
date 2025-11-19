import time
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from unidecode import unidecode
from llama_cpp import Llama # <-- NOU: Biblioteca pentru rulare locală

# --- Configurare ---
DB_FAISS_PATH = 'vectorstore/'
EMBEDDING_MODEL_NAME = "thenlper/gte-small"

# Calea directă către modelul tău descărcat (schimbat de la LLM_MODEL la LLM_MODEL_PATH)
LLM_MODEL_PATH = "./models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

def load_db():
    """Încarcă baza de date vectorială FAISS."""
    print(f"[INFO] Se încarcă baza de date...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

# --- NOU: Funcție pentru a încărca LLM-ul local ---
def load_llm():
    """Încarcă modelul LLM (GGUF) folosind llama-cpp-python."""
    print(f"[INFO] Se încarcă modelul GPT-OSS-20B din: {LLM_MODEL_PATH}...")
    try:
        # n_gpu_layers=-1 folosește GPU-ul la maxim (Metal/CUDA)
        llm = Llama(
            model_path=LLM_MODEL_PATH,
            n_gpu_layers=-1, 
            n_ctx=4096,
            verbose=False
        )
        print("[SUCCESS] Modelul LLM a fost încărcat pe GPU.")
        return llm
    except Exception as e:
        print(f"[EROARE FATALĂ] Nu s-a putut încărca modelul GGUF: {e}")
        print("Asigură-te că fișierul există la calea specificată.")
        return None

def create_prompt(context_docs, query):
    """Creează promptul 'Tutore AI' structurat."""
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    prompt_template = f"""
Ești un Asistent Universitar AI expert, prietenos și răbdător.

INSTRUCȚIUNI STRICTE:
1. Răspunsul tău trebuie să fie bazat **EXCLUSIV** pe textul de la secțiunea "CONTEXT" de mai jos. 
2. Structurează răspunsul în două părți clare:
   a) **Definiția/Răspunsul direct:** Preia informația exactă și riguroasă din text.
   b) **Explicația simplă:** Reformulează pe scurt, "ca pentru studenți", ca să fie ușor de înțeles.
3. Dacă informația nu există în context, spune sincer: "Nu am găsit această informație în materialele de curs."
4. Răspunde în limba română.

CONTEXT DIN MANUAL:
---
{context}
---

ÎNTREBAREA STUDENTULUI:
{query}

RĂSPUNSUL TĂU (Structurat):
"""
    return prompt_template

# --- MODIFICAT: Funcția de Răspuns folosește instanța locală LLM ---
def get_llm_response(prompt, llm_instance):
    """Trimite promptul către instanța Llama (local) și preia răspunsul curat."""
    if llm_instance is None: return None
    
    try:
        start_time = time.time()
        
        # Rulare locală
        output = llm_instance.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        end_time = time.time()
        
        # Extragem textul brut
        raw_text = output['choices'][0]['message']['content']
        
        # --- FILTRARE PENTRU GPT-OSS-20B ---
        # Acest model pune răspunsul final după tag-ul: <|channel|>final<|message|>
        # Verificăm dacă există acest tag și tăiem tot ce e înainte de el.
        marker = "<|channel|>final<|message|>"
        
        if marker in raw_text:
            # Luăm ultima parte (după marker)
            clean_text = raw_text.split(marker)[-1]
        else:
            # Dacă nu găsește markerul, înseamnă că a răspuns direct (păstrăm tot)
            clean_text = raw_text
            
        # Curățăm eventuale tag-uri de final rămase
        clean_text = clean_text.replace("<|end|>", "").strip()
        # -----------------------------------
        
        print(f"[INFO] Generare finalizată în {end_time - start_time:.2f} secunde.")
        return clean_text

    except Exception as e:
        print(f"[EROARE] Eroare la generarea răspunsului: {e}")
        return None

# --- Funcția principală (Actualizată) ---
if __name__ == "__main__":
    # 1. Inițializare
    db = load_db()
    llm = load_llm() # <-- Încărcăm modelul local, nu clientul API
    
    if not llm:
        exit()
        
    print("\n" + "="*60)
    print("🎓 TUTORE AI ACTIVAT (Local GPT-OSS/Llama)")
    print(f" Model: {LLM_MODEL_PATH.split('/')[-1]}")
    print(" Scrie 'exit' pentru a ieși.")
    print("="*60 + "\n")

    # 2. Buclă interactivă
    while True:
        query_original = input("\nÎntrebarea ta: ")
        
        if query_original.lower() in ['exit', 'quit']:
            break
            
        # 3. Retrieval
        query_normalized = unidecode(query_original)
        context_docs = db.similarity_search(query_normalized, k=4)
        
        # Extragerea Surselor
        surse_gasite = set()
        for doc in context_docs:
            raw_source = doc.metadata.get('source', 'Manual')
            sursa_curata = " ".join(raw_source.split())
            surse_gasite.add(sursa_curata)

        # 4. Generare Răspuns
        prompt = create_prompt(context_docs, query_original)
        response = get_llm_response(prompt, llm) # <-- Trimitem la instanța LLM locală
        
        if response:
            print("\n" + "="*60)
            print("🎓 RĂSPUNS GENERAT:")
            print("-" * 60)
            print(response.strip())
            
            print("-" * 60)
            print("📚 SURSE BIBLIOGRAFICE:")
            sorted_sources = sorted(list(surse_gasite))
            
            for i, sursa in enumerate(sorted_sources):
                if i < 3:
                    print(f"   📍 {sursa}")
                else:
                    print(f"   ... (și altele)")
                    break
            print("="*60 + "\n")