import json
import time
from tqdm import tqdm  
from unidecode import unidecode
from openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rouge_score import rouge_scorer 
from scipy.spatial.distance import cosine 

DB_FAISS_PATH = 'vectorstore/'
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
LM_STUDIO_URL = "http://localhost:1234/v1"
LLM_MODEL = "mistral-7b-instruct-v0.3.Q4_K_M.gguf" 
EVAL_FILE_PATH = "evaluare.json" 


def load_db():
    """Încarcă baza de date vectorială FAISS."""
    print(f"[INFO] Încărcarea modelului de embedding: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    print(f"[INFO] Încărcarea bazei de date vectoriale din '{DB_FAISS_PATH}'...")
    db = FAISS.load_local(
        DB_FAISS_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True 
    )
    print("[INFO] Baza de date a fost încărcată cu succes.")
    return db

def create_prompt(context_docs, query):
    """Creează promptul final pentru LLM."""
    context = "\n\n---\n\n".join([doc.page_content for doc in context_docs])
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

def get_llm_response(prompt, client):
    """Trimite promptul către LM Studio. Reutilizează clientul."""
    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL, 
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"\n[EROARE] Eroare la interogarea LM Studio: {e}")
        return None

# --- Funcții Noi pentru Evaluare ---

def load_evaluation_set(file_path):
    """Încarcă setul de date de evaluare din JSON."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[EROARE] Nu am putut citi fișierul {file_path}: {e}")
        return None

def calculate_metrics(generated_text, expected_text, embedding_model):
    """Calculează ROUGE-L și Similaritatea Semantică."""
    
    # 1. Metrica ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(expected_text, generated_text)
    rouge_l_f1 = rouge_scores['rougeL'].fmeasure
    
    # 2. Metrica Similaritate Semantică
    # Creăm embedding-urile pentru ambele texte
    expected_embedding = embedding_model.embed_query(expected_text)
    generated_embedding = embedding_model.embed_query(generated_text)
    
    # Calculăm similaritatea cosinus (1 - distanța cosinus)
    # Rezultatul va fi între -1 și 1 (dar de obicei 0 și 1 pentru text)
    semantic_similarity = 1 - cosine(expected_embedding, generated_embedding)
    
    return {
        "rouge_l_f1": rouge_l_f1,
        "semantic_similarity": semantic_similarity
    }

# --- Funcția Principală de Evaluare ---
if __name__ == "__main__":
    print("--- START EVALUARE SISTEM RAG ---")
    
    # 1. Încarcă Baza de Date Vectorială
    db = load_db()
    
    # 2. Încarcă Modelul de Embedding (pentru calculul metricii de similaritate)
    print("[INFO] Încărcarea modelului de embedding pentru metrici...")
    # Folosim același model pentru a fi consistenți
    metric_embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    
    # 3. Încarcă Setul de Date de Evaluare
    print(f"[INFO] Încărcarea setului de evaluare din {EVAL_FILE_PATH}...")
    eval_set = load_evaluation_set(EVAL_FILE_PATH)
    if not eval_set:
        print("[EROARE] Nu se poate continua fără setul de evaluare. Opreste scriptul.")
        exit()
        
    
    print("[INFO] Inițializare client LM Studio...")
    client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

    total_rouge_l = 0
    total_semantic_similarity = 0
    results = []

    print(f"\n[INFO] Se începe evaluarea pe {len(eval_set)} exemple...")
    
    for item in tqdm(eval_set, desc="Procesare întrebări"):
        query_original = item["intrebare"]
        raspuns_asteptat_norm = item["raspuns_asteptat"] 

        # 5.1. Normalizează întrebarea pentru căutare
        query_normalized = unidecode(query_original)
        
        # 5.2. Retrieval (Căutare Context)
        context_docs = db.similarity_search(query_normalized, k=4)
        
        # 5.3. Prompting
        prompt = create_prompt(context_docs, query_original)
        
        # 5.4. Generation (Generare Răspuns)
        raspuns_generat_raw = get_llm_response(prompt, client)
        
        if raspuns_generat_raw is None:
            raspuns_generat_raw = "" # Marchează ca eșec

        # 5.5. Normalizare Răspuns Generat
        
        raspuns_generat_norm = unidecode(raspuns_generat_raw)

        # 5.6. Calcul Metrici
        metrics = calculate_metrics(raspuns_generat_norm, raspuns_asteptat_norm, metric_embedding_model)
        
        total_rouge_l += metrics["rouge_l_f1"]
        total_semantic_similarity += metrics["semantic_similarity"]
        
        results.append({
            "intrebare": query_original,
            "raspuns_asteptat": raspuns_asteptat_norm,
            "raspuns_generat": raspuns_generat_raw, # Salvăm răspunsul brut pentru a-l citi
            "metrics": metrics
        })

    # 6. Afișare Rezultate
    print("\n\n--- EVALUARE COMPLETATĂ ---")
    
    num_items = len(eval_set)
    if num_items > 0:
        avg_rouge_l = (total_rouge_l / num_items) * 100
        avg_semantic_similarity = (total_semantic_similarity / num_items) * 100
        
        print(f"\n--- REZULTATE MEDII ({num_items} exemple) ---")
        print(f"**ROUGE-L (F1 Score Mediu): {avg_rouge_l:.2f}%**")
        print(f"**Similaritate Semantică (Medie): {avg_semantic_similarity:.2f}%**")
        
        print("\n--- EXEMPLE (Corect / Greșit) ---")
        
        for i, res in enumerate(results[:3]):
            print(f"\nExemplul #{i+1}")
            print(f"  Întrebare: {res['intrebare']}")
            print(f"  Răspuns Așteptat (norm): {res['raspuns_asteptat']}")
            print(f"  Răspuns Generat (brut): {res['raspuns_generat']}")
            print(f"  Metrici: ROUGE-L: {res['metrics']['rouge_l_f1']:.2f}, Similaritate: {res['metrics']['semantic_similarity']:.2f}")