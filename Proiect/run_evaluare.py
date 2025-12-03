import json
import time
import torch
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  
from unidecode import unidecode
from llama_cpp import Llama # <--- ADĂUGAT: Rulare locală
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rouge_score import rouge_scorer 
from scipy.spatial.distance import cosine 
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction # <--- ADĂUGAT: BLEU

# --- Configurare ---
# DB_FAISS_PATH = 'vectorstore/'
DB_FAISS_PATH = 'vectorstoretmp/'
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
# Calea către modelul Llama 3 (Verifică să fie corectă)
LLM_MODEL_PATH = "./models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
EVAL_FILE_PATH = "evaluare.json" 


# def load_db():
#     """Încarcă baza de date vectorială FAISS."""
#     print(f"[INFO] Încărcarea modelului de embedding: {EMBEDDING_MODEL_NAME}...")
#     embeddings = HuggingFaceEmbeddings(
#         model_name=EMBEDDING_MODEL_NAME,
#         model_kwargs={'device': 'cuda'}
#     )
#     print(f"[INFO] Încărcarea bazei de date vectoriale din '{DB_FAISS_PATH}'...")
#     db = FAISS.load_local(
#         DB_FAISS_PATH, 
#         embeddings, 
#         allow_dangerous_deserialization=True 
#     )
#     return db

def load_db():
    print("Loading FAISS index…")

    # Load FAISS index
    index = faiss.read_index(DB_FAISS_PATH+"index.faiss")

    # Load metadata (texts/documents)
    with open(DB_FAISS_PATH+"index.pkl", "rb") as f:
        metadata = pickle.load(f)

    return index, metadata

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

def embed(texts):
    return embedding_model.encode(
        texts,
        convert_to_numpy=True,
        batch_size=64,
        show_progress_bar=False
    )

def load_llm():
    """Încarcă modelul Llama 3 local pe GPU."""
    print(f"[INFO] Se încarcă modelul Llama 3 din: {LLM_MODEL_PATH}...")
    print("CUDA Available: ", torch.cuda.is_available())
    try:
        llm = Llama(
            model_path=LLM_MODEL_PATH,
            n_gpu_layers=-1, # Totul pe GPU
            n_ctx=4096,
            n_threads=8,
            n_batch=512,
            use_mlock=False,
            use_mmap=True,
            verbose=False
        )
        print("[SUCCESS] Modelul a fost încărcat.")
        return llm
    except Exception as e:
        print(f"[EROARE] Nu s-a putut încărca modelul: {e}")
        return None

def create_prompt(context_docs, query):
    """Creează promptul (același stil ca în run_rag.py)."""
    # context = "\n\n---\n\n".join([doc.page_content for doc in context_docs])
    context = "\n\n---\n\n".join(context_docs)
    prompt_template = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Ești un Asistent Universitar AI expert.
Răspunde la întrebare folosind EXCLUSIV contextul de mai jos.
Răspunde în limba română.
<|eot_id|><|start_header_id|>user<|end_header_id|>
CONTEXT:
{context}

ÎNTREBARE: {query}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    return prompt_template

# def get_llm_response(prompt, llm_instance):
#     """Trimite promptul către modelul local."""
#     if llm_instance is None: return ""
#     try:
#         output = llm_instance.create_chat_completion(
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.1, # Temperatură mică pentru consistență la evaluare
#             max_tokens=1024
#         )
#         return output['choices'][0]['message']['content']
#     except Exception as e:
#         print(f"\n[EROARE] Generare eșuată: {e}")
#         return ""

def get_llm_response(prompt, llm_instance):
    """Send prompt to local Llama model (GPU optimized)."""
    if llm_instance is None:
        return ""

    try:
        output = llm_instance(
            prompt,
            max_tokens=512,
            temperature=0.2,
            top_p=0.9,
            stop=["<|eot_id|>"]
        )

        # llama.cpp uses "choices"[0]["text"]
        return output["choices"][0]["text"].strip()

    except Exception as e:
        print(f"\n[EROARE] Generare eșuată: {e}")
        return ""


def load_evaluation_set(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[EROARE] Nu am putut citi fișierul {file_path}: {e}")
        return None

def calculate_metrics(generated_text, expected_text, embedding_model):
    """Calculează ROUGE, Semantic Similarity și BLEU."""
    
    # 1. Metrica ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(expected_text, generated_text)
    rouge_l_f1 = rouge_scores['rougeL'].fmeasure
    
    # 2. Metrica Similaritate Semantică
    expected_embedding = embedding_model.embed_query(expected_text)
    generated_embedding = embedding_model.embed_query(generated_text)
    semantic_similarity = 1 - cosine(expected_embedding, generated_embedding)
    
    # 3. Metrica BLEU (NOU)
    # BLEU compară n-grame (cuvinte). Are nevoie de textul spart în cuvinte (tokens).
    ref_tokens = expected_text.split()
    cand_tokens = generated_text.split()
    
    # SmoothingFunction e necesară pentru texte scurte, ca să nu dea 0 dacă nu găsește 4 cuvinte la fel la rând.
    chencherry = SmoothingFunction()
    bleu_score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=chencherry.method1)

    return {
        "rouge_l_f1": rouge_l_f1,
        "semantic_similarity": semantic_similarity,
        "bleu_score": bleu_score
    }

# --- Funcția Principală ---
if __name__ == "__main__":
    print("--- START EVALUARE SISTEM RAG (Llama 3 Local) ---")
    
    # 1. Încărcare Resurse
    # db = load_db()
    index, metadata = load_db()
    print("FAISS index size:", index.ntotal)
    print("Metadata size:", len(metadata))

    # Avem nevoie de modelul de embedding și separat pentru calculul metricilor
    metric_embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'}
    )
    
    llm = load_llm()
    if not llm: exit()

    # 2. Încărcare Date
    eval_set = load_evaluation_set(EVAL_FILE_PATH)
    if not eval_set: exit()

    # Variabile pentru medii
    total_rouge_l = 0
    total_semantic_similarity = 0
    total_bleu = 0
    results = []

    print(f"\n[INFO] Se începe evaluarea pe {len(eval_set)} exemple...")
    
    # for item in tqdm(eval_set, desc="Procesare întrebări"):
    #     query_original = item["intrebare"]
    #     raspuns_asteptat_norm = item["raspuns_asteptat"] 

    #     # Procesare RAG
    #     query_normalized = unidecode(query_original)
    #     # context_docs = db.similarity_search(query_normalized, k=4)

    #     # 1. Embed query using the GPU embedder
    #     query_vec = embed([query_normalized])[0]      # (768-dim vector on GPU)
    #     query_vec = np.array(query_vec, dtype="float32")

    #     # 2. Search FAISS for top-k matches
    #     D, I = index.search(query_vec.reshape(1, -1), k=4)

    #     # 3. Retrieve the corresponding documents from metadata
    #     context_docs = [metadata[i] for i in I[0]]

    #     # 4. Build prompt using retrieved docs
    #     prompt = create_prompt(context_docs, query_original)

        
    #     # Generare
    #     raspuns_generat_raw = get_llm_response(prompt, llm)
        
    #     # Normalizare răspuns pentru comparație corectă
    #     raspuns_generat_norm = unidecode(raspuns_generat_raw)

    #     # Calcul Metrici (inclusiv BLEU)
    #     metrics = calculate_metrics(raspuns_generat_norm, raspuns_asteptat_norm, metric_embedding_model)
        
    #     total_rouge_l += metrics["rouge_l_f1"]
    #     total_semantic_similarity += metrics["semantic_similarity"]
    #     total_bleu += metrics["bleu_score"]
        
    #     results.append({
    #         "intrebare": query_original,
    #         "raspuns_asteptat": raspuns_asteptat_norm,
    #         "raspuns_generat": raspuns_generat_raw,
    #         "metrics": metrics
    #     })
    # =============================================
    # STEP 3A – Batch embed all questions (GPU)
    # =============================================
    all_queries_original = [item["intrebare"] for item in eval_set]
    all_queries_normalized = [unidecode(q) for q in all_queries_original]

    print("Embedding all queries on GPU...")
    query_vectors = embed(all_queries_normalized).astype("float32")

    # =============================================
    # STEP 3B – Batch FAISS search
    # =============================================
    print("Searching FAISS index for all queries...")
    D, I = index.search(query_vectors, k=4)

    # =============================================
    # STEP 3C – Build prompts in a single vectorized pass
    # =============================================
    print("Building prompts...")
    prompts = []
    for idx, item in enumerate(eval_set):
        context_docs = [metadata[j] for j in I[idx]]
        prompt = create_prompt(context_docs, all_queries_original[idx])
        prompts.append(prompt)

    # =============================================
    # STEP 3D – LLM responses (this remains a loop)
    # =============================================
    print("Generating responses with LLM...")
    responses_raw = []
    responses_norm = []

    for prompt in tqdm(prompts):
        raw = get_llm_response(prompt, llm)
        responses_raw.append(raw)
        responses_norm.append(unidecode(raw))

    # =============================================
    # STEP 3E – Metrics per item
    # =============================================
    print("Computing metrics...")
    results = []
    total_rouge_l = 0
    total_semantic_similarity = 0
    total_bleu = 0

    for idx, item in enumerate(eval_set):
        expected = item["raspuns_asteptat"]
        generated_norm = responses_norm[idx]

        metrics = calculate_metrics(generated_norm, expected, metric_embedding_model)

        total_rouge_l += metrics["rouge_l_f1"]
        total_semantic_similarity += metrics["semantic_similarity"]
        total_bleu += metrics["bleu_score"]

        results.append({
            "intrebare": item["intrebare"],
            "raspuns_asteptat": expected,
            "raspuns_generat": responses_raw[idx],
            "metrics": metrics
        })


    # Afișare Rezultate Finale
    print("\n\n" + "="*40)
    print("       RAPORT EVALUARE FINAL")
    print("="*40)
    
    num_items = len(eval_set)
    if num_items > 0:
        avg_rouge = (total_rouge_l / num_items) * 100
        avg_sem = (total_semantic_similarity / num_items) * 100
        avg_bleu = (total_bleu / num_items) * 100
        
        print(f"Model: Llama 3 8B (Local)")
        print(f"Exemple: {num_items}")
        print("-" * 40)
        print(f"✅ Similaritate Semantică: {avg_sem:.2f}%")
        print(f"📝 ROUGE-L (Suprapunere):  {avg_rouge:.2f}%")
        print(f"🔵 BLEU Score (Precizie):  {avg_bleu:.2f}%")
        print("-" * 40)
        
        print("\n--- EXEMPLE DETALIATE ---")
        for i, res in enumerate(results[:3]):
            print(f"\nExemplul #{i+1}")
            print(f"  Î: {res['intrebare']}")
            print(f"  R (Așteptat): {res['raspuns_asteptat'][:100]}...")
            print(f"  R (Generat):  {res['raspuns_generat'][:100]}...")
            print(f"  Scoruri -> Sem: {res['metrics']['semantic_similarity']:.2f}, ROUGE: {res['metrics']['rouge_l_f1']:.2f}, BLEU: {res['metrics']['bleu_score']:.2f}")