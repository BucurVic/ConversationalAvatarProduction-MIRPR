import os
import time
import pickle
import faiss
import numpy as np
import subprocess
from pathlib import Path
from unidecode import unidecode
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, CrossEncoder 
import re
from rank_bm25 import BM25Okapi 

# --- Importuri Avatar ---
# Încercăm să importăm modulele de generare video. Dacă nu există, folosim funcții "mock" (pentru testare).
try:
    from scripts.config import (
        SADTALKER_REPO,
        USER_IMAGE,
        WAV2LIP_REPO
    )
    from generate_avatar.generate_tts import tts_piper
    from generate_avatar.generate_lip_sync_wav2lip import wav2lip_generate_video
except ImportError:
    print("[WARN] Nu s-au putut importa modulele de Avatar. Rulăm pe modul text-only/mock.")
    USER_IMAGE = "input_img.png"
    WAV2LIP_REPO = "external/Wav2Lip"
    def tts_piper(text, path): pass
    def wav2lip_generate_video(**kwargs): return "mock.mp4"

# --- CONFIGURARE SISTEM ---
DB_FOLDER_PATH = 'vectorstoretmp/' 
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# AICI PUI NUMELE EXACT AL MODELULUI TĂU FINE-TUNED (GGUF)
LLM_MODEL_PATH = "./models/Llama-3-Profesor-Geometrie.gguf" 
# Dacă nu ai apucat să muți fișierul, poți lăsa modelul vechi:
# LLM_MODEL_PATH = "./models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

# Clasă simplă pentru a stoca documentele găsite (compatibilitate cu structura veche)
class SimpleDoc:
    def __init__(self, content, source="Manual", score=0.0):
        self.page_content = content
        self.metadata = {'source': source, 'score': score}

# ----------------------------------------------------------
#            1. ÎNCĂRCARE RESURSE (Optimizare)
# ----------------------------------------------------------
def load_resources():
    print(f"\n[INIT] Se inițializează sistemul...")
    
    # 1. FAISS (Vector Database)
    try:
        index = faiss.read_index(f"{DB_FOLDER_PATH}/index.faiss")
        with open(f"{DB_FOLDER_PATH}/index.pkl", "rb") as f:
            texts = pickle.load(f)
        print("[OK] Baza de date vectorială încărcată.")
    except Exception as e:
        print(f"[EROARE CRITICĂ] Nu am putut încărca vector store-ul: {e}")
        return None, None, None, None, None, None

    # 2. Embedding Model (pentru Vector Search) - Pe GPU
    print(f"[INIT] Se încarcă modelul de embedding...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cuda")

    # 3. Reranker Model (pentru filtrare fină) - Pe CPU (economie VRAM)
    print(f"[INIT] Se încarcă modelul de reranking...")
    reranker_model = CrossEncoder(RERANKER_MODEL_NAME, device="cpu")
    
    # 4. BM25 Index (pentru Hybrid Search - Cuvinte Cheie)
    print(f"[INIT] Se construiește indexul lexical BM25...")
    tokenized_corpus = [doc.split(" ") for doc in texts]
    bm25_index = BM25Okapi(tokenized_corpus)

    # 5. LLM (Generare Text) - Pe GPU
    print(f"[INIT] Se încarcă Llama 3...")
    try:
        llm = Llama(
            model_path=LLM_MODEL_PATH,
            n_gpu_layers=-1, # Totul pe GPU
            n_ctx=4096,      # Context window
            verbose=False
        )
        print("[OK] Llama 3 încărcat.")
    except Exception as e:
        print(f"[EROARE CRITICĂ] Llama model lipsă sau incompatibil: {e}")
        return None, None, None, None, None, None

    return index, texts, embed_model, reranker_model, bm25_index, llm

# ----------------------------------------------------------
#            2. QUERY EXPANSION (Multi-Query)
# ----------------------------------------------------------
def generate_multi_queries(original_query, llm):
    """
    Folosește LLM-ul pentru a genera 3 variante alternative ale întrebării,
    pentru a acoperi ambiguitățile.
    """
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Generează 3 variante scurte de căutare pentru întrebarea utilizatorului (definiții, concepte conexe).
Doar textul, separat prin linie nouă.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Întrebare: {original_query}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    try:
        output = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=60
        )
        content = output['choices'][0]['message']['content'].strip()
        queries = [q.strip() for q in content.split('\n') if q.strip()]
        
        # Ne asigurăm că originalul e inclus (și e ultimul, prin convenție)
        if original_query not in queries:
            queries.append(original_query)
        return queries[:4] 
    except:
        return [original_query]

# ----------------------------------------------------------
#            3. HYBRID SEARCH (Vector + BM25 + RRF)
# ----------------------------------------------------------
def hybrid_search_expanded(queries_list, index, texts, embed_model, reranker_model, bm25_index, final_k=5):
    """
    Execută căutarea pentru toate variantele de întrebare și combină rezultatele
    folosind algoritmul Reciprocal Rank Fusion (RRF).
    """
    rrf_k = 60 # Constantă de regularizare RRF
    merged_scores = {} # {text_document: scor_cumulat}

    # Pas A: Culegem candidați pentru fiecare variantă de query
    for query in queries_list:
        initial_k = 10 
        
        # 1. Vector Search (Semantic)
        query_vec = embed_model.encode([query]).astype("float32")
        distances, indices = index.search(query_vec, initial_k)
        
        for rank, idx in enumerate(indices[0]):
            if 0 <= idx < len(texts):
                doc = texts[idx]
                score = 1 / (rrf_k + rank)
                merged_scores[doc] = merged_scores.get(doc, 0) + score

        # 2. BM25 Search (Lexical / Cuvinte cheie)
        query_tokens = query.split(" ")
        bm25_docs = bm25_index.get_top_n(query_tokens, texts, n=initial_k)
        
        for rank, doc in enumerate(bm25_docs):
            score = 1 / (rrf_k + rank)
            merged_scores[doc] = merged_scores.get(doc, 0) + score

    # Pas B: Sortăm candidații unici după scorul RRF
    sorted_candidates = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Păstrăm top 20 cei mai buni candidați pentru verificare finală
    candidates_text = [item[0] for item in sorted_candidates[:20]]

    if not candidates_text:
        return []

    # Pas C: Reranking Final cu Cross-Encoder
    # Folosim query-ul original (care e ultimul în listă) pentru a verifica relevanța reală
    original_query = queries_list[-1]
    
    pairs = [[original_query, doc_text] for doc_text in candidates_text]
    scores = reranker_model.predict(pairs)
    
    ranked_final = sorted(zip(candidates_text, scores), key=lambda x: x[1], reverse=True)
    
    # Construim obiectele finale
    final_docs = []
    for doc_text, score in ranked_final[:final_k]:
        source = "Manual"
        # Încercare rudimentară de a extrage titlul/sursa din text, dacă există
        if doc_text.startswith("Title:"):
            parts = doc_text.split(":", 2)
            if len(parts) > 1: source = parts[1].strip()
            
        final_docs.append(SimpleDoc(doc_text, source, score))
            
    return final_docs

# ----------------------------------------------------------
#            4. GENERARE RAG (Prompt & Inference)
# ----------------------------------------------------------
def create_prompt(context_docs, query):
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    # Prompt optimizat pentru stilul "Profesor"
    prompt_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Ești un Profesor de Geometrie prietenos și clar.
Folosește contextul de mai jos pentru a răspunde studentului. 
Dacă informația nu este în context, spune că nu știi.
Nu folosi liste cu bullet points (*), vorbește cursiv.
<|eot_id|><|start_header_id|>user<|end_header_id|>
CONTEXT DIN MANUAL:
{context}

ÎNTREBAREA STUDENTULUI:
{query}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    return prompt_template

def get_llm_response(prompt, llm):
    try:
        output = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, # Temperatură mică pentru precizie factuală
        )
        return output['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"[EROARE GENERARE] {e}")
        return None

# ----------------------------------------------------------
#            5. GENERARE AVATAR
# ----------------------------------------------------------
def generate_avatar_video_wav2lip(answer_text: str):
    # Curățăm textul de simboluri care strică sunetul
    text_for_tts = answer_text.replace("*", "").replace("#", "").replace("_", "")
    
    base_dir = Path(__file__).resolve().parent
    work_dir = base_dir / "runtime" / "avatar_wav2lip"
    work_dir.mkdir(parents=True, exist_ok=True)
    audio_path = work_dir / "answer.wav"
    video_output_dir = work_dir / "video"
    video_output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[VIDEO] Generare Audio (TTS)...")
    try:
        tts_piper(text_for_tts, str(audio_path))
    except Exception as e:
        print(f"[EROARE TTS] {e}")
        return None

    print("[VIDEO] Generare LipSync (Wav2Lip)...")
    try:
        wav2lip_generate_video(
            image_path=str(base_dir / USER_IMAGE),
            audio_path=str(audio_path),
            output_dir=str(video_output_dir),
            wav2lip_repo=str(base_dir / WAV2LIP_REPO) 
        )
    except Exception as e:
        print(f"[EROARE VIDEO] {e}")
        return None

    # Găsim cel mai recent video creat
    mp4_files = list(video_output_dir.glob("*.mp4"))
    if mp4_files:
        final_video = max(mp4_files, key=lambda p: p.stat().st_mtime)
        print(f"[SUCCESS] Video generat: {final_video}")
        # Deschide automat video-ul (merge pe Linux/WSL cu GUI support)
        try:
            subprocess.run(["xdg-open", str(final_video)], stderr=subprocess.DEVNULL)
        except:
            pass
        return final_video
    return None

# ----------------------------------------------------------
#                       MAIN LOOP
# ----------------------------------------------------------
if __name__ == "__main__":
    # 1. Încărcare
    index, texts, embed_model, reranker_model, bm25_index, llm = load_resources()
    
    if not llm:
        print("[EXIT] Sistemul nu a putut porni.")
        exit()
        
    print("\n" + "="*60)
    print("🎓 PROFESOR AI - GEOMETRIE VECTORIALĂ")
    print("   (Optimizări: Hybrid Search, Reranking, Multi-Query, Fine-Tuned)")
    print("   Scrie 'exit' pentru a ieși.")
    print("="*60 + "\n")

    while True:
        query_original = input("\nÎntrebarea ta: ")
        if query_original.lower() in ['exit', 'quit']:
            break
        
        t0 = time.time()

        # 2. Query Expansion
        print("[AI] Analizez întrebarea...")
        expanded_queries = generate_multi_queries(query_original, llm)
        # Afișăm variantele generate (fără original) pentru debug
        print(f"     -> Variante: {expanded_queries[:-1]}") 
        
        # 3. Retrieval (Hybrid + Rerank)
        print("[SEARCH] Caut în manuale (Vector + Keywords)...")
        context_docs = hybrid_search_expanded(
            expanded_queries, index, texts, embed_model, reranker_model, bm25_index, final_k=5
        )
        
        if not context_docs:
            print("[INFO] Nu am găsit informații relevante în cursuri.")
            continue
            
        # Afișăm sursele găsite
        print(f"     -> Am găsit {len(context_docs)} fragmente relevante (Scor max: {context_docs[0].metadata['score']:.2f})")

        # 4. Generare Text
        print("[GENERATE] Profesorul formulează răspunsul...")
        prompt = create_prompt(context_docs, query_original)
        response_text = get_llm_response(prompt, llm)
        
        if response_text:
            t1 = time.time()
            print("\n" + "="*60)
            print("🎓 RĂSPUNS:")
            print("-" * 60)
            print(response_text)
            print("-" * 60)
            print(f"⏱️ Timp răspuns: {t1-t0:.2f} secunde")
            print("="*60 + "\n")
            
            generate_avatar_video_wav2lip(response_text)