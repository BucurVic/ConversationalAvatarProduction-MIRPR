import os
import time
import pickle
import faiss
import numpy as np
import shutil
import re
from pathlib import Path
from unidecode import unidecode
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, CrossEncoder 

# --- IMPORTURI API ---
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# --- Importuri Avatar (Păstrate) ---
try:
    from scripts.config import (
        SADTALKER_REPO,
        USER_IMAGE,
        WAV2LIP_REPO
    )
    from generate_avatar.generate_tts import tts_piper
    from generate_avatar.generate_lip_sync_wav2lip import wav2lip_generate_video
except ImportError:
    print("[WARN] Nu s-au putut importa modulele de Avatar. Rulăm pe mock.")
    USER_IMAGE = "input_img.png"
    WAV2LIP_REPO = "external/Wav2Lip"
    def tts_piper(text, path): print(f"[MOCK TTS] Se generează audio la {path}...")
    def wav2lip_generate_video(**kwargs): print("[MOCK VIDEO] Se generează video..."); return "mock.mp4"

# --- CONFIGURARE ---
DB_FOLDER_PATH = 'vectorstoretmp/' 
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL_PATH = "./models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

# Configurare directoare pentru API
BASE_DIR = Path(__file__).resolve().parent
VIDEO_OUTPUT_DIR = BASE_DIR / "runtime" / "avatar_wav2lip" / "video"
VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- INIȚIALIZARE APP FASTAPI ---
app = FastAPI(title="Face2Learn API", description="Backend RAG + Avatar Video")

# Configurare CORS (Permite frontend-ului să facă cereri)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permite orice origine (pt development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servim folderul cu videoclipuri static
# URL-ul va fi: http://localhost:8000/videos/nume_video.mp4
app.mount("/videos", StaticFiles(directory=str(VIDEO_OUTPUT_DIR)), name="videos")

# Modelul de date primit de la Frontend
class ChatRequest(BaseModel):
    message: str

# Variabilă globală pentru resurse (ca să nu le reîncărcăm la fiecare request)
resources = {}

# O clasă simplă care să mimeze structura Document
class SimpleDoc:
    def __init__(self, content, source="Manual", score=0.0):
        self.page_content = content
        self.metadata = {'source': source, 'score': score}

# ----------------------------------------------------------
#            1. RESURSE (Încărcare la Startup)
# ----------------------------------------------------------
def load_resources():
    print(f"[INFO] Se încarcă resursele...")
    
    try:
        index = faiss.read_index(f"{DB_FOLDER_PATH}/index.faiss")
        with open(f"{DB_FOLDER_PATH}/index.pkl", "rb") as f:
            texts = pickle.load(f)
        
        print(f"[INFO] Se încarcă modelul de embedding {EMBEDDING_MODEL_NAME}...")
        embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cuda") # GPU daca e disp

        print(f"[INFO] Se încarcă modelul de reranking {RERANKER_MODEL_NAME}...")
        reranker_model = CrossEncoder(RERANKER_MODEL_NAME, device="cpu")

        print(f"[INFO] Se încarcă Llama 3 8B...")
        llm = Llama(
            model_path=LLM_MODEL_PATH,
            n_gpu_layers=-1, 
            n_ctx=4096, 
            verbose=False
        )
        print("[SUCCESS] Toate resursele încărcate!")
        return {
            "index": index,
            "texts": texts,
            "embed_model": embed_model,
            "reranker_model": reranker_model,
            "llm": llm
        }
    except Exception as e:
        print(f"[EROARE CRITICĂ] Nu s-au putut încărca resursele: {e}")
        return None

# ----------------------------------------------------------
#            2. FUNCȚII DE LOGICĂ (RAG)
# ----------------------------------------------------------
def generate_expanded_queries(original_query, llm_instance):
    prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Ești un expert în căutare semantică.
Sarcina ta este să generezi 3 moduri diferite și mai clare de a formula întrebarea utilizatorului pentru a găsi cele mai bune rezultate într-un manual de geometrie.
Returnează DOAR cele 3 întrebări, separate prin linie nouă. Fără numerotare.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Întrebare originală: {original_query}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    try:
        output = llm_instance.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=128
        )
        text = output['choices'][0]['message']['content'].strip()
        queries = [q.strip() for q in text.split('\n') if q.strip()]
        clean_queries = [re.sub(r'^[\d\.\-\s]+', '', q) for q in queries]
        clean_queries.append(original_query)
        return list(set(clean_queries))
    except Exception as e:
        print(f"[WARN] Query Expansion failed: {e}")
        return [original_query]

def search_with_rerank(original_query, expanded_queries, index, texts, embed_model, reranker_model, final_k=5):
    candidate_indices = set()
    for q in expanded_queries:
        query_vec = embed_model.encode([q]).astype("float32")
        _, indices = index.search(query_vec, 10)
        for idx in indices[0]:
            if 0 <= idx < len(texts):
                candidate_indices.add(idx)
    
    if not candidate_indices:
        return []

    candidate_docs_text = [texts[i] for i in candidate_indices]
    pairs = [[original_query, doc_text] for doc_text in candidate_docs_text]
    scores = reranker_model.predict(pairs)
    ranked_results = sorted(zip(candidate_docs_text, scores), key=lambda x: x[1], reverse=True)
    
    final_docs = []
    for doc_text, score in ranked_results[:final_k]:
        source = "Manual"
        if doc_text.startswith("Title:"):
            parts = doc_text.split(":", 2)
            if len(parts) > 1:
                source = parts[1].strip()
        final_docs.append(SimpleDoc(doc_text, source, score))
    return final_docs

def create_prompt(context_docs, query):
    context = "\n\n".join([doc.page_content for doc in context_docs])
    prompt_template = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Ești un Asistent Universitar AI expert, prietenos și răbdător.

INSTRUCȚIUNI STRICTE:
1. Răspunsul tău trebuie să fie bazat **EXCLUSIV** pe textul de la secțiunea "CONTEXT" de mai jos. 
2. Structurează răspunsul în două părți clare:
   a) **Definiția/Răspunsul direct:** Preia informația exactă și riguroasă din text.
   b) **Explicația simplă:** Reformulează pe scurt, "ca pentru studenți", ca să fie ușor de înțeles.
3. Dacă informația nu există în context, spune sincer: "Nu am găsit această informație în materialele de curs."
4. Răspunde în limba română.
<|eot_id|><|start_header_id|>user<|end_header_id|>
CONTEXT:
{context}

ÎNTREBAREA STUDENTULUI:
{query}

RĂSPUNSUL TĂU (Structurat):
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    return prompt_template

def get_llm_response(prompt, llm_instance):
    if llm_instance is None: return None
    try:
        output = llm_instance.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return output['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"[EROARE] Eroare la generarea textului: {e}")
        return None

# ----------------------------------------------------------
#            3. GENERARE VIDEO (Adaptată pentru API)
# ----------------------------------------------------------
def generate_avatar_video_api(answer_text: str):
    """
    Generează videoul și returnează numele fișierului pentru URL.
    """
    text_for_tts = answer_text.replace("*", "").replace("#", "").replace("a)", "").replace("b)", "")
    
    # Folosim directoare temporare pentru procesare
    work_dir = BASE_DIR / "runtime" / "avatar_wav2lip"
    audio_path = work_dir / "answer.wav"
    
    # Generăm un nume unic bazat pe timp
    timestamp = int(time.time())
    final_filename = f"response_{timestamp}.mp4"
    final_path = VIDEO_OUTPUT_DIR / final_filename

    print("\n[INFO] Generare TTS (Piper)...")
    try:
        tts_piper(text_for_tts, str(audio_path))
    except Exception as e:
        print(f"[EROARE TTS] {e}")
        return None

    print("\n[INFO] Generare avatar cu Wav2Lip...")
    try:
        # Wav2Lip salvează într-un loc fix, noi trebuie să luăm rezultatul
        temp_out_dir = work_dir / "temp_vid"
        temp_out_dir.mkdir(parents=True, exist_ok=True)

        wav2lip_generate_video(
            image_path=str(BASE_DIR / USER_IMAGE),
            audio_path=str(audio_path),
            output_dir=str(temp_out_dir),
            wav2lip_repo=str(BASE_DIR / WAV2LIP_REPO) 
        )
        
        # Căutăm ultimul video generat în temp și îl mutăm în public
        generated = list(temp_out_dir.glob("*.mp4"))
        if generated:
            latest = max(generated, key=lambda p: p.stat().st_mtime)
            shutil.move(str(latest), str(final_path))
            return final_filename
    except Exception as e:
        print(f"[EROARE Wav2Lip] {e}")
        return None
    return None

# ----------------------------------------------------------
#            4. ENDPOINTS API
# ----------------------------------------------------------

@app.on_event("startup")
def startup_event():
    global resources
    res = load_resources()
    if res:
        resources = res
    else:
        print("[CRITICAL] Nu s-au putut încărca modelele. API-ul nu va funcționa corect.")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint principal: Primește text, returnează răspuns + link video.
    """
    if not resources:
        raise HTTPException(status_code=500, detail="Modelele AI nu sunt încărcate.")
    
    query_original = request.message
    query_normalized = unidecode(query_original)
    
    # 1. Query Expansion
    print(f"[API] Expandare interogare: {query_original}")
    expanded_queries = generate_expanded_queries(query_normalized, resources['llm'])
    
    # 2. Search & Rerank
    print(f"[API] Căutare...")
    context_docs = search_with_rerank(
        query_normalized,
        expanded_queries,
        resources['index'],
        resources['texts'],
        resources['embed_model'],
        resources['reranker_model'],
        final_k=5
    )

    sources_list = []
    if context_docs:
        sources_list = list(set([d.metadata['source'] for d in context_docs]))
        prompt = create_prompt(context_docs, query_original)
        
        # 3. Generate Text Response
        print(f"[API] Generare răspuns text...")
        response_text = get_llm_response(prompt, resources['llm'])
        
        # 4. Generate Video
        video_url = None
        if response_text:
            # Extragem doar explicația simplă pentru video ca să fie mai rapid
            text_pentru_avatar = response_text
            parts = re.split(r"\*\*Explicați[ea] simplă:\*\*", response_text, maxsplit=1)
            if len(parts) > 1:
                text_pentru_avatar = parts[1].strip()
            
            print(f"[API] Generare video...")
            video_filename = generate_avatar_video_api(text_pentru_avatar)
            
            if video_filename:
                # Construim URL-ul complet la care frontend-ul poate accesa video-ul
                video_url = f"http://localhost:8000/videos/{video_filename}"
    else:
        response_text = "Nu am găsit informații relevante în manual pentru această întrebare."
        video_url = None

    return {
        "text": response_text,
        "video_url": video_url,
        "sources": sources_list
    }

# ----------------------------------------------------------
#            5. MAIN (Start Server)
# ----------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)