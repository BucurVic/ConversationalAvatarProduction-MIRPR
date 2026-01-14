import os
import sys

# --- FIX CRITIC PENTRU CRASH/SEGFAULT PE MAC/LINUX ---
# Acestea trebuie să fie primele linii!
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import time
import warnings
import gc
import psutil
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- SETUP IMPORTURI DIN RĂDĂCINĂ ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import pickle
import faiss
import numpy as np
import shutil
import re
import torch
from unidecode import unidecode
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, CrossEncoder 

# --- IMPORTURI API ---
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# --- IMPORTURI PROIECT ---
try:
    from scripts.config import (
        DB_FAISS_PATH,
        EMBEDDING_MODEL_NAME,
        RERANKER_MODEL_NAME,
        LLM_MODEL_PATH,
        USER_IMAGE,
        WAV2LIP_REPO,
        SADTALKER_REPO,
        VIDEO_OUTPUT_DIR
    )
    from generate_avatar.generate_tts import tts_piper
    from generate_avatar.generate_lip_sync_wav2lip import wav2lip_generate_video
    from generate_avatar.generate_lip_sync_sadtalker import sadtalker_generate_video
    
    Path(VIDEO_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
except ImportError as e:
    print(f"[EROARE FATALĂ] Nu pot importa modulele proiectului: {e}")
    sys.exit(1)

CURRENT_VIDEO_MODEL = "wav2lip"

# --- UTILS MEMORIE (ACTUALIZAT CU MPS/CUDA) ---
def log_memory(stage="Status"):
    """
    Funcție de monitorizare detaliată (RAM + VRAM).
    """
    process = psutil.Process(os.getpid())
    ram_usage_gb = process.memory_info().rss / (1024 ** 3) # Conversie în GB
    
    vram_usage_gb = 0.0
    device_name = "CPU"
    
    if torch.cuda.is_available():
        vram_usage_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        device_name = "NVIDIA GPU"
    elif torch.backends.mps.is_available():
        try:
            vram_usage_gb = torch.mps.current_allocated_memory() / (1024 ** 3)
            device_name = "Apple MPS"
        except:
            device_name = "Apple MPS (No stats)"

    print(f"\n📊 [{stage}]")
    print(f"   RAM Proces (System): {ram_usage_gb:.2f} GB")
    if device_name != "CPU":
        print(f"   VRAM ({device_name}): {vram_usage_gb:.2f} GB")
    print("-" * 30)

# --- DEFINIȚII CLASE ---
class ChatRequest(BaseModel):
    message: str

class SimpleDoc:
    def __init__(self, content, source="Manual", score=0.0):
        self.page_content = content
        self.metadata = {'source': source, 'score': score}

resources = {}

def get_device_type():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

# ----------------------------------------------------------
#            1. RESURSE (OPTIMIZARE INFRASTRUCTURĂ)
# ----------------------------------------------------------
def load_resources():
    print(f"\n[BOOT] Se încarcă resursele (Optimizat)...")
    log_memory("Start Boot")
    
    device = get_device_type()
    
    # OPTIMIZARE 1: Modelele auxiliare pe CPU.
    aux_device = "cpu" 
    print(f"[BOOT] Modele auxiliare (Embed/Rerank) vor rula pe: {aux_device}")

    try:
        # 1. Index FAISS
        index_path = Path(DB_FAISS_PATH) / "index.faiss"
        pkl_path = Path(DB_FAISS_PATH) / "index.pkl"
        
        if not index_path.exists():
            raise FileNotFoundError(f"Lipsă index FAISS: {index_path}")
            
        index = faiss.read_index(str(index_path))
        with open(str(pkl_path), "rb") as f:
            texts = pickle.load(f)
        
        # 2. Modele Embeddings & Reranker
        print(f"[BOOT] Embedding Model: {EMBEDDING_MODEL_NAME}")
        embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=aux_device)

        print(f"[BOOT] Reranker Model: {RERANKER_MODEL_NAME}")
        reranker_model = CrossEncoder(RERANKER_MODEL_NAME, device=aux_device)

        # 3. LLM (Llama 3)
        print(f"[BOOT] LLM Llama 3: {Path(LLM_MODEL_PATH).name}")
        
        n_gpu = -1 
        if device == "cpu": n_gpu = 0
        
        use_flash_attn = True if device == "mps" else False 

        # OPTIMIZARE 2: Parametri tehnici pentru Llama
        llm = Llama(
            model_path=LLM_MODEL_PATH,
            n_gpu_layers=n_gpu,     
            n_ctx=2048,           # OPTIMIZARE: 2048 vs 4096
            n_batch=256,          # OPTIMIZARE: Batch mic
            verbose=False,       
            flash_attn=use_flash_attn 
        )
        
        log_memory("Resurse Încărcate")
        
        # Curățare inițială
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if torch.backends.mps.is_available(): torch.mps.empty_cache()

        return {
            "index": index,
            "texts": texts,
            "embed_model": embed_model,
            "reranker_model": reranker_model,
            "llm": llm
        }
    except Exception as e:
        print(f"[EROARE CRITICĂ LA ÎNCĂRCARE] {e}")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global resources
    res = load_resources()
    if res: resources = res
    yield 
    resources.clear()
    gc.collect()
    print("[SHUTDOWN] Resurse eliberate.")

app = FastAPI(title="Face2Learn API", description="Backend RAG + Avatar Video", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/videos", StaticFiles(directory=str(VIDEO_OUTPUT_DIR)), name="videos")

# ----------------------------------------------------------
#            2. RAG LOGIC (CALITATE MAXIMĂ PĂSTRATĂ)
# ----------------------------------------------------------
def generate_expanded_queries(original_query, llm_instance):
    """
    Generează variații folosind un exemplu NEUTRU.
    """
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    Ești un motor de căutare semantică.
    Sarcina ta: Reformulează întrebarea utilizatorului în 2 moduri alternative (sinonime, cuvinte cheie).

    REGULI:
    1. Păstrează limba română.
    2. Nu numerota liniile.
    3. Nu pune introduceri. Doar cele 2 variații.

    EXEMPLU NEUTRU:
    User: "Cum funcționează fotosinteza?"
    Assistant:
    Procesul de producere a hranei la plante
    Transformarea luminii în energie chimică plante

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    User: "{original_query}"
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    try:
        output = llm_instance.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=64,
            stop=["<|eot_id|>"]
        )
        text = output['choices'][0]['message']['content'].strip()
        
        queries = []
        for q in text.split('\n'):
            clean_q = re.sub(r'^[\d\.\-\*\s]+', '', q).strip()
            if clean_q and len(clean_q) > 3:
                queries.append(clean_q)
        
        queries.append(original_query)
        return list(set(queries))

    except Exception as e:
        return [original_query]

def search_with_rerank(original_query, expanded_queries, index, texts, embed_model, reranker_model, final_k=3):
    candidate_indices = set()
    
    # 1. Căutare rapidă FAISS
    for q in expanded_queries:
        query_vec = embed_model.encode([q]).astype("float32")
        _, indices = index.search(query_vec, 5) 
        for idx in indices[0]:
            if 0 <= idx < len(texts):
                candidate_indices.add(idx)
    
    if not candidate_indices: return []

    candidate_indices = list(candidate_indices)[:15]

    candidate_docs_text = [texts[i] for i in candidate_indices]
    pairs = [[original_query, doc_text] for doc_text in candidate_docs_text]
    
    # 2. Rerank
    scores = reranker_model.predict(pairs)
    ranked_results = sorted(zip(candidate_docs_text, scores), key=lambda x: x[1], reverse=True)
    
    final_docs = []
    for doc_text, score in ranked_results[:final_k]:
        source = "Manual"
        if " - " in doc_text[:100]:
             parts = doc_text.split(" - ", 1)
             source = parts[0].strip()
        final_docs.append(SimpleDoc(doc_text, source, score))
    return final_docs

def get_llm_response(context_docs, query, llm_instance):
    """
    Prompt complet pentru personalitate.
    """
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    Ești un Asistent Universitar AI prietenos.

    SARCINĂ:
    Răspunde la întrebarea studentului bazându-te DOAR pe context.

    REGULI DE STIL:
    1. Nu folosi titluri, bullet points sau liste.
    2. Scrie un singur paragraf fluid.
    3. Începe cu definiția corectă, apoi fă o trecere naturală ("practic", "altfel spus") către o explicație simplă sau o analogie.
    4. Fii CONCIS (Maxim 3-4 fraze).

    EXEMPLU DE FORMAT (Biologie):
    Mitocondria este organitul celular responsabil cu respirația și producerea de energie sub formă de ATP. Practic, poți să te gândești la ea ca la o "uzină electrică" a celulei, care transformă nutrienții în combustibilul necesar supraviețuirii.

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    CONTEXT:
    {context}

    ÎNTREBARE STUDENT:
    {query}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    try:
        output = llm_instance.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, 
            max_tokens=256, 
            stop=["<|eot_id|>"]
        )
        return output['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"[LLM Error] {e}")
        return None

# ----------------------------------------------------------
#            3. VIDEO GENERATION (OPTIMIZARE MEMORIE)
# ----------------------------------------------------------
# ----------------------------------------------------------
#            HELPER: MUTARE SIGURĂ (WINDOWS ROBUSTNESS)
# ----------------------------------------------------------
def safe_move_video(source_path: str, dest_path: str, retries=3):
    """
    Încearcă să mute un fișier video cu mecanism de retry și fallback copy+delete.
    Rezolvă problema 'PermissionError' pe Windows când fișierul e încă folosit.
    """
    src = Path(source_path).resolve()
    dst = Path(dest_path).resolve()

    # 1. Verificăm dacă sursa există
    if not src.exists():
        print(f"[MOVE ERR] Sursa nu există: {src}")
        return False

    # 2. Dacă sursa e deja la destinație, nu facem nimic
    if src == dst:
        print("[MOVE INFO] Sursa este deja la destinație.")
        return True

    # 3. Ștergem destinația dacă există deja (overwrite)
    if dst.exists():
        try:
            os.remove(dst)
        except Exception as e:
            print(f"[MOVE WARN] Nu am putut șterge fișierul vechi de la destinație: {e}")

    # 4. Bucla de încercări
    for attempt in range(retries):
        try:
            # Încercare mutare atomică
            shutil.move(str(src), str(dst))
            print(f"[MOVE SUCCESS] Fișier mutat la: {dst}")
            return True
        except Exception as e:
            print(f"[MOVE RETRY] Încercarea {attempt + 1}/{retries} eșuată ({e}). Aștept...")
            time.sleep(1.0) # Așteptăm 1 secundă eliberarea fișierului
            
            # Fallback: Copy + Delete (mai robust pe Windows)
            try:
                shutil.copy2(str(src), str(dst))
                try: os.remove(src) # Curățăm originalul
                except: pass
                print(f"[MOVE SUCCESS] Fișier copiat (fallback) la: {dst}")
                return True
            except Exception as copy_err:
                print(f"[MOVE ERR] Copy fallback eșuat: {copy_err}")

    return False

# ----------------------------------------------------------
#            3. VIDEO GENERATION (LOGICA HIBRIDĂ)
# ----------------------------------------------------------
def generate_video_response(text_response):
    if not text_response: return None

    # Curățare memorie PRE-generare
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    if torch.backends.mps.is_available(): torch.mps.empty_cache()
    
    # Logging Model
    print(f"[VIDEO] Generare folosind motorul: {CURRENT_VIDEO_MODEL.upper()}")
    log_memory(f"Start Video ({CURRENT_VIDEO_MODEL})")

    # Procesare text TTS
    tts_text = text_response.replace("*", "").replace("#", "").replace("\n", " ").strip()
    if len(tts_text) > 400: 
        tts_text = tts_text[:400].rsplit('.', 1)[0] + "." 

    # Pregătire nume fișiere și căi
    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    audio_filename = f"audio_{current_time_str}.wav"
    video_filename = f"response_{current_time_str}.mp4"
    
    runtime_dir = Path(VIDEO_OUTPUT_DIR).parent
    audio_path = runtime_dir / audio_filename
    
    # ACEASTA este destinația finală unde API-ul se așteaptă să găsească fișierul
    final_video_path = Path(VIDEO_OUTPUT_DIR) / video_filename

    try:
        # 1. Generare Audio (Comun)
        tts_piper(tts_text, str(audio_path))
        
        generated_path = None

        # 2. RULARE MOTOR VIDEO
        if CURRENT_VIDEO_MODEL == "sadtalker":
            # SadTalker returnează o cale absolută dintr-un folder temporar
            generated_path = sadtalker_generate_video(
                image_path=USER_IMAGE,
                audio_path=str(audio_path),
                output_dir=str(Path(VIDEO_OUTPUT_DIR)), # SadTalker poate crea subfoldere aici
                sadtalker_repo=str(SADTALKER_REPO)
            )
            
        elif CURRENT_VIDEO_MODEL == "wav2lip":
            # Wav2Lip
            generated_path = wav2lip_generate_video(
                image_path=USER_IMAGE,
                audio_path=str(audio_path),
                output_dir=str(Path(VIDEO_OUTPUT_DIR)),
                wav2lip_repo=str(WAV2LIP_REPO)
            )

        # 3. FINALIZARE ȘI MUTARE
        # Curățare memorie POST-generare
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        if generated_path:
            print(f"[DEBUG] Video generat brut la: {generated_path}")
            
            # Folosim funcția robustă pentru a duce fișierul la destinația finală 'final_video_path'
            success = safe_move_video(generated_path, str(final_video_path))
            
            if success:
                # Verificăm fizic existența înainte de a returna numele
                if final_video_path.exists() and final_video_path.stat().st_size > 0:
                    return video_filename
                else:
                    print("[VIDEO ERR] Fișierul final are 0 bytes sau lipsește.")
                    return None
            else:
                print("[VIDEO ERR] Mutarea fișierului a eșuat după retries.")
                return None
        else:
            print(f"[VIDEO ERR] Motorul {CURRENT_VIDEO_MODEL} a returnat None.")
            return None

    except Exception as e:
        print(f"[VIDEO ERR] Excepție generală: {e}")
        return None
    finally:
        # Curățenie audio (mereu)
        if audio_path.exists():
            try: os.remove(audio_path)
            except: pass
# ----------------------------------------------------------
#            4. ENDPOINT CHAT
# ----------------------------------------------------------
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not resources:
        raise HTTPException(status_code=500, detail="Sistemul se încarcă...")
    
    query = request.message
    print(f"\n[USER] {query}")
    
    # OPTIMIZARE 5: Curățare preventivă
    gc.collect()
    
    t0 = time.time()
    
    # 1. Search 
    t_s = time.time()
    expanded = generate_expanded_queries(unidecode(query), resources['llm'])
    docs = search_with_rerank(
        unidecode(query), 
        expanded, 
        resources['index'], 
        resources['texts'], 
        resources['embed_model'], 
        resources['reranker_model']
    )
    search_time = time.time() - t_s
    
    response_text = "Nu am găsit informații relevante."
    sources = []
    video_url = None
    llm_time = 0.0

    # 2. Gen Text (Prompt FULL)
    if docs:
        sources = list(set([d.metadata['source'] for d in docs]))
        t_l = time.time()
        print("[API] Generare text Llama 3...")
        response_text = get_llm_response(docs, query, resources['llm'])
        llm_time = time.time() - t_l
        log_memory("După LLM (Gen Text)")
    
    # 3. Gen Video
    video_time = 0.0
    if docs and response_text:
        print("[API] Generare video...")
        t_v = time.time()
        vid_name = generate_video_response(response_text)
        video_time = time.time() - t_v
        if vid_name:
            video_url = f"http://localhost:8000/videos/{vid_name}"
        log_memory("După Video (Final)")

    total = time.time() - t0
    
    print(f"--- TIMING ---")
    print(f"Search: {search_time:.2f}s")
    print(f"LLM:    {llm_time:.2f}s")
    print(f"Video:  {video_time:.2f}s")
    print(f"Total:  {total:.2f}s")
    print(f"--------------")

    return {
        "text": response_text,
        "video_url": video_url,
        "sources": sources,
        "metrics": {
            "search": round(search_time, 2),
            "llm": round(llm_time, 2),
            "video": round(video_time, 2),
            "total": round(total, 2)
        }
    }

class ModelSettings(BaseModel):
    model_name: str  # Frontend-ul va trimite "sadtalker" sau "wav2lip"

@app.post("/settings/video-model")
async def set_video_model(settings: ModelSettings):
    global CURRENT_VIDEO_MODEL
    
    # Normalizăm input-ul (litere mici, fără spații)
    normalized_name = settings.model_name.lower().strip()
    
    # Validare strictă
    if normalized_name not in ["sadtalker", "wav2lip"]:
        raise HTTPException(status_code=400, detail="Model invalid. Alege 'sadtalker' sau 'wav2lip'.")
    
    # Actualizăm starea globală
    CURRENT_VIDEO_MODEL = normalized_name
    
    print(f"\n[SYSTEM] Model video schimbat pe: {CURRENT_VIDEO_MODEL.upper()}")
    
    return {
        "status": "success", 
        "current_model": CURRENT_VIDEO_MODEL,
        "message": f"Model activat: {CURRENT_VIDEO_MODEL}"
    }

# Endpoint opțional ca frontend-ul să știe ce e selectat la refresh
@app.get("/settings/video-model")
async def get_video_model():
    return {"current_model": CURRENT_VIDEO_MODEL}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)