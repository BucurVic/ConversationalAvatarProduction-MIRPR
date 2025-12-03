import os
import time
import pickle
import faiss
import numpy as np
import subprocess
from pathlib import Path
from unidecode import unidecode
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer 
import re

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
# Calea trebuie să fie exact cea din create_vector_store.py
DB_FOLDER_PATH = 'vectorstoretmp/' 
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
LLM_MODEL_PATH = "./models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

# O clasă simplă care să mimeze structura Document din LangChain
# (ca să nu modificăm funcția create_prompt)
class SimpleDoc:
    def __init__(self, content, source="Manual"):
        self.page_content = content
        self.metadata = {'source': source}

# ----------------------------------------------------------
#            1. RESURSE (Încărcare Manuală)
# ----------------------------------------------------------
def load_resources():
    print(f"[INFO] Se încarcă resursele...")
    
    # 1. Încărcăm Indexul FAISS
    try:
        index = faiss.read_index(f"{DB_FOLDER_PATH}/index.faiss")
        print("[SUCCESS] Index FAISS încărcat.")
    except Exception as e:
        print(f"[EROARE] Nu am putut citi index.faiss din {DB_FOLDER_PATH}: {e}")
        return None, None, None

    # 2. Încărcăm Textele (Pickle)
    try:
        with open(f"{DB_FOLDER_PATH}/index.pkl", "rb") as f:
            texts = pickle.load(f)
        print(f"[SUCCESS] {len(texts)} fragmente de text încărcate.")
    except Exception as e:
        print(f"[EROARE] Nu am putut citi index.pkl: {e}")
        return None, None, None

    # 3. Încărcăm Modelul de Embedding (SentenceTransformer)
    # Îl punem pe CPU ca să nu ocupe VRAM-ul necesar pentru Llama
    print(f"[INFO] Se încarcă modelul de embedding {EMBEDDING_MODEL_NAME}...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cuda")

    # 4. Încărcăm Llama 3 (Pe GPU)
    print(f"[INFO] Se încarcă Llama 3 8B...")
    try:
        llm = Llama(
            model_path=LLM_MODEL_PATH,
            n_gpu_layers=-1, # Totul pe GPU
            n_ctx=4096, 
            verbose=False
        )
        print("[SUCCESS] Llama 3 încărcat pe GPU.")
    except Exception as e:
        print(f"[EROARE] Nu s-a putut încărca Llama: {e}")
        return None, None, None

    return index, texts, embed_model, llm

# ----------------------------------------------------------
#            2. LOGICA DE RETRIEVAL MANUALĂ
# ----------------------------------------------------------
def search_manual(query, index, texts, embed_model, k=4):
    # 1. Vectorizăm întrebarea
    query_vec = embed_model.encode([query]).astype("float32")
    
    # 2. Căutăm în FAISS
    distances, indices = index.search(query_vec, k)
    
    # 3. Extragem textele corespunzătoare indecșilor găsiți
    found_docs = []
    for idx in indices[0]:
        if 0 <= idx < len(texts):
            # Creăm un obiect simplu compatibil cu restul codului
            # (create_vector_store-ul tău păstra doar textul, nu și sursa explicită per chunk,
            # deci punem o sursă generică sau încercăm să o deducem dacă e în text)
            content = texts[idx]
            # Încercare simplă de a extrage titlul dacă e la început (Title: ...)
            source = "Manual"
            if content.startswith("Title:"):
                parts = content.split(":", 2)
                if len(parts) > 1:
                    source = parts[1].strip()
            
            found_docs.append(SimpleDoc(content, source))
            
    return found_docs

# ----------------------------------------------------------
#            3. RAG & PROMPT (Neschimbat)
# ----------------------------------------------------------
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
#            4. AVATAR (Video)
# ----------------------------------------------------------
def generate_avatar_video_wav2lip(answer_text: str):
    text_for_tts = answer_text.replace("*", "").replace("#", "").replace("a)", "").replace("b)", "")
    base_dir = Path(__file__).resolve().parent
    work_dir = base_dir / "runtime" / "avatar_wav2lip"
    work_dir.mkdir(parents=True, exist_ok=True)
    audio_path = work_dir / "answer.wav"
    video_output_dir = work_dir / "video"
    video_output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[INFO] Generare TTS (Piper)...")
    try:
        tts_piper(text_for_tts, str(audio_path))
    except Exception as e:
        print(f"[EROARE TTS] {e}")
        return None

    print("\n[INFO] Generare avatar cu Wav2Lip...")
    try:
        wav2lip_generate_video(
            image_path=str(base_dir / USER_IMAGE),
            audio_path=str(audio_path),
            output_dir=str(video_output_dir),
            wav2lip_repo=str(base_dir / WAV2LIP_REPO) 
        )
    except Exception as e:
        print(f"[EROARE Wav2Lip] {e}")
        return None

    mp4_files = list(video_output_dir.glob("*.mp4"))
    if mp4_files:
        final_video = max(mp4_files, key=lambda p: p.stat().st_mtime)
        print(f"\n[SUCCESS] Video Wav2Lip generat: {final_video}")
        try:
            subprocess.run(["xdg-open", str(final_video)])
        except:
            pass
        return final_video
    return None

# ----------------------------------------------------------
#                       MAIN
# ----------------------------------------------------------
if __name__ == "__main__":
    # 1. Încărcăm resursele manual
    index, texts, embed_model, llm = load_resources()
    
    if not llm or not index:
        print("[EXIT] Resursele nu au putut fi încărcate.")
        exit()
        
    print("\n" + "="*60)
    print("🎓 TUTORE AI (Manual FAISS + Llama 3)")
    print(" Scrie 'exit' pentru a ieși.")
    print("="*60 + "\n")

    while True:
        query_original = input("\nÎntrebarea ta: ")
        
        if query_original.lower() in ['exit', 'quit']:
            break
            
        query_normalized = unidecode(query_original)
        
        # 2. Căutare Manuală (Fără LangChain)
        context_docs = search_manual(query_normalized, index, texts, embed_model, k=4)
        
        # Extragere surse (simplificat)
        surse = set([d.metadata['source'] for d in context_docs])

        # 3. Generare Răspuns
        prompt = create_prompt(context_docs, query_original)
        response_text = get_llm_response(prompt, llm)
        
        if response_text:
            print("\n" + "="*60)
            print("🎓 RĂSPUNS GENERAT:")
            print("-" * 60)
            print(response_text.strip())
            print("-" * 60)
            print("📚 SURSE POSIBILE:", list(surse)[:3])
            print("="*60 + "\n")
            
            # 4. Video (Extragem doar explicația simplă dacă se poate)
            text_pentru_avatar = response_text
            parts = re.split(r"\*\*Explicați[ea] simplă:\*\*", response_text, maxsplit=1)
            if len(parts) > 1:
                text_pentru_avatar = parts[1].strip()

            print(f"[INFO] Generare video...")
            generate_avatar_video_wav2lip(text_pentru_avatar)