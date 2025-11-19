import os
import time
import subprocess
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from unidecode import unidecode
from llama_cpp import Llama 

try:
    from scripts.config import (
        SADTALKER_REPO,
        USER_IMAGE,
    )
    # Importăm funcția de TTS a colegului
    from generate_avatar.generate_tts import tts_piper
except ImportError:
    print("[WARN] Nu s-au putut importa modulele de Avatar (scripts.config). Se folosesc valori default.")
    SADTALKER_REPO = "SadTalker"
    USER_IMAGE = "input_img.jpg"
  

DB_FAISS_PATH = 'vectorstore/'
EMBEDDING_MODEL_NAME = "thenlper/gte-small"

LLM_MODEL_PATH = "./models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"




def load_db():
    """Încarcă baza de date vectorială FAISS."""
    print(f"[INFO] Se încarcă baza de date...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

def load_llm():
    """Încarcă modelul Llama 3 8B folosind llama-cpp-python."""
    print(f"[INFO] Se încarcă modelul Llama 3 8B din: {LLM_MODEL_PATH}...")
    try:
        llm = Llama(
            model_path=LLM_MODEL_PATH,
            n_gpu_layers=-1, 
            n_ctx=4096, 
            verbose=False
        )
        print("[SUCCESS] Modelul Llama 3 a fost încărcat.")
        return llm
    except Exception as e:
        print(f"[EROARE FATALĂ] Nu s-a putut încărca modelul: {e}")
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

def get_llm_response(prompt, llm_instance):
    """Trimite promptul către instanța Llama (local)."""
    if llm_instance is None: return None
    
    try:
        start_time = time.time()
        
        output = llm_instance.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        end_time = time.time()
        
        # La Llama 3 nu avem nevoie de filtrarea complexă de la GPT-OSS (fără <|channel|>)
        clean_text = output['choices'][0]['message']['content']
        
        print(f"[INFO] Generare text finalizată în {end_time - start_time:.2f} secunde.")
        return clean_text

    except Exception as e:
        print(f"[EROARE] Eroare la generarea răspunsului text: {e}")
        return None



def generate_avatar_video(answer_text: str):
    """
    Pipeline integrat:
    - generează audio cu Piper (în venv principal)
    - apelează SadTalker prin Python 3.10 (venv310) pentru lip-sync
    """
    text_for_tts = answer_text.replace("*", "").replace("#", "").replace("a)", "").replace("b)", "")

    base_dir = Path(__file__).resolve().parent
    work_dir = base_dir / "runtime" / "avatar"
    work_dir.mkdir(parents=True, exist_ok=True)

    audio_path = work_dir / "answer.wav"
    video_output_dir = work_dir / "video"
    video_output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[INFO] Generare TTS (Piper)...")
    try:
        tts_piper(text_for_tts, str(audio_path))
        print(f"[INFO] Audio generat: {audio_path}")
    except Exception as e:
        print(f"[EROARE TTS] Nu s-a putut genera audio: {e}")
        return None

    print("\n[INFO] Generare avatar cu SadTalker (Python 3.10)...")

    python310 = base_dir / "venv310" / "bin" / "python"
    run_sadtalker_script = base_dir / "generate_avatar" / "run_sadtalker.py"

    if not python310.exists():
        print(f"[EROARE] Nu am găsit mediul virtual pentru SadTalker la: {python310}")
        print("Verifică dacă ai copiat folderul venv310 de la coleg.")
        # Continuăm fără video, doar text
        return None

    cmd = [
        str(python310),
        str(run_sadtalker_script),
        "--image",
        str((base_dir / USER_IMAGE).resolve()) if not Path(USER_IMAGE).is_absolute() else USER_IMAGE,
        "--audio",
        str(audio_path.resolve()),
        "--output",
        str(video_output_dir.resolve()),
        "--repo",
        str((base_dir / SADTALKER_REPO).resolve()) if not Path(SADTALKER_REPO).is_absolute() else SADTALKER_REPO,
    ]

    env = os.environ.copy()
    env["PATH"] = f"{base_dir / 'venv310' / 'bin'}:" + env.get("PATH", "")

    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"[EROARE SADTALKER] Generarea video a eșuat: {e}")
        return None

    # 3. Caută ultimul .mp4 generat în folderul video
    mp4_files = list(video_output_dir.glob("*.mp4"))
    final_video = None
    if mp4_files:
        final_video = max(mp4_files, key=lambda p: p.stat().st_mtime)

    if final_video:
        print(f"\n[SUCCESS] Video generat cu succes: {final_video}")
    else:
        print("\n[AVERTISMENT] Nu am găsit niciun .mp4 în folderul video.")

    return final_video



#legatura creier cu avatar mai jos