import os
import time
import subprocess
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from unidecode import unidecode

# Import config global
from scripts.config import (
    DB_FAISS_PATH,
    EMBEDDING_MODEL_NAME,
    LM_STUDIO_URL,
    LLM_MODEL,
    SADTALKER_REPO,
    USER_IMAGE,
)

# TTS (Piper)
from generate_avatar.generate_tts import tts_piper


# ----------------------------------------------------------
#                  ÎNCĂRCARE BAZĂ VECTORIALĂ
# ----------------------------------------------------------
def load_db():
    print(f"Încărcarea modelului de embedding: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
    )

    print(f"Încărcarea bazei vectoriale din '{DB_FAISS_PATH}'...")

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    print("Baza de date a fost încărcată cu succes.")
    return db


# ----------------------------------------------------------
#              GENERARE PROMPT PENTRU RAG
# ----------------------------------------------------------
def create_prompt(context_docs, query):
    context = "\n\n---\n\n".join([doc.page_content for doc in context_docs])

    prompt_template = f"""
Ești un asistent AI specializat în cursul de geometrie. 
Răspunde la următoarea întrebare bazându-te **exclusiv** pe contextul oferit mai jos**.**
Textul contextului este în limba română, dar fără diacritice.
Răspunsul tău trebuie să fie în limba română (poți folosi diacritice).
Dacă răspunsul nu se află în context, spune: "Informatia nu a fost gasita in materialele de curs."

CONTEXT:
---
{context}
---

ÎNTREBARE:
{query}

RĂSPUNS:
"""
    return prompt_template


# ----------------------------------------------------------
#          GENERARE VIDEO AVATAR CU SADTALKER (3.10)
# ----------------------------------------------------------
def generate_avatar_video(answer_text: str):
    """
    Pipeline:
    - generează audio cu Piper (în venv principal, Py 3.13)
    - apelează SadTalker prin Python 3.10 (venv310) pentru lip-sync
    """

    base_dir = Path(__file__).resolve().parent
    work_dir = base_dir / "runtime" / "avatar"
    work_dir.mkdir(parents=True, exist_ok=True)

    audio_path = work_dir / "answer.wav"
    video_output_dir = work_dir / "video"
    video_output_dir.mkdir(parents=True, exist_ok=True)

    # 1. TTS cu Piper
    print("\n[INFO] Generare TTS (Piper)...")
    tts_piper(answer_text, str(audio_path))
    print(f"[INFO] Audio generat: {audio_path}")

    # 2. Rulează SadTalker în venv310
    print("\n[INFO] Generare avatar cu SadTalker (Python 3.10)...")

    python310 = base_dir / "venv310" / "bin" / "python"
    run_sadtalker_script = base_dir / "generate_avatar" / "run_sadtalker.py"

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
        str((base_dir / SADTALKER_REPO).resolve())
        if not Path(SADTALKER_REPO).is_absolute()
        else SADTALKER_REPO,
    ]

    env = os.environ.copy()
    # Punem venv310/bin pe PATH ca siguranță
    env["PATH"] = f"{base_dir / 'venv310' / 'bin'}:" + env.get("PATH", "")

    print("[run_rag] Comandă SadTalker:")
    print(" ", " ".join(cmd))

    subprocess.run(cmd, check=True, env=env)

    # 3. Caută ultimul .mp4 generat în folderul video
    mp4_files = list(video_output_dir.glob("*.mp4"))
    final_video = None
    if mp4_files:
        final_video = max(mp4_files, key=lambda p: p.stat().st_mtime)

    if final_video:
        print(f"\n[INFO] Video generat cu succes: {final_video}")
    else:
        print("\n[AVERTISMENT] Nu am găsit niciun .mp4 în folderul video.")

    return final_video


# ----------------------------------------------------------
#                        MAIN DEMO
# ----------------------------------------------------------
if __name__ == "__main__":

    response = (
        "Spațiul vectorial este o structură algebrică în care vectorii pot fi adunați "
        "și înmulțiți cu numere reale, respectând anumite reguli fundamentale."
    )

    import torch
    print(torch.backends.mps.is_available())


    if response:
        print("\n--- RĂSPUNSUL RAG ---")
        print(response)

        print("\n[INFO] Inițiere pipeline avatar...")
        generate_avatar_video(response)