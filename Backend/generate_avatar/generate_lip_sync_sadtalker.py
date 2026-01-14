import os
import sys
import subprocess
from pathlib import Path

# NOTĂ: Nu mai avem nevoie de hack-urile pentru Numpy (np.float, VisibleDeprecationWarning)
# deoarece venv-ul dedicat va avea versiunile corecte native!

def sadtalker_generate_video(
    image_path: str,
    audio_path: str,
    output_dir: str,
    sadtalker_repo: str = None
) -> str:
    
    # 1. Determinăm căile
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    
    # Calea către folderul SadTalker
    if sadtalker_repo is None:
        sadtalker_repo_path = project_root / "external" / "SadTalker"
    else:
        sadtalker_repo_path = Path(sadtalker_repo)

    if not sadtalker_repo_path.exists():
        raise FileNotFoundError(f"Folderul SadTalker nu există la: {sadtalker_repo_path}")

    # --- MODIFICAREA CRITICĂ: CALEA CĂTRE PYTHON-UL DEDICAT ---
    # Căutăm executabilul python din venv-ul secundar
    sadtalker_python = sadtalker_repo_path / "venv_sadtalker" / "Scripts" / "python.exe"

    if not sadtalker_python.exists():
        raise FileNotFoundError(
            f"Nu găsesc Python-ul dedicat pentru SadTalker la: {sadtalker_python}\n"
            "Te rog creează venv-ul: cd external/SadTalker && python -m venv venv_sadtalker && pip install ..."
        )
    # ----------------------------------------------------------

    image_path_obj = Path(image_path).resolve()
    audio_path_obj = Path(audio_path).resolve()
    output_dir_obj = Path(output_dir).resolve()
    output_dir_obj.mkdir(parents=True, exist_ok=True)

    inference_script = sadtalker_repo_path / "inference.py"
    if not inference_script.exists():
        raise FileNotFoundError(f"Nu găsesc inference.py în: {sadtalker_repo_path}")

    # 2. Construim comanda folosind PYTHON-ul DEDICAT
    cmd = [
        str(sadtalker_python),  # <--- Folosim python.exe din venv_sadtalker
        str(inference_script),
        "--driven_audio", str(audio_path_obj),
        "--source_image", str(image_path_obj),
        "--result_dir", str(output_dir_obj),
        
        # Argumente 3D / Calitate
        "--preprocess", "full",
        "--enhancer", "gfpgan",
        "--background_enhancer", "gfpgan",
        "--expression_scale", "1.0",
        "--size", "256",
        "--batch_size", "1"
    ]
    
    # 3. Configurare Mediu
    env = os.environ.copy()
    # Adăugăm folderul SadTalker în PYTHONPATH pentru acel proces
    env["PYTHONPATH"] = str(sadtalker_repo_path) + os.pathsep + env.get("PYTHONPATH", "")
    
    # Important: Trebuie să scoatem referințele către venv-ul PRINCIPAL din PATH
    # pentru a nu contamina venv-ul secundar.
    # (Resetăm PATH-ul pentru procesul copil să favorizeze noul venv, opțional dar recomandat)
    # Totuși, simpla apelare a lui python.exe din Scripts setează de obicei mediul corect.

    print(f"[SadTalker] Rulare cu Python izolat: {sadtalker_python}")

    try:
        subprocess.run(
            cmd, 
            cwd=str(sadtalker_repo_path),
            env=env, 
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"[SadTalker Error] Execuția a eșuat. Verifică logurile din consola SadTalker.")
        raise e

    # 4. Găsire video rezultat
    generated_videos = list(output_dir_obj.rglob("*.mp4"))
    valid_videos = [v for v in generated_videos if "concat" not in v.name]
    
    if not valid_videos:
        raise RuntimeError("SadTalker (venv izolat) a rulat, dar nu a generat video.")

    newest_video = max(valid_videos, key=os.path.getmtime)
    return str(newest_video)