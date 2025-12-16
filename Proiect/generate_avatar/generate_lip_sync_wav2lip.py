import os
import sys
import subprocess
import cv2
import soundfile as sf
from pathlib import Path

import numpy as np
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'int'):
    np.int = np.int_
# -----------------------------------

def get_audio_duration(audio_path):
    try:
        f = sf.SoundFile(audio_path)
        return len(f) / f.samplerate
    except Exception as e:
        print(f"[WARN] Nu pot citi durata audio: {e}. Folosesc 5s default.")
        return 5.0

def create_static_video(image_path, output_video, duration, fps=25):
    image_path = Path(image_path)
    output_video = Path(output_video)

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Nu pot citi imaginea: {image_path}")

    height, width, _ = frame.shape
    
    # Codec compatibil
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    total_frames = int(duration * fps) + 1
    for _ in range(total_frames):
        out.write(frame)

    out.release()

def wav2lip_generate_video(
    image_path: str,
    audio_path: str,
    output_dir: str,
    wav2lip_repo: str = None 
) -> str:
    
    # 1. Determinăm căile
    current_dir = Path(__file__).resolve().parent # Proiect/generate_avatar
    project_root = current_dir.parent             # Proiect/
    
    # Dacă nu primim repo-ul ca argument, îl căutăm în ../external/Wav2Lip
    if wav2lip_repo is None:
        wav2lip_repo_path = project_root / "external" / "Wav2Lip"
    else:
        wav2lip_repo_path = Path(wav2lip_repo)

    if not wav2lip_repo_path.exists():
        raise FileNotFoundError(f"Folderul Wav2Lip nu există la: {wav2lip_repo_path}")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    static_video = output_dir / "static_avatar.mp4"
    output_video = output_dir / "avatar_lipsync.mp4"
    
    audio_path_obj = Path(audio_path).resolve()
    
    # 2. Durata și Video Static
    duration = get_audio_duration(str(audio_path_obj))
    create_static_video(image_path, static_video, duration)

    # 3. Checkpoint
    checkpoint_path = wav2lip_repo_path / "checkpoints" / "wav2lip_gan.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint lipsă: {checkpoint_path}")

    # 4. Comanda
    # Folosim sys.executable pentru a rula cu același python din venv
    inference_script = current_dir / "inference_unified.py"
    
    cmd = [
        sys.executable,
        str(inference_script),
        "--face", str(static_video),
        "--audio", str(audio_path_obj),
        "--outfile", str(output_video),
        "--checkpoint_path", str(checkpoint_path),
        "--static", "True",
        "--pads", "0", "10", "0", "0"
    ]

    print(f"[Wav2Lip] Rulare inference din: {wav2lip_repo_path}")
    
    # Important: Rulăm comanda din folderul Wav2Lip ca să găsească importurile interne
    subprocess.run(cmd, cwd=str(wav2lip_repo_path), check=True)

    return str(output_video)