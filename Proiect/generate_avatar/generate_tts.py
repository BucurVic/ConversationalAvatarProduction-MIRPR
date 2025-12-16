import subprocess
import os
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PIPER_MODEL = CURRENT_DIR / "ro_RO-mihai-medium.onnx"

def tts_piper(text, output_file="answer.wav"):
    if not PIPER_MODEL.exists():
        raise FileNotFoundError(f"[TTS] Modelul nu există la: {PIPER_MODEL}")

    output_file_str = str(Path(output_file).resolve())

    cmd = [
        "piper",
        "--model", str(PIPER_MODEL),
        "--output_file", output_file_str,
    ]
    
    try:
        subprocess.run(cmd, input=text.encode("utf-8"), check=True)
    except subprocess.CalledProcessError as e:
        print(f"[TTS Error] Piper a eșuat. Verifică dacă 'piper' este instalat în PATH.")
        raise e
    except FileNotFoundError:
        print("[TTS Error] Comanda 'piper' nu a fost găsită. Instalează piper-tts binary.")
        raise

    return output_file