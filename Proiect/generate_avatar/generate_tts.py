import subprocess
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent

PIPER_MODEL = CURRENT_DIR / "ro_RO-mihai-medium.onnx"

def tts_piper(text, output_file="answer.wav"):
    cmd = [
        "piper",
        "--model", str(PIPER_MODEL),
        "--output_file", output_file,
    ]
    subprocess.run(cmd, input=text.encode("utf-8"), check=True)
    return output_file
